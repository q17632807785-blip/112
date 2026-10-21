"""G1-MST（预注册第 2.5 节）：掩码时空 Transformer，约 3 万参数。

主模型只用真实开发集训练，不用 sim_train。本模块可独立训练与推理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Estimate, GeometryInfo, Method, Series, fill_estimate_from_denoised
from .simple import B1

N_PTS = 59


def fourier_features(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """x (...,) → (..., 2*n_freq) sin/cos of 2^k π x。"""
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)


class G1MSTNet(nn.Module):
    def __init__(self, geo: GeometryInfo, d_model=32, n_layers=2, ff_width=64, dropout=0.1, fourier_freqs=8, no_temporal=False):
        super().__init__()
        self.d = d_model
        self.n_layers = n_layers
        self.ff_width = ff_width
        self.no_temporal = no_temporal
        xy = torch.tensor(geo.xy, dtype=torch.float32)
        self.register_buffer("xy", xy)
        # neighbor bias mask: +b for neighbors, -inf across hemifield
        hemi = np.sign(geo.xy[:, 1])
        hemi[geo.center] = 0
        cross = (hemi[:, None] * hemi[None, :]) < 0  # opposite signs
        nb = geo.neighbor.astype(bool)
        # attention bias base: 0; neighbors get learnable b; cross = -inf
        self.register_buffer("cross_mask", torch.tensor(cross, dtype=torch.bool))
        self.register_buffer("neighbor_mask", torch.tensor(nb, dtype=torch.bool))
        self.neighbor_bias = nn.Parameter(torch.zeros(1))

        self.y_emb = nn.Linear(2, d_model)          # [y/35, censored]
        self.xy_proj = nn.Linear(2 * fourier_freqs * 2, d_model)  # x and y each 2*n_freq
        self.t_proj = nn.Linear(2 * fourier_freqs, d_model)
        self.age_proj = nn.Linear(1, d_model)
        self.visit_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.n_freq = fourier_freqs

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict(dict(
                spatial_qkv=nn.Linear(d_model, 3 * d_model),
                temporal_qkv=nn.Linear(d_model, 3 * d_model),
                spatial_out=nn.Linear(d_model, d_model),
                temporal_out=nn.Linear(d_model, d_model),
                ff=nn.Sequential(
                    nn.Linear(d_model, ff_width), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(ff_width, d_model), nn.Dropout(dropout),
                ),
                n1=nn.LayerNorm(d_model),
                n2=nn.LayerNorm(d_model),
                n3=nn.LayerNorm(d_model),
            )))
        self.head = nn.Linear(d_model, 2)  # mu, log_sigma
        self.dropout = nn.Dropout(dropout)

    def encode_points(self, y, cens, t, age, masked):
        """y,cens,masked: (B,V,59); t,age: (B,V) → tokens (B,V,59,d) and visit tokens (B,V,d)."""
        B, V, _ = y.shape
        y_in = torch.stack([y / 35.0, cens.float()], dim=-1)
        y_in = y_in.masked_fill(masked.unsqueeze(-1), 0.0)
        tok = self.y_emb(y_in)
        # xy fourier once
        xf = fourier_features(self.xy[:, 0], self.n_freq)
        yf = fourier_features(self.xy[:, 1], self.n_freq)
        xy_f = torch.cat([xf, yf], dim=-1)  # (59, 4*n_freq) wait 2*n_freq each → 4*n_freq
        tok = tok + self.xy_proj(xy_f)[None, None, :, :]
        tf = fourier_features(t, self.n_freq)  # (B,V,2*n_freq)
        tok = tok + self.t_proj(tf)[:, :, None, :]
        tok = tok + self.age_proj(age.unsqueeze(-1) / 100.0)[:, :, None, :]
        tok = torch.where(masked.unsqueeze(-1), self.mask_token.view(1, 1, 1, -1).expand_as(tok), tok)
        vtok = self.visit_token.view(1, 1, -1).expand(B, V, -1) + self.t_proj(tf)
        return tok, vtok

    def spatial_attn(self, layer, tok, vtok):
        # tok (B,V,59,d), vtok (B,V,d) → concat visit as 60th
        B, V, P, d = tok.shape
        x = torch.cat([tok, vtok.unsqueeze(2)], dim=2)  # (B,V,60,d)
        x = x.reshape(B * V, 60, d)
        qkv = layer["spatial_qkv"](x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = d ** -0.5
        att = torch.bmm(q, k.transpose(1, 2)) * scale  # (BV,60,60)
        # bias on point-point block
        bias = torch.zeros(60, 60, device=x.device)
        bias[:59, :59] = torch.where(
            self.cross_mask,
            torch.tensor(-1e9, device=x.device),
            torch.where(self.neighbor_mask, self.neighbor_bias, torch.zeros((), device=x.device)),
        )
        att = att + bias.unsqueeze(0)
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.bmm(att, v)
        out = layer["spatial_out"](out).reshape(B, V, 60, d)
        return out[:, :, :59], out[:, :, 59]

    def temporal_attn(self, layer, tok, vtok):
        # tok (B,V,59,d): attention over V for each point; vtok over V
        B, V, P, d = tok.shape
        # points
        x = tok.permute(0, 2, 1, 3).reshape(B * P, V, d)
        qkv = layer["temporal_qkv"](x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = d ** -0.5
        att = torch.bmm(q, k.transpose(1, 2)) * scale
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out_p = layer["temporal_out"](torch.bmm(att, v)).reshape(B, P, V, d).permute(0, 2, 1, 3)
        # visit tokens
        qkv = layer["temporal_qkv"](vtok)
        q, k, v = qkv.chunk(3, dim=-1)
        att = torch.bmm(q, k.transpose(1, 2)) * scale
        att = torch.softmax(att, dim=-1)
        out_v = layer["temporal_out"](torch.bmm(att, v))
        return out_p, out_v

    def forward(self, y, cens, t, age, masked):
        tok, vtok = self.encode_points(y, cens, t, age, masked)
        for layer in self.layers:
            sp, sv = self.spatial_attn(layer, tok, vtok)
            tok = layer["n1"](tok + sp)
            vtok = layer["n1"](vtok + sv)
            if not self.no_temporal:
                tp, tv = self.temporal_attn(layer, tok, vtok)
                tok = layer["n2"](tok + tp)
                vtok = layer["n2"](vtok + tv)
            tok = layer["n3"](tok + layer["ff"](tok))
            vtok = layer["n3"](vtok + layer["ff"](vtok))
        out = self.head(tok)  # (B,V,59,2)
        mu, log_s = out[..., 0], out[..., 1]
        sigma = F.softplus(log_s) + 1e-3
        return mu, sigma


def censored_nll(mu, sigma, y, cens, mask):
    """只对被遮 token 计损失。"""
    # uncensored: -log N(y;mu,sigma); censored: -log Phi((0-mu)/sigma)
    z = (y - mu) / sigma
    nll_unc = 0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma) + z ** 2)
    # censored at 0: y* <= 0 → Phi((0-mu)/sigma)
    z0 = (0.0 - mu) / sigma
    # log Phi via log_ndtr
    from torch.distributions.normal import Normal
    nll_c = -Normal(0, 1).cdf(z0).clamp(min=1e-6).log()
    nll = torch.where(cens.bool(), nll_c, nll_unc)
    m = mask.bool()
    if m.sum() == 0:
        return mu.sum() * 0.0
    return nll[m].mean()


@dataclass
class G1MSTConfig:
    d_model: int = 32
    n_layers: int = 2
    ff_width: int = 64
    dropout: float = 0.1
    fourier_freqs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 16
    max_epochs: int = 400
    early_stop_patience: int = 30
    inference_K: int = 20
    mask_frac: float = 0.20
    seed: int = 20261019
    no_temporal: bool = False  # True → MST-S


class G1MST(Method):
    name = "G1-MST"
    needs_fit = True

    def __init__(self, geo: GeometryInfo, cfg: G1MSTConfig | None = None, device: str | None = None):
        self.geo = geo
        self.cfg = cfg or G1MSTConfig()
        self.name = "MST-S" if self.cfg.no_temporal else "G1-MST"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = G1MSTNet(
            geo,
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            ff_width=self.cfg.ff_width,
            dropout=self.cfg.dropout,
            fourier_freqs=self.cfg.fourier_freqs,
            no_temporal=self.cfg.no_temporal,
        ).to(self.device)
        self._b1 = B1(geo)
        self.fitted = False

    def count_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def _sample_mask(self, B, V, rng: torch.Generator):
        """等概率三种遮罩。返回 (B,V,59) bool。"""
        mode = torch.randint(0, 3, (B,), generator=rng, device=self.device)
        masked = torch.zeros(B, V, N_PTS, dtype=torch.bool, device=self.device)
        for i in range(B):
            m = int(mode[i])
            if m == 0:
                # random 20% tokens
                n = max(1, int(round(V * N_PTS * self.cfg.mask_frac)))
                flat = torch.randperm(V * N_PTS, generator=rng, device=self.device)[:n]
                masked[i].view(-1)[flat] = True
            elif m == 1:
                # whole visit
                v = int(torch.randint(0, V, (1,), generator=rng, device=self.device))
                masked[i, v, :] = True
            else:
                # whole location
                p = int(torch.randint(0, N_PTS, (1,), generator=rng, device=self.device))
                masked[i, :, p] = True
        return masked

    def fit(self, train_series: list[Series], val_series: list[Series] | None = None, max_epochs: int | None = None) -> dict:
        cfg = self.cfg
        max_epochs = max_epochs if max_epochs is not None else cfg.max_epochs
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # pad series to batches by grouping same length
        by_len: dict[int, list] = {}
        for s in train_series:
            by_len.setdefault(len(s.t), []).append(s)
        if val_series is None:
            # hold out 20% eyes
            rng = np.random.default_rng(cfg.seed)
            idx = rng.permutation(len(train_series))
            n_val = max(1, len(train_series) // 5)
            val_set = [train_series[i] for i in idx[:n_val]]
            train_set = [train_series[i] for i in idx[n_val:]]
            by_len = {}
            for s in train_set:
                by_len.setdefault(len(s.t), []).append(s)
        else:
            val_set = val_series
            train_set = train_series

        best_val = float("inf")
        best_state = None
        patience = 0
        history = []
        g = torch.Generator(device=self.device)
        g.manual_seed(cfg.seed)

        for epoch in range(max_epochs):
            self.net.train()
            total, ntok = 0.0, 0
            # iterate length groups
            keys = list(by_len.keys())
            rng_np = np.random.default_rng(cfg.seed + epoch)
            rng_np.shuffle(keys)
            for L in keys:
                group = by_len[L]
                order = rng_np.permutation(len(group))
                for start in range(0, len(group), cfg.batch_size):
                    batch = [group[i] for i in order[start:start + cfg.batch_size]]
                    y, cens, t, age = self._stack(batch)
                    masked = self._sample_mask(len(batch), L, g)
                    mu, sigma = self.net(y, cens, t, age, masked)
                    loss = censored_nll(mu, sigma, y, cens, masked)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * int(masked.sum())
                    ntok += int(masked.sum())
            train_loss = total / max(ntok, 1)
            val_loss = self._eval_loss(val_set, g)
            history.append(dict(epoch=epoch, train=train_loss, val=val_loss))
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  G1-MST epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f}", flush=True)
            if patience >= cfg.early_stop_patience:
                print(f"  early stop at epoch {epoch+1}", flush=True)
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fitted = True
        return dict(best_val=best_val, history=history, n_params=self.count_params())

    def _stack(self, batch: list[Series]):
        y = torch.tensor(np.stack([s.y for s in batch]), dtype=torch.float32, device=self.device)
        cens = torch.tensor(np.stack([s.censored for s in batch]), dtype=torch.bool, device=self.device)
        t = torch.tensor(np.stack([s.t for s in batch]), dtype=torch.float32, device=self.device)
        age = torch.tensor(np.stack([
            s.age if s.age is not None else np.zeros_like(s.t) for s in batch
        ]), dtype=torch.float32, device=self.device)
        return y, cens, t, age

    @torch.no_grad()
    def _eval_loss(self, series_list, g):
        self.net.eval()
        if not series_list:
            return float("inf")
        total, ntok = 0.0, 0
        by_len: dict[int, list] = {}
        for s in series_list:
            by_len.setdefault(len(s.t), []).append(s)
        for L, group in by_len.items():
            for start in range(0, len(group), self.cfg.batch_size):
                batch = group[start:start + self.cfg.batch_size]
                y, cens, t, age = self._stack(batch)
                masked = self._sample_mask(len(batch), L, g)
                mu, sigma = self.net(y, cens, t, age, masked)
                nll = censored_nll(mu, sigma, y, cens, masked)
                total += float(nll.item()) * int(masked.sum())
                ntok += int(masked.sum())
        return total / max(ntok, 1)

    @torch.no_grad()
    def denoise(self, series: Series) -> np.ndarray:
        self.net.eval()
        y, cens, t, age = self._stack([series])
        V = y.shape[1]
        acc = torch.zeros(V, N_PTS, device=self.device)
        cnt = torch.zeros(V, N_PTS, device=self.device)
        g = torch.Generator(device=self.device)
        g.manual_seed(self.cfg.seed)
        K = self.cfg.inference_K
        for k in range(K):
            # random 20% mask; ensure coverage
            masked = torch.zeros(1, V, N_PTS, dtype=torch.bool, device=self.device)
            n = max(1, int(round(V * N_PTS * self.cfg.mask_frac)))
            flat = torch.randperm(V * N_PTS, generator=g, device=self.device)[:n]
            masked.view(-1)[flat] = True
            mu, _ = self.net(y, cens, t, age, masked)
            acc += mu[0] * masked[0]
            cnt += masked[0].float()
        # tokens never masked: fall back to observed
        den = y[0].clone()
        m = cnt > 0
        den[m] = acc[m] / cnt[m]
        return den.cpu().numpy()


    @torch.no_grad()
    def denoise_batch(self, series_list: list[Series]) -> list[np.ndarray]:
        """同长度序列批量去噪；不同长度请分组后调用。"""
        if not series_list:
            return []
        return list(self.denoise_array(
            np.stack([s.y for s in series_list]),
            np.stack([s.censored for s in series_list]),
            np.stack([s.t for s in series_list]),
            np.stack([s.age if s.age is not None else np.zeros_like(s.t) for s in series_list]),
        ))

    @torch.no_grad()
    def denoise_array(self, y_np, cens_np, t_np, age_np) -> np.ndarray:
        """y (B,V,59) → denoised (B,V,59)。向量化遮罩，避免 Python 逐条 randperm。"""
        self.net.eval()
        y = torch.as_tensor(y_np, dtype=torch.float32, device=self.device)
        cens = torch.as_tensor(cens_np, dtype=torch.bool, device=self.device)
        t = torch.as_tensor(t_np, dtype=torch.float32, device=self.device)
        age = torch.as_tensor(age_np, dtype=torch.float32, device=self.device)
        B, V, _ = y.shape
        acc = torch.zeros(B, V, N_PTS, device=self.device)
        cnt = torch.zeros(B, V, N_PTS, device=self.device)
        K = self.cfg.inference_K
        n_mask = max(1, int(round(V * N_PTS * self.cfg.mask_frac)))
        g = torch.Generator(device=self.device)
        g.manual_seed(self.cfg.seed)
        # (K,B,V*59) random scores → top n_mask
        for k in range(K):
            scores = torch.rand(B, V * N_PTS, generator=g, device=self.device)
            # smallest n_mask scores masked
            thresh = torch.kthvalue(scores, n_mask, dim=1).values.unsqueeze(1)
            masked = (scores <= thresh).view(B, V, N_PTS)
            # ensure exactly coverage: if ties, OK
            mu, _ = self.net(y, cens, t, age, masked)
            acc += mu * masked
            cnt += masked.float()
        den = y.clone()
        m = cnt > 0
        den[m] = acc[m] / cnt[m]
        return den.cpu().numpy()

    def estimate(self, series: Series) -> Estimate:
        if not self.fitted or len(series.t) < 2:
            est = self._b1.estimate(series)
            est.failed = True
            est.fallback = "B1"
            return est
        try:
            den = self.denoise(series)
            return fill_estimate_from_denoised(series.t, den, self.geo.cluster)
        except Exception:
            est = self._b1.estimate(series)
            est.failed = True
            est.fallback = "B1"
            return est

    def save(self, path: str):
        torch.save(dict(state=self.net.state_dict(), cfg=self.cfg.__dict__, fitted=self.fitted), path)

    def load(self, path: str):
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(blob["state"])
        self.fitted = blob.get("fitted", True)
