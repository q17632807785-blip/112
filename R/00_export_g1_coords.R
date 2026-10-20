# 导出 visualFields::locmaps$g1 坐标（Octopus G1，右眼视野坐标，+x 为颞侧）
.libPaths(c("/workspace/grapeA/Rlib", .libPaths()))
library(visualFields)
args <- commandArgs(trailingOnly = TRUE); out <- if (length(args)) args[1] else "data/visualFields_g1_coord.csv"
co <- locmaps$g1$coord
co$vf_loc <- as.integer(rownames(co))
write.csv(co[, c("vf_loc", "x", "y")], out, row.names = FALSE)
cat("visualFields", as.character(packageVersion("visualFields")), "G1 n =", nrow(co), "\n")
