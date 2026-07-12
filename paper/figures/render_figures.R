#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
repo <- if (length(args) >= 1) normalizePath(args[[1]]) else normalizePath(".")
data_dir <- file.path(repo, "artifacts/auto_research/free-ralph-trace-audit/analysis")
out_dir <- file.path(repo, "paper/figures/generated")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(ggplot2))

theme_paper <- theme_minimal(base_size = 8, base_family = "Helvetica") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 7),
    legend.position = "none",
    plot.margin = margin(4, 6, 4, 6)
  )

summary <- read.csv(file.path(data_dir, "fault_detection_summary.csv"), check.names = FALSE)
summary$method <- factor(summary$method, levels = c("Manifest only", "Semantic grader"))
seed_level <- read.csv(file.path(data_dir, "seed_level_fault_detection.csv"), check.names = FALSE)
seed_level$method <- factor(seed_level$method, levels = c("Manifest only", "Semantic grader"))
seed_level$seed_index <- ave(seed_level$seed, seed_level$method, FUN = rank)
seed_level$x <- as.numeric(seed_level$method) + c(-0.09, -0.03, 0.03, 0.09)[seed_level$seed_index]

p_recall <- ggplot(summary, aes(x = method, y = recall, fill = method)) +
  geom_col(width = 0.62, color = "black", linewidth = 0.35) +
  geom_point(
    data = seed_level,
    aes(x = x, y = recall),
    inherit.aes = FALSE,
    shape = 21,
    size = 2.0,
    stroke = 0.45,
    fill = "white",
    color = "black"
  ) +
  geom_text(aes(label = sprintf("%d/%d", detected, corrupt_total)), vjust = -0.45, size = 2.7) +
  scale_fill_manual(values = c("Manifest only" = "#999999", "Semantic grader" = "#0072B2")) +
  scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
  labs(x = NULL, y = "Fault-detection recall") +
  theme_paper

cairo_pdf(file.path(out_dir, "fault_detection_recall.pdf"), width = 3.25, height = 2.15)
print(p_recall)
dev.off()

nodes <- data.frame(
  x = 1:5,
  y = 1,
  label = c("Proposal", "Execution", "Independent\nevaluation", "Decision", "Evidence\nrelease")
)
edges <- data.frame(x = 1:4 + 0.28, xend = 2:5 - 0.28, y = 1, yend = 1)
p_flow <- ggplot() +
  geom_segment(
    data = edges,
    aes(x = x, xend = xend, y = y, yend = yend),
    arrow = arrow(length = unit(0.09, "inches")),
    linewidth = 0.4,
    color = "#444444"
  ) +
  geom_label(
    data = nodes,
    aes(x = x, y = y, label = label),
    size = 2.6,
    linewidth = 0.3,
    fill = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"),
    color = "black"
  ) +
  annotate("text", x = 3, y = 0.56, label = "hashes + typed identities + re-derived constraints", size = 2.6) +
  coord_cartesian(xlim = c(0.55, 5.45), ylim = c(0.42, 1.30), clip = "off") +
  theme_void(base_family = "Helvetica")

cairo_pdf(file.path(out_dir, "evidence_workflow.pdf"), width = 6.75, height = 1.45)
print(p_flow)
dev.off()
