#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
repo <- if (length(args) >= 1) normalizePath(args[[1]]) else normalizePath(".")
out_dir <- file.path(repo, "paper", "mlip_study", "figures")

jq_lines <- function(filter, path) {
  result <- system2(
    "jq",
    args = c("-r", shQuote(filter), shQuote(path)),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(result, "status")
  if (!is.null(status) && status != 0) stop(paste(result, collapse = "\n"))
  result
}

baseline <- file.path(
  repo, "artifacts", "novel_mlip_campaign", "pullback-50d09184", "baseline"
)
model_files <- sort(Sys.glob(file.path(
  baseline, "models", "static_initial", "round-0", "seed-*", "model_manifest.json"
)))
stopifnot(length(model_files) == 3)

trace_filter <- paste0(
  ".seed as $s | .training_records[] | ",
  "[$s, .epoch, .validation_force_component_mae_ev_per_a, ",
  ".training_loss, .learning_rate] | @tsv"
)
trace_lines <- unlist(lapply(model_files, function(path) jq_lines(trace_filter, path)))
trace <- read.delim(
  text = paste(trace_lines, collapse = "\n"), header = FALSE,
  col.names = c("seed", "epoch", "validation_force_mae", "training_loss", "learning_rate")
)
trace$seed <- factor(trace$seed, levels = c(42, 43, 44))
write.csv(trace, file.path(out_dir, "training_trace_source.csv"), row.names = FALSE)

zero_metrics <- file.path(baseline, "evaluations", "zero_shot", "validation", "metrics.json")
zero_mae <- as.numeric(jq_lines(".metrics.force_component_mae_ev_per_a", zero_metrics))
stopifnot(abs(zero_mae - 0.07298000828145236) < 1e-14)

p_trace <- ggplot(
  trace,
  aes(x = epoch, y = validation_force_mae, color = seed, shape = seed, linetype = seed)
) +
  geom_hline(
    yintercept = zero_mae, linewidth = 0.45, color = "grey25", linetype = "longdash"
  ) +
  geom_line(linewidth = 0.45) +
  geom_point(size = 1.25, stroke = 0.25) +
  annotate(
    "text", x = 19, y = zero_mae, label = "zero shot", hjust = 1,
    vjust = -0.55, size = 2.4, color = "grey20"
  ) +
  scale_color_manual(values = c("#0072B2", "#D55E00", "#009E73")) +
  scale_shape_manual(values = c(16, 17, 15)) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
  scale_x_continuous(breaks = c(0, 5, 10, 15, 19), limits = c(0, 19)) +
  scale_y_continuous(limits = c(0, 0.15), expand = expansion(mult = c(0, 0.03))) +
  labs(
    x = "Full training epoch",
    y = expression("Validation force-component MAE (eV " * ring(A)^-1 * ")"),
    color = "Seed", shape = "Seed", linetype = "Seed"
  ) +
  theme_bw(base_size = 8) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.2, color = "grey88"),
    legend.position = "top",
    legend.margin = margin(0, 0, 0, 0),
    legend.key.width = unit(0.8, "lines"),
    plot.margin = margin(2, 3, 2, 2)
  )
ggsave(
  file.path(out_dir, "validation_training_trace.pdf"), p_trace,
  width = 3.25, height = 2.25, units = "in", device = grDevices::pdf,
  family = "Times", useDingbats = FALSE
)
ggsave(
  file.path(out_dir, "validation_training_trace.png"), p_trace,
  width = 3.25, height = 2.25, units = "in", dpi = 600, bg = "white"
)

score_file <- file.path(baseline, "acquisition", "d0_scores.json")
score_filter <- paste0(
  "to_entries[] | [.key, ",
  ".value.mean_population_vector_disagreement_ev_per_a] | @tsv"
)
score_lines <- jq_lines(score_filter, score_file)
scores <- read.delim(
  text = paste(score_lines, collapse = "\n"), header = FALSE,
  col.names = c("record_id", "disagreement")
)

manifest_file <- file.path(repo, "data_registry", "datasets", "cu_phase2", "normalized_manifest.json")
group_lines <- jq_lines(
  ".configurations[] | [.config_id, .group_id] | @tsv", manifest_file
)
groups <- read.delim(
  text = paste(group_lines, collapse = "\n"), header = FALSE,
  col.names = c("record_id", "group_id")
)
split_file <- file.path(repo, "artifacts", "research_spec", "novel_mlip_ralph_split.json")
pool_ids <- jq_lines(".record_ids.acquisition_pool[]", split_file)
pool <- merge(scores, groups, by = "record_id", all.x = TRUE, sort = FALSE)
pool <- pool[pool$record_id %in% pool_ids, ]
stopifnot(nrow(pool) == 141, !any(is.na(pool$group_id)))

pool$family <- ifelse(
  grepl("^elastic_", pool$group_id), "Elastic\n(n=80)",
  ifelse(pool$group_id == "aimd_nvt_1000k", "AIMD 1000 K\n(n=40)",
    ifelse(pool$group_id == "vacancy_3000k", "Vacancy 3000 K\n(n=20)", "Surface\n(n=1)")
  )
)
pool$family <- factor(
  pool$family,
  levels = c("Elastic\n(n=80)", "AIMD 1000 K\n(n=40)", "Vacancy 3000 K\n(n=20)", "Surface\n(n=1)")
)
pool <- pool[order(pool$family, pool$record_id), ]
write.csv(pool, file.path(out_dir, "pool_disagreement_source.csv"), row.names = FALSE)

floor_force <- as.numeric(jq_lines(
  ".same_calculator_repeat.maximum_absolute_force_component_difference_ev_per_a",
  file.path(baseline, "calibration", "colab_same_session.json")
))
threshold <- max(10 * floor_force, 1e-6)
top36_min <- sort(pool$disagreement, decreasing = TRUE)[36]
stopifnot(sum(pool$disagreement > threshold) == 61)
stopifnot(abs(top36_min - 0.006854111555129643) < 1e-14)

line_data <- data.frame(
  label = factor(c("Numerical gate", "36th-largest score"),
                 levels = c("Numerical gate", "36th-largest score")),
  value = c(threshold, top36_min)
)
p_pool <- ggplot(pool, aes(x = family, y = disagreement)) +
  geom_hline(
    data = line_data,
    aes(yintercept = value, linetype = label),
    linewidth = 0.48, color = "grey18"
  ) +
  geom_jitter(
    position = position_jitter(width = 0.16, height = 0, seed = 20260712),
    size = 1.05, alpha = 0.68, color = "#0072B2", shape = 16
  ) +
  scale_y_log10(
    limits = c(2e-7, 2e-2),
    breaks = c(1e-7, 1e-5, 1e-3, 1e-2),
    labels = c(expression(10^-7), expression(10^-5), expression(10^-3), expression(10^-2))
  ) +
  scale_linetype_manual(values = c("dashed", "dotdash")) +
  labs(
    x = NULL,
    y = expression("Committee disagreement (eV " * ring(A)^-1 * ")"),
    linetype = NULL
  ) +
  theme_bw(base_size = 8) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.2, color = "grey88"),
    axis.text.x = element_text(size = 7),
    legend.position = "top",
    legend.margin = margin(0, 0, 0, 0),
    legend.key.width = unit(1.0, "lines"),
    plot.margin = margin(2, 3, 2, 2)
  )
ggsave(
  file.path(out_dir, "pool_disagreement.pdf"), p_pool,
  width = 3.75, height = 2.25, units = "in", device = grDevices::pdf,
  family = "Times", useDingbats = FALSE
)
ggsave(
  file.path(out_dir, "pool_disagreement.png"), p_pool,
  width = 3.75, height = 2.25, units = "in", dpi = 600, bg = "white"
)

deps <- c(
  paste("R", paste(R.version$major, R.version$minor, sep = ".")),
  paste("ggplot2", as.character(packageVersion("ggplot2")))
)
writeLines(deps, file.path(out_dir, "figure_dependencies.txt"))
