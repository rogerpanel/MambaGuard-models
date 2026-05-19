# Expected Artefacts per Run

Quick reference for what each script writes to disk. Paths are relative to
`--output-dir` (or the default printed at the top of each script).

## `scripts.train`

```
<output-dir>/
  config.yaml              # full resolved training config
  train.log                # human-readable training log (timestamps + losses)
  metrics.json             # per-epoch train + val metrics
  best.ckpt                # best-by-val-macro-F1 checkpoint
  last.ckpt                # final-epoch checkpoint (for resume)
  lipschitz_history.csv    # per-step L_f estimates (from LipschitzMonitor)
  certified_radius.csv     # per-epoch certified radius (CertifiedRadiusMonitor)
```

## `scripts.evaluate`

```
<output-dir>/
  eval_metrics.json        # accuracy, macro-F1, per-class P/R/F1, AUROC
  confusion_matrix.csv     # NxN integer confusion matrix
```

## `scripts.certify`

```
<output-dir>/
  certificate.json         # L_f, V*, pi_defender, epsilon, hedge_avg_loss,
                           # certificate_lower_bound, horizon, n_actions
```

## `scripts.benchmark_latency`

```
benchmarks/results/
  latency.csv              # batch_size, p50, p95, p99, throughput
```

## `scripts.reproduce_table --table <name>`

```
benchmarks/results/
  main_table.csv           # for --table main
  ablation_table.csv       # for --table ablation
  certification_radii.csv  # for --table certification
  attack_table.csv         # for --table attacks
```

## `scripts.run_protocolbench`

```
<output-dir>/
  protocolbench_cases.csv      # one row per MCP test case
  protocolbench_summary.json   # per-category detection rate
```

## `scripts.prepare_data`

```
<output>/
  window_000000.pt         # one PyG Data object per window
  window_000001.pt
  ...
  manifest.json            # n_messages, n_agents, n_tools, label distribution
```

## Hash manifest

After a full reproduction, the maintainers ship a `sha256sums.txt` next to
each release tarball so you can verify byte-equivalence of the released
artefacts.
