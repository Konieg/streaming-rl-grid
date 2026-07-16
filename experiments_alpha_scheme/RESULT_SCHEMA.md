# Shared result schema and plots

Every formal Phase writes a schema-v1 `summary.json` and one compressed trace file per run.
The common top-level fields are `phase`, `subexperiment`, `task`, `protocol`, `runs`, and
`aggregates`. Each run contains immutable metadata, status, core metrics, continual-learning
analysis, diagnostics, and a relative `trace_file` path.

The core prediction diagnostic is named `online_squared_td_error`. It is δ², not a claim of
reference MSVE. A phase may add `reference_msve` only when it specifies and records an independent
reference value function.

Render any completed result bundle with:

```powershell
python -m experiments.plotting path\to\summary.json
```

Pass multiple summaries from the same task to compare subexperiments without rerunning baselines:

```powershell
python -m experiments.plotting path\to\fixed\summary.json path\to\adaptive\summary.json
```

The shared plots are `learning_curves.png`, `final_metrics.png`, `adaptation_retention.png`,
`a_probes.png`, and `alpha_dynamics.png`. The adaptation/retention file appears only after a run contains a mode change.
Phase-specific plots may be added, but must not replace these common outputs.
