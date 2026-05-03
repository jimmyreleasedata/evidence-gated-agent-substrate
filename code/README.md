# NeurIPS Benchmark Suite v1

Benchmark suite substrate for tool-using agent post-training experiments on
the evaluation environment. The repository is organized around four layers:

- `docs/specs/`: authoritative specifications for suite scope, trace schema,
  replay classes, and release policy.
- `runner/` and `trace/`: shared execution and telemetry substrate.
- `adapters/`: benchmark family adapters.
- `scripts/`, `reports/`, `plots/`: operational entrypoints and publication
  artifacts.

The implementation order follows `codex_master_execution_plan_neurips_evaluation_environment.md`.

## Real Backend Bootstrap

The repository now supports project-local bootstrapping of the official
BrowserGym and SWE-bench upstream repositories under `vendor/`.

```bash
./scripts/bootstrap_real_backends.sh
./scripts/validate_real_backends.sh
```

The generated local manifest lives at `manifests/local_paths.yaml` and is kept
out of git because it contains machine-specific absolute paths.
