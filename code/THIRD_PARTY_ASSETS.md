# THIRD_PARTY_ASSETS

This file records the major third-party assets that are actually referenced by
the NeurIPS benchmark-suite repository and release artifacts. Values are only
filled when they can be grounded in repository files, vendored metadata, or
generated artifact metadata already present in this checkout.

| Asset | Asset type | Upstream paper/project URL | Version / commit / access date used | License or terms of use | Re-packaged? | Notes / restrictions |
|---|---|---|---|---|---|---|
| BrowserGym | benchmark environment framework | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | `0.14.3`; vendored commit `9e779f087de9a65668b6974d11f9ce9816026e96`; commit date `2026-03-17T15:45:10-04:00` | Apache-2.0 | yes | Vendored under `vendor/browsergym`; used by the real-backend bootstrap path and BrowserGym integration packages. |
| WebArena Verified (via BrowserGym integration) | benchmark-family asset | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | BrowserGym integration package inherits BrowserGym version `0.14.3`; upstream project page accessed `2026-04-10` | Apache-2.0 | yes | Offline replay is the primary release path. Live site-backed use still requires `NIPS_BENCH_WEBARENA_BASE_URL` and `NIPS_BENCH_WEBARENA_STORAGE_STATE`. |
| MiniWoB++ | benchmark-family asset | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | upstream project page accessed `2026-04-10`; local bundle references BrowserGym MiniWoB integration `0.14.3` | MIT | no | The repo exposes a MiniWoB family adapter and evidence surface, but does not vendor the upstream MiniWoB++ environment as a standalone asset. BrowserGym’s `browsergym-miniwob` integration package is Apache-2.0. |
| SWE-bench | benchmark-family asset / harness | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | `4.1.0`; vendored commit `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`; commit date `2026-03-18T20:19:47-04:00` | MIT | yes | Vendored under `vendor/SWE-bench`; used by the real-backend bootstrap and the SWE-family harness path. |
| SWE-Gym | benchmark-family asset | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | upstream repository page accessed `2026-04-10`; local bundle uses the bounded `swe_gym` slice contract defined in this repo | Apache-2.0 | no | The suite uses a bounded `swe_gym` slice and verifier/replay contract, but the upstream SWE-Gym repository is not vendored in this checkout. |
| Qwen/Qwen2.5-14B-Instruct | model weights | `https://huggingface.co/Qwen/Qwen2.5-14B-Instruct` | snapshot `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`; verified on `2026-03-28` | Apache-2.0 | no | Used only by the appendix-only RL closure slice. The release stores local snapshot metadata in `manifests/models/qwen_local_snapshots.yaml`; it does not redistribute the model weights. |
| veRL | external RL framework (appendix-only case study) | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | upstream project page accessed `2026-04-10`; local bundle references the `veRL` appendix harness and controller integration defined in this repo | Apache-2.0 | no | The repository implements a `veRL`-targeted downstream harness, but does not vendor veRL itself. |
| vLLM | inference backend (appendix-only case study) | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | upstream project page accessed `2026-04-10`; local launch contract references `vllm.entrypoints.openai.api_server` | Apache-2.0 | no | The appendix closure launches the upstream module path `vllm.entrypoints.openai.api_server`; no vLLM source is vendored here. |
| SGLang | inference backend (appendix-only case study) | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | upstream project page accessed `2026-04-10`; local launch contract references `python -m sglang.launch_server` | Apache-2.0 | no | The appendix closure launches `python -m sglang.launch_server`; no SGLang source is vendored here. |
| Apptainer | external runtime / container backend | `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate | `apptainer version 1.4.1` (from `release_manifest.yaml`); upstream project page accessed `2026-04-10` | BSD-3-Clause | no | Required for the target-cluster SWE path; runtime itself is external and not shipped in this repository. |

## Remaining provenance precision gaps

- Exact pinned upstream commit/version actually used for standalone
  `webarena-verified`, `SWE-Gym`, `veRL`, `vLLM`, and `SGLang`; the current repo
  grounds the shipped integration points and access dates, but it does not
  vendor those upstream revisions.

## Grounding sources used for this table

- `vendor/browsergym/browsergym/core/src/browsergym/core/__init__.py:1`
- `vendor/browsergym/LICENSE:1-12`
- `vendor/browsergym/browsergym/webarena_verified/pyproject.toml:6-24`
- `vendor/browsergym/browsergym/miniwob/pyproject.toml:5-28`
- `vendor/browsergym/browsergym/webarena_verified/README.md:1-20`
- `vendor/browsergym/docs/src/environments/miniwob.rst:1-23`
- `vendor/SWE-bench/pyproject.toml:5-20`
- `vendor/SWE-bench/pyproject.toml:89-93`
- `vendor/SWE-bench/LICENSE:1-21`
- `vendor/SWE-bench/swebench/__init__.py:1`
- `manifests/models/qwen_local_snapshots.yaml:42-57`
- `docs/plans/2026-04-06-verl-rl-closure.md:5-18` (local provenance note; not shipped in the reviewer bundle)
- `scripts/run_rl_closure_case_study.sh:13-19` (local provenance note; not shipped in the reviewer bundle)
- `runtime/rl_closure/backends.py:62-75` (local provenance note; not shipped in the reviewer bundle)
- `runtime/rl_closure/backends.py:155-180` (local provenance note; not shipped in the reviewer bundle)
- `release_manifest.yaml:16-19`
- `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate (MIT license; v1.0 release page checked 2026-04-10)
- `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate (Apache-2.0 license; project page checked 2026-04-10)
- `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate (Apache-2.0 license; releases page checked 2026-04-10)
- `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate (Apache-2.0 license; releases page checked 2026-04-10)
- `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate (Apache-2.0 license; releases page checked 2026-04-10)
- `https://github.com/jimmyreleasedata/evidence-gated-agent-substrate (BSD-3-Clause license; project page checked 2026-04-10)
- `https://huggingface.co/Qwen/Qwen2.5-14B-Instruct` (Apache-2.0 model card license checked 2026-04-10)
