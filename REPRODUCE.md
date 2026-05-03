# Reproduce Released Evidence Checks

Use the reviewer quick path to check the released evidence products and claim maps. These checks do not rerun live workloads.

## Reviewer quick path

These commands check released evidence tables, claim maps, and package integrity. They do not rerun live WebArena/SWE/MiniWoB++ workloads. Full live reruns require external workload infrastructure and are not part of the quick anonymous review path. Credential-bearing browser state, request metadata, raw browser-session materials, and unredacted logs are excluded.

1. Clone code repo:

```bash
git clone https://github.com/jimmyreleasedata/evidence-gated-agent-substrate
cd evidence-gated-agent-substrate
```

2. Validate code package:

```bash
python validate_bundle.py
```

3. Clone or download HF artifact dataset:

```bash
git clone https://huggingface.co/datasets/jimmyreleasedata/evidence-gated-agent-substrate-release hf_dataset
```

or:

```bash
python scripts/download_hf_artifact_index.py --repo jimmyreleasedata/evidence-gated-agent-substrate-release --out hf_dataset
```

4. Check canonical evidence gate:

```bash
python scripts/reviewer/check_evidence_gate.py --dataset-root hf_dataset
```

Expected: 930 admitted / 1,184 excluded.

5. Check decision study:

```bash
python scripts/reviewer/check_decision_study.py --dataset-root hf_dataset
```

Expected: fixed-budget slice 56/56 admitted, 0 blocked; budget-grid study 336/336 admitted, 0 blocked; comparable cells 12; reversal cells 12/12.

6. Check table/figure inputs:

```bash
python scripts/reviewer/check_table_inputs.py --dataset-root hf_dataset
```

7. Check claim-to-artifact map:

```bash
python scripts/reviewer/check_claim_artifact_map.py --dataset-root hf_dataset --code-root .
```

8. Check checksums:

```bash
python scripts/reviewer/check_checksums.py --dataset-root hf_dataset
```

Checksum scope note:
- In the OpenReview supplementary zip, `checksums/SHA256SUMS` validates only files present in the unzipped supplementary root. From the unzip root, `sha256sum -c checksums/SHA256SUMS` should pass without missing-file errors.
- In the full Hugging Face dataset root, `checksums/HF_DATASET_SHA256SUMS` validates the claim-complete dataset files. The reviewer helper `python scripts/reviewer/check_checksums.py --dataset-root hf_dataset` uses that HF dataset checksum file when present.


9. Print OpenReview fields:

```bash
python scripts/reviewer/print_openreview_fields.py
```

