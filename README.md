# RC-WSSI reproducibility package

This repository accompanies manuscript v38, *Geometry-Defined Single-Owner PPE Evidence Allocation and Cross-Worker Contamination Analysis*.

It provides the implementation, paper source, split/audit manifests, and blinded annotation templates for reproducing the reported protocol. It does **not** distribute the self-collected construction-scene images, detector weights, prediction caches, or per-image labels. The raw frames may contain identifiable people and remain under the authoring organization's confidentiality and privacy controls.

## Scope

The study evaluates deterministic single-owner PPE evidence allocation and cross-worker evidence contamination under a geometry-defined reference. Its Clopper-Pearson calculation is a model-conditional feasibility diagnostic under an independent-binomial worker model. It is not a cluster-valid statistical guarantee or a deployment safety claim.

The completed human audits are complementary: a difficult candidate-restricted ownership audit with third-expert adjudication, a pre-frozen random open-candidate ownership audit, a detector-output audit, and a pre-frozen random open-set worker-state audit. The worker-state audit covers 220 images and 686 directly human-labeled worker rows; RC-WSSI matched-exact agreement is 0.570 as a 15-cell mean and 0.568 when the same rows are pooled over five repeated seeds. The detector-output audit covers 66 pre-frozen final-test images and 376 rows (305 predicted PPE and 71 reference PPE misses); RC-WSSI agrees on 265/297 (0.892) predicted PPE rows with determinate human owners. These are sampled, conditional diagnostics, not full-corpus semantic accuracy or deployment validation. The public package contains protocol materials and aggregate outputs only; raw images, per-image labels, annotated images, weights, and prediction caches remain private.

## Repository layout

- `manuscript/`: v38 manuscript source, bibliography, and non-sensitive analysis figures. The compiled submission PDF and qualitative source frames remain in the local submission workspace until publication authorization is confirmed.
- `scripts/`: training, validation, cached prediction, analysis, and annotation utilities.
- `audit/difficult_20260810_v1/`: manifest and instructions for the completed difficult audit. Raw images are excluded.
- `audit/random_20260812_v1/`: frozen source-stratified random audit manifest, blank templates, and the v2 open-set three-pass protocol. Raw images are excluded.
- `audit/random_20260812_v1/results_v31/`: aggregate random-audit evaluation outputs only; per-image expert labels are excluded.
- `audit/end_to_end_detector_output_v1/`: templates, instructions, and aggregate results from the completed detector-output audit. Raw images, detector caches, sealed references, and human response rows are excluded.
- `audit/independent_worker_state_random_20260827_v1/`: protocol and frozen manifests for the completed random open-set worker-state audit. Restricted images and per-annotator rows are excluded from the public result package.
- `audit/independent_worker_state_random_20260827_v1/results_v37_public/`: frozen public aggregate worker-state distributions, method summaries, filename-group summaries, selected-policy audit, and evaluation manifest used to report the v38 analysis. The directory name preserves the audit artifact's original frozen version.
- `SUBMISSION_METADATA_REQUIRED.md`: fields that require author verification before submission; the v38 manuscript does not invent missing ethics or archival identifiers.

## Environment

The verification environment used Python 3.12 with the pinned packages in `requirements.txt`. GPU-dependent training additionally requires a compatible CUDA-enabled PyTorch installation and the underlying authorized dataset. The non-GPU audit pages use only the Python standard library after the manifests and annotated images have been prepared locally.

Install the recorded dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The recorded `torch==2.6.0` requirement is intentionally CUDA-neutral. Install the PyTorch wheel matching the local CUDA driver from the official PyTorch index before attempting detector training.

The manuscript uses the Springer Nature `sn-jnl` class and `sn-mathphys-num` bibliography style. Obtain those template files from the current MVA/Springer author package when compiling; they are not redistributed here. The public source has a fallback for the restricted qualitative figure, so a clean-room code-package build does not require the original images.

## Reproduction outline

1. Obtain written authorization from the data-owning organization before accessing the self-collected original images.
2. Create the four-role filename-group protocol with `scripts/build_source_disjoint_protocol.py` and validate it with `scripts/source_disjoint_validation.py`.
3. Train/cache predictions using the source-disjoint scripts, then compute worker-state summaries with `scripts/formal_worker_state_experiment.py` and `scripts/analyze_cached_rc_wssi_robustness.py`.
4. Use `scripts/serve_manual_assignment_audit.py` for the completed difficult candidate-restricted audit. For the complementary random audit, use `scripts/prepare_open_set_random_assignment_audit.py` and `scripts/serve_open_set_assignment_audit.py` for three independent open-set blind passes; freeze all three before running `scripts/analyze_open_set_random_assignment_audit.py` and `scripts/evaluate_open_set_random_human_reference.py`.
5. For the detector-output audit, freeze a held-out prediction cache, run `scripts/prepare_end_to_end_detection_audit.py`, serve three independent passes with `scripts/serve_open_set_assignment_audit.py`, and run `scripts/analyze_end_to_end_detection_audit.py` only after all three passes are frozen. The completed public aggregate outputs are in `audit/end_to_end_detector_output_v1/results_v35_public/`; the sealed detector reference and per-image response rows remain restricted.
6. For the worker-state audit, run `scripts/evaluate_worker_state_human_audit.py` only after the A/B passes and adjudication are frozen. The public `results_v37_public/` directory contains aggregate outputs from 686 consensus worker rows replayed over 15 seed--fold cells; the consensus CSV, annotated images, and per-row method outputs remain restricted. Its original directory name is retained to preserve the frozen audit provenance used by manuscript v38.

The project-specific scripts expect the authorized data, protocol manifests, run directories, and weights at the paths supplied on their command lines. They deliberately fail rather than downloading or redistributing restricted inputs.

## Citation and archival release

The repository URL is https://github.com/eew221/11222. Release `v0.4.2` is archived at commit `6b916c59570ec5db1d068b00299ff8802b037846`, with Zenodo DOI `10.5281/zenodo.22119278`. The v38 source is prepared for immutable software release `v0.4.3`; its commit and Zenodo DOI are recorded after the release is archived. The older DOI must not be cited as if it archived v38.

## License and data notice

Original repository code is released under the MIT License. This license does not grant any right to access, copy, or redistribute the self-collected image data, per-image labels, weights, caches, or image-derived overlays, which remain restricted. See `LICENSE`, `DATA_AND_ETHICS.md`, and `RELEASE_NOTES_v0.4.2.md`.
