# RC-WSSI reproducibility package

This repository accompanies manuscript v30, *Geometry-Defined Single-Owner PPE Evidence Allocation and Cross-Worker Contamination Analysis*.

It provides the implementation, paper source, split/audit manifests, and blinded annotation templates for reproducing the reported protocol. It does **not** distribute the self-collected construction-scene images, detector weights, prediction caches, or per-image labels. The raw frames may contain identifiable people and remain under the authoring organization's confidentiality and privacy controls.

## Scope

The study evaluates deterministic single-owner PPE evidence allocation and cross-worker evidence contamination under a geometry-defined reference. Its Clopper-Pearson calculation is a model-conditional feasibility diagnostic under an independent-binomial worker model. It is not a cluster-valid statistical guarantee or a deployment safety claim.

The completed human audits are complementary: a blinded, independently adjudicated, deliberately difficult candidate-restricted subset and a pre-frozen source-stratified random open-candidate audit with three blind passes. The public package contains protocol materials and aggregate outputs only; raw images and per-image labels remain private.

## Repository layout

- `manuscript/`: v30 manuscript source, bibliography, and non-sensitive analysis figures. The compiled submission PDF and qualitative source frames remain in the local submission workspace until publication authorization is confirmed.
- `scripts/`: training, validation, cached prediction, analysis, and annotation utilities.
- `audit/difficult_20260810_v1/`: manifest and instructions for the completed difficult audit. Raw images are excluded.
- `audit/random_20260812_v1/`: frozen source-stratified random audit manifest, blank templates, and the v2 open-set three-pass protocol. Raw images are excluded.
- `audit/random_20260812_v1/results_v30/`: aggregate random-audit evaluation outputs only; per-image expert labels are excluded.
- `SUBMISSION_METADATA_REQUIRED.md`: fields that require author verification before submission.

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

## Reproduction outline

1. Obtain written authorization from the data-owning organization before accessing the self-collected original images.
2. Create the four-role filename-group protocol with `scripts/build_source_disjoint_protocol.py` and validate it with `scripts/source_disjoint_validation.py`.
3. Train/cache predictions using the source-disjoint scripts, then compute worker-state summaries with `scripts/formal_worker_state_experiment.py` and `scripts/analyze_cached_rc_wssi_robustness.py`.
4. Use `scripts/serve_manual_assignment_audit.py` for the completed difficult candidate-restricted audit. For the complementary random audit, use `scripts/prepare_open_set_random_assignment_audit.py` and `scripts/serve_open_set_assignment_audit.py` for three independent open-set blind passes; freeze all three before running `scripts/analyze_open_set_random_assignment_audit.py` and `scripts/evaluate_open_set_random_human_reference.py`.

The project-specific scripts expect the authorized data, protocol manifests, run directories, and weights at the paths supplied on their command lines. They deliberately fail rather than downloading or redistributing restricted inputs.

## Citation and archival release

The repository URL is https://github.com/eew221/11222. Before manuscript submission, create a tagged GitHub release and archive that release with Zenodo (or an equivalent service). Put the resulting immutable release URL, commit hash, and DOI in `CITATION.cff`, the manuscript, and the submission materials. Do not cite a placeholder DOI.

## License and data notice

Original repository code is released under the MIT License. This license does not grant any right to access, copy, or redistribute the self-collected image data, per-image labels, weights, caches, or image-derived overlays, which remain restricted. See `LICENSE`, `DATA_AND_ETHICS.md`, and `RELEASE_NOTES_v0.4.1.md`.
