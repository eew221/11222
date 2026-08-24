# RC-WSSI reproducibility package

This repository accompanies the manuscript *Geometry-Defined Single-Owner PPE Evidence Allocation and Cross-Worker Contamination Analysis*.

It provides the implementation, paper source, split/audit manifests, and blinded annotation templates for reproducing the reported protocol. It does **not** distribute the self-collected construction-scene images, detector weights, prediction caches, or per-image labels. The raw frames may contain identifiable people and remain under the authoring organization's confidentiality and privacy controls.

## Scope

The study evaluates deterministic single-owner PPE evidence allocation and cross-worker evidence contamination under a geometry-defined reference. Its Clopper-Pearson calculation is a model-conditional feasibility diagnostic under an independent-binomial worker model. It is not a cluster-valid statistical guarantee or a deployment safety claim.

The completed human audit is a blinded, independently adjudicated, deliberately difficult and candidate-restricted subset. The complementary source-stratified random audit package is frozen before annotation; its blank templates are included, but it is not reported as an outcome in the current manuscript.

## Repository layout

- `manuscript/`: v28 manuscript source, bibliography, and non-sensitive analysis figures. The compiled PDF is kept out of the public release until image-publication permission is verified.
- `scripts/`: training, validation, cached prediction, analysis, and annotation utilities.
- `audit/difficult_20260810_v1/`: manifest and instructions for the completed difficult audit. Raw images are excluded.
- `audit/random_20260812_v1/`: frozen source-stratified random audit manifest, blank templates, Chinese annotation page, and instructions. Raw images are excluded.
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
4. Use `scripts/serve_manual_assignment_audit.py` for two independent human annotation passes. Freeze both passes before running agreement or adjudication scripts. The random-audit templates contain no outcome labels.

The project-specific scripts expect the authorized data, protocol manifests, run directories, and weights at the paths supplied on their command lines. They deliberately fail rather than downloading or redistributing restricted inputs.

## Citation and archival release

The repository URL is https://github.com/eew221/11222. Before manuscript submission, create a tagged GitHub release and archive that release with Zenodo (or an equivalent service). Put the resulting immutable release URL, commit hash, and DOI in `CITATION.cff`, the manuscript, and the submission materials. Do not cite a placeholder DOI.

## License and data notice

The repository source code is publicly accessible. No broad code-reuse license has yet been granted, so standard copyright applies unless the authors later add an explicit `LICENSE` file. This decision is independent of the self-collected image data, which are not publicly released. See `LICENSE_SELECTION_REQUIRED.md` and `DATA_AND_ETHICS.md`.
