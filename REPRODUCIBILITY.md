# Reproducibility record

## Current manuscript

- Source: `manuscript/main_v25.tex`
- Bibliography: `manuscript/references.bib`
- Local compiled PDF: `RC_WSSI_MVA_manuscript_v25.pdf` in the author workspace
- Local PDF page count: 21
- Local PDF SHA-256: `C1D74A18B38EEF9C40F3AEEDD323062ECF1A85C197EA3A4A848B9A4CF91DD684`
- Build sequence: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`

The public repository intentionally excludes the compiled PDF until permission to publish the qualitative image figure is verified. The manuscript source also requires an authorized local copy of the underlying image source for a complete rebuild.

## Completed difficult audit

The completed audit is a selected, candidate-restricted set of 60 multi-worker outer-test images and 597 PPE evidence boxes. The primary row-wise exact inter-annotator agreement is 587/597 (98.32%); Cohen's kappa is reported only as an auxiliary descriptor. The final adjudicated result is limited to this selected subset and is not a full-corpus semantic ground truth.

## Frozen random audit

The complementary audit contains 66 source-stratified outer-test images, six from each of 11 filename groups, with blank annotation templates. At the manuscript version represented here, expert outcomes were not yet available and are not reported. After both passes are frozen, run the agreement and adjudication scripts and create a new immutable release containing only materials permitted by the verified data terms.

## Verification status

- Python syntax check: passed for the 21 repository scripts.
- Restricted image/weight binaries in repository tree: none at initial package creation.
- Public GitHub push: pending network/authentication availability.
- Code license: pending author selection; see `LICENSE_SELECTION_REQUIRED.md`.
- Dataset license, image permission, privacy, ethics basis, release tag, and DOI: pending author verification; see `DATA_AND_ETHICS.md` and `RELEASE_CHECKLIST.md`.

