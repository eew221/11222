# Reproducibility record

## Current manuscript

- Source: `manuscript/main_v35.tex`
- Bibliography: `manuscript/references.bib`
- Local compiled PDF: `RC_WSSI_MVA_manuscript_v35.pdf` in the author workspace
- Local PDF page count: recorded after the final local compile; the compiled PDF is not public until qualitative-image publication authorization is verified.
- Local PDF SHA-256: recorded only after the final local compile.
- Build sequence: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`

The public repository intentionally excludes the compiled PDF until permission to publish the qualitative image figure is verified. The manuscript source also requires an authorized local copy of the underlying image source for a complete rebuild.

## Completed difficult audit

The completed audit is a selected, candidate-restricted set of 60 multi-worker outer-test images and 597 PPE evidence boxes. The primary row-wise exact inter-annotator agreement is 587/597 (98.32%); Cohen's kappa is reported only as an auxiliary descriptor. The final adjudicated result is limited to this selected subset and is not a full-corpus semantic ground truth.

## Completed random audit

The complementary audit contains 66 source-stratified outer-test images, six from each of 11 filename groups, with three frozen open-candidate blind passes. It contains 334 PPE rows, 321 determinate blind-majority human-owner rows, and 13 rows retained as `AMBIGUOUS`. Only aggregate outputs are distributed. The audit is sampled semantic ownership evidence, not full-corpus worker-state truth or a cluster-valid inference result.

## Completed detector-output audit

The completed detector-output audit uses a frozen held-out 10% YOLOv8s seed-0 cache, three final-test folds, and six images from each of 11 filename groups. It renders actual predicted person/PPE boxes and reference-PPE miss rows, then collects three separate frozen blind passes. Its aggregate result is 376 audit rows: 305 predicted PPE and 71 missed reference PPE. The mutually exclusive predicted-PPE partition is 290 in-set owners, 7 owners outside the detected-person set, 7 human false detections, and 1 ambiguous row. Conditional on the 297 predicted PPE rows with determinate human owners, RC-WSSI has 265 exact owners. Of 209 predictions geometrically matched to a reference PPE, 1 is human-judged false PPE; among 96 unmatched predictions, 6 are human-judged false PPE and 90 are not. The public `audit/end_to_end_detector_output_v1/results_v35_public/` directory includes only aggregate and filename-group count decompositions. The sealed reference, raw images, cache, and per-row expert responses remain restricted.

## Verification status

- Python syntax check: passed for the 21 repository scripts.
- Restricted image/weight binaries in repository tree: none at initial package creation.
- Public GitHub release: v0.4.1 is public; v0.4.2 is a local candidate until tagged and published.
- Code license: MIT for original code only; restricted data are excluded.
- The dataset is self-collected and not publicly released. Organization-approved ethics/legal basis must be retained before submission; see `DATA_AND_ETHICS.md`, `RELEASE_CHECKLIST.md`, and `templates/单位数据研究使用与脱敏图发表授权记录_中文.md`.

