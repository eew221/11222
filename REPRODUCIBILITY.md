# Reproducibility record

## Current manuscript

- Source: `manuscript/main_v38.tex`
- Bibliography: `manuscript/references.bib`
- Local compiled PDF: `output/pdf/RC_WSSI_MVA_manuscript_v38.pdf` in the author workspace
- Local PDF page count: 26 pages. The compiled PDF is not public because qualitative source images remain restricted.
- Local PDF SHA-256: `789FBAAE232F847449879C7A5D335D6EEF4F7E00C3A8C17F60561CF30E411FD5`.
- Build sequence: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`

The public repository intentionally excludes the compiled PDF until permission to publish the qualitative image figure is verified. The manuscript source also requires an authorized local copy of the underlying image source for a complete rebuild.

## Completed difficult audit

The completed audit is a selected, candidate-restricted set of 60 multi-worker outer-test images and 597 PPE evidence boxes. The primary row-wise exact inter-annotator agreement is 587/597 (98.32%); Cohen's kappa is reported only as an auxiliary descriptor. The final adjudicated result is limited to this selected subset and is not a full-corpus semantic ground truth.

## Completed random audit

The complementary audit contains 66 source-stratified outer-test images, six from each of 11 filename groups, with three frozen open-candidate blind passes. It contains 334 PPE rows, 321 determinate blind-majority human-owner rows, and 13 rows retained as `AMBIGUOUS`. Only aggregate outputs are distributed. The audit is sampled semantic ownership evidence, not full-corpus worker-state truth or a cluster-valid inference result.

## Completed random worker-state audit

The completed audit contains 220 pre-frozen final-test images from all 11 filename groups and 686 worker rows labeled directly as `SAFE`, `UNSAFE`, or `REVIEW` by two frozen primary passes, with three component disagreements adjudicated by a third expert. The consensus distribution is 164 `SAFE`, 342 `UNSAFE`, and 180 `REVIEW`. The two primary passes agree on the overall state for 686/686 rows and on all component fields for 683/686 rows. Replaying the five frozen seeds and three folds produces 17,150 method rows (686 rows x 5 methods x 5 seeds); it does not create 17,150 independent workers. RC-WSSI matched-exact agreement is 0.570 as a 15-cell mean and 0.568 when repeated records are pooled over five seeds. The corresponding repeated-record unsafe denominator is 1,710; automatic-safe errors are 576 and composite errors are 755. These are sampled, detector-person-matching-conditional diagnostics, not full-corpus accuracy, cluster-valid inference, or deployment safety estimates.

The public files are in `audit/independent_worker_state_random_20260827_v1/results_v37_public/`. The public package excludes the consensus CSV, annotator responses, annotated images, and the per-row method output because those materials are image-derived or row-level restricted records. The directory name preserves the frozen audit artifact version; the manuscript that consumes it is v38.

## Completed detector-output audit

The completed detector-output audit uses a frozen held-out 10% YOLOv8s seed-0 cache, three final-test folds, and six images from each of 11 filename groups. It renders actual predicted person/PPE boxes and reference-PPE miss rows, then collects three separate frozen blind passes. Its aggregate result is 376 audit rows: 305 predicted PPE and 71 missed reference PPE. The mutually exclusive predicted-PPE partition is 290 in-set owners, 7 owners outside the detected-person set, 7 human false detections, and 1 ambiguous row. Conditional on the 297 predicted PPE rows with determinate human owners, RC-WSSI has 265 exact owners. Of 209 predictions geometrically matched to a reference PPE, 1 is human-judged false PPE; among 96 unmatched predictions, 6 are human-judged false PPE and 90 are not. The public `audit/end_to_end_detector_output_v1/results_v35_public/` directory includes only aggregate and filename-group count decompositions. The sealed reference, raw images, cache, and per-row expert responses remain restricted.

## Verification status

- Python syntax check: passed for the 21 repository scripts.
- Restricted image/weight binaries in repository tree: none at initial package creation.
- Public GitHub release: v0.4.2 is published and archived. The release tag resolves to commit `6b916c59570ec5db1d068b00299ff8802b037846`; Zenodo version DOI: `10.5281/zenodo.22119278`. This record predates manuscript v38 and must not be cited as the v38 archive.
- v38 archival status: the v0.4.3 GitHub release and its Zenodo record must be created from the frozen v38 source and non-image artifacts; record the resulting commit and DOI here after publication.
- Code license: MIT for original code only; restricted data are excluded.
- The dataset is self-collected and not publicly released. Organization-approved ethics/legal basis and the local qualitative-image redaction record remain outside this code package and must be verified before submission; see `DATA_AND_ETHICS.md`, `RELEASE_CHECKLIST.md`, and `templates/单位数据研究使用与脱敏图发表授权记录_中文.md`.

