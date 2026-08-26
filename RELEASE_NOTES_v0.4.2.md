# v0.4.2 candidate release notes

This candidate artifact accompanies manuscript v35. It is not a Zenodo archive and does not claim a DOI. Create the annotated Git tag and archive it only after completing the checks below.

## Included

- v35 manuscript source and non-sensitive figures;
- aggregate outputs from the completed three-pass random open-candidate audit, including ambiguity and crowding/competition strata;
- detector-output audit scripts:
  - `prepare_end_to_end_detection_audit.py`;
  - `serve_open_set_assignment_audit.py`;
  - `analyze_end_to_end_detection_audit.py`;
- a blank end-to-end audit protocol and Chinese annotation instructions;
- the authorization/ethics record template.

## Detector-output audit status

The detector-output audit is complete on a restricted local package using a held-out frozen `10% YOLOv8s` seed-0 cache, three final-test folds, and six images per each of 11 filename groups. The aggregate result is 376 rows (305 predicted PPE and 71 reference PPE misses), 265/297 (0.892) RC-WSSI owner agreement among predicted PPE rows with determinate human owners, 7 human-judged false detections, 24 owner-outside-detected-person events, and 361/376 all-three exact agreements. Only aggregate results are released; raw images, detector caches, sealed references, and per-image human responses remain restricted.

## Deliberately excluded restricted materials

- original images and image-derived overlays;
- per-image labels and per-row human responses;
- detector weights and prediction caches;
- sealed detector-owner references;
- redaction coordinates and unredacted source crops.

## Required before submission

1. Obtain and retain the verified data-owner authorization, approval, or exemption basis for research use and redacted qualitative-image publication.
2. Verify that no restricted file is staged or has entered the Git history.
3. Create tag `v0.4.2`, publish the GitHub release, archive it through Zenodo or an equivalent service, and insert the real commit hash and DOI into `CITATION.cff`, the manuscript, and submission metadata.
