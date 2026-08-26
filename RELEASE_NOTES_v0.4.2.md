# v0.4.2 release notes

This reproducibility artifact accompanies manuscript v35. The GitHub release and its Zenodo archive identify the immutable version of the public materials.

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

## Access and compliance note

The public artifact intentionally excludes restricted materials. Any editorial request for controlled data access remains subject to the data owner's written authorization and applicable privacy controls. The final submission records the actual authorization, approval, or exemption basis separately; this release does not invent such an identifier.
