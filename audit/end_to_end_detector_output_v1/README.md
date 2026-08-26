# Detector-output end-to-end audit protocol

This directory intentionally contains no images, prediction cache, sealed reference, or human response rows.

## Local execution with authorized restricted inputs

1. Freeze one held-out final-test detector cache before inspecting human labels or allocation outcomes.
2. Create the local audit package:

```powershell
python scripts/prepare_end_to_end_detection_audit.py `
  --prediction-cache <frozen_final_test_cache.json> `
  --out <restricted_local_audit_root>
```

3. Run three separate blind annotation passes. Each annotator may select a detected person, `OUTSIDE_DETECTED_PERSON_SET`, `FALSE_DETECTION`, `NONE`, or `AMBIGUOUS`. Do not disclose the sealed reference, thresholds, allocation outputs, or other annotators' files.
4. Freeze all three passes, then analyze locally:

```powershell
python scripts/analyze_end_to_end_detection_audit.py `
  --audit-root <restricted_local_audit_root> `
  --out <restricted_local_results>
```

The output separates ownership agreement for predicted PPE, false PPE detections, PPE reference misses, and real owners outside the detected-person set. Matching is frozen as same-class, one-to-one greedy IoU matching at 0.50 after a prediction confidence floor of 0.05; unmatched predictions remain distinct audit rows and are not automatically false detections.

The public aggregate results in `results_v35_public/` contain the mutually exclusive predicted-PPE partition, conditional owner-score partition, the geometric-match by human-false-detection cross-tabulation, and a count decomposition for each declared filename group. They intentionally exclude images, detector caches, sealed references, and per-row human labels. The audit is a sampled error decomposition, not a deployment-risk guarantee or a full-corpus worker-state evaluation.
