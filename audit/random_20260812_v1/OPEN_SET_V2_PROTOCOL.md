# Open-Set Random Ownership Audit v2

This protocol reuses the frozen 66-image source-stratified random sample in
this directory. It does not resample images and does not read detector
predictions, thresholds, or rule outputs.

The original v1 evidence template displayed only geometrically supported
candidate persons. The v2 protocol addresses that limitation by allowing each
blind annotator to choose any person ID visible in the rendered image, `NONE`,
or `AMBIGUOUS`. The original geometric candidate list is retained only for a
post hoc candidate-recall calculation.

## Blind passes

Three experts, A, B, and C, annotate separate local CSV files. Each pass is
frozen after completion. C is an independent third pass and must not see A or
B's selections. The predeclared final label is the majority choice when at
least two annotators agree. A three-way split remains `AMBIGUOUS`; there is no
post-hoc adjudication that exposes another expert's answer.

## Local preparation

The following commands require authorized local access to the redacted audit
images. They intentionally do not download or publish restricted frames.

```powershell
python scripts/prepare_open_set_random_assignment_audit.py `
  --source-audit <frozen-v1-audit-root> `
  --out <new-open-set-v2-audit-root>

python scripts/serve_open_set_assignment_audit.py `
  --audit-root <new-open-set-v2-audit-root> --annotator A --port 8773
```

Run separate local instances for B and C, for example on ports `8774` and
`8775`. After all three passes have frozen, analyze them without modifying any
input CSV:

```powershell
python scripts/analyze_open_set_random_assignment_audit.py `
  --audit-root <new-open-set-v2-audit-root> `
  --out <new-analysis-directory>
```

The analysis reports pairwise and three-way agreement, source-group summaries,
and the recall of the original geometric candidate list relative to determinate
open-set human owner labels. It does not claim full-corpus semantic truth,
cluster-valid risk control, or deployment validation.
