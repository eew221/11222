# Data, image permission, privacy, and ethics statement

## Data availability

The study uses a self-collected low-light tunnel PPE image dataset held locally by the authoring organization. The original images, detector weights, prediction caches, and per-image annotations are **not publicly available** because the raw frames may contain identifiable people and are subject to organizational confidentiality and privacy controls. This repository therefore distributes no original images or image-derived overlays.

The public artifact contains code, paper source, aggregate tables, non-image protocol manifests, and blank annotation templates only. The corresponding author may be contacted about the feasibility of editorial data-access requests, which remain subject to the data owner's written authorization and applicable privacy controls. This statement does not promise public release or unrestricted access to the original dataset.

## Qualitative images and redaction

The manuscript contains a limited number of qualitative construction-scene crops. Before submission, each crop must be manually reviewed for faces, badges/names, vehicle plates, company marks, and other identifiable information. Each identified region must be strongly pixelated, and the review coordinates must be frozen in a local redaction manifest before the final Figure 5 is rendered. The raw source frames and redaction overlays are not uploaded to this repository.

At the repository version represented by the current manuscript source, the redaction review is still a local pre-submission task. The final PDF must not be submitted until the review has been completed and frozen.

## Ethics and legal basis

The dataset is self-collected. The authors must confirm, before submission, the organization-approved legal/ethical basis for research use and for publication of manually redacted qualitative crops. The applicable approval number, written authorization, or documented exemption basis is not asserted in this repository because no verified record has yet been supplied.

A Chinese local record template is provided at `templates/单位数据研究使用与脱敏图发表授权记录_中文.md`. It is a blank template, not evidence of authorization. Do not replace this statement or insert an approval identifier until a verified record has been obtained and retained.

## Public artifact boundary

This repository intentionally excludes original frames, annotated image overlays, detector weights, prediction caches, and per-image labels. Public code availability does not grant any right to access, copy, or redistribute the self-collected image data.
