# Pre-submission release checklist

Complete these items in order. Do not invent values for any item.

1. Obtain organization approval for research use of the self-collected images and for publication of manually redacted qualitative crops; record the verified approval, authorization, or exemption basis in `DATA_AND_ETHICS.md`.
2. Complete and freeze the local Figure 5 manual redaction review. Do not upload raw frames, redaction overlays, or per-image labels.
3. Completed: original project code is released under the root MIT `LICENSE`. Keep third-party software and restricted data outside this grant.
4. Verify that no restricted originals, image overlays, model weights, or prediction caches have entered the Git history.
5. Add only the final non-image split manifests, aggregate tables, frozen annotation CSVs, and checksums that the data owner authorizes for distribution.
6. Completed: `CITATION.cff` and `.zenodo.json` record v0.4.2. The annotated Git tag is the immutable release pointer; GitHub displays its exact commit on the release page.
7. Completed: GitHub release `v0.4.2` is published from the fixed tag.
8. Completed: Zenodo archived the GitHub release as version DOI `10.5281/zenodo.22119278`.
9. Completed: the paper was rebuilt locally; the release commit and DOI are recorded in the manuscript and citation metadata. Re-check the active MVA page limit before submission.
