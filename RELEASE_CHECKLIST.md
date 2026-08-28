# Pre-submission release checklist

Complete these items in order. Do not invent values for any item.

1. Obtain organization approval for research use of the self-collected images and for publication of manually redacted qualitative crops; record the verified approval, authorization, or exemption basis in `DATA_AND_ETHICS.md`.
2. Complete and freeze the local Figure 5 manual redaction review. Do not upload raw frames, redaction overlays, or per-image labels.
3. Completed: original project code is released under the root MIT `LICENSE`. Keep third-party software and restricted data outside this grant.
4. Verify that no restricted originals, image overlays, model weights, or prediction caches have entered the Git history.
5. Add only the final non-image split manifests, aggregate tables, frozen annotation CSVs, and checksums that the data owner authorizes for distribution.
6. Completed: `CITATION.cff` and `.zenodo.json` identify the v0.4.3 software release associated with manuscript v38; Zenodo DOI `10.5281/zenodo.22141331` is recorded.
7. Completed: GitHub release `v0.4.2` is published from the fixed tag.
8. Completed: Zenodo archived the GitHub release as version DOI `10.5281/zenodo.22119278`.
9. Local build completed: manuscript v38 was rebuilt and checked. The accompanying release resolves to commit `0d32f36361a01b8d4c2a456ac6c4858bcbb4d4fb` and Zenodo DOI `10.5281/zenodo.22141331`.

