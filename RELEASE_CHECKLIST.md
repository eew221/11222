# Pre-submission release checklist

Complete these items in order. Do not invent values for any item.

1. Obtain organization approval for research use of the self-collected images and for publication of manually redacted qualitative crops; record the verified approval, authorization, or exemption basis in `DATA_AND_ETHICS.md`.
2. Complete and freeze the local Figure 5 manual redaction review. Do not upload raw frames, redaction overlays, or per-image labels.
3. Decide the license for code authored by this project, add the actual `LICENSE` file, and check that it does not conflict with included third-party code.
4. Verify that no restricted originals, image overlays, model weights, or prediction caches have entered the Git history.
5. Add only the final non-image split manifests, aggregate tables, frozen annotation CSVs, and checksums that the data owner authorizes for distribution.
6. Update `CITATION.cff` with the final version and the commit hash in the release notes.
7. Create a GitHub release, for example `v1.0.0`.
8. Connect the repository to Zenodo, archive that GitHub release, and copy the newly generated DOI into `CITATION.cff` and the manuscript.
9. Rebuild the paper, confirm that its PDF and release commit point to the same artifact version, and re-check the active MVA page limit.
