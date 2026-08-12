# Pre-submission release checklist

Complete these items in order. Do not invent values for any item.

1. Complete every `TODO` in `DATA_AND_ETHICS.md` using the official data provider's terms and institutional records.
2. Confirm in writing whether the paper's qualitative frames and any annotated overlays may be published.
3. Decide the license for code authored by this project, add the actual `LICENSE` file, and check that it does not conflict with included third-party code.
4. Verify that no restricted originals, image overlays, model weights, or prediction caches have entered the Git history.
5. Add the final non-image split manifests, aggregate tables, frozen annotation CSVs, and checksums only when the verified data terms permit their distribution.
6. Update `CITATION.cff` with the final version and the commit hash in the release notes.
7. Create a GitHub release, for example `v1.0.0`.
8. Connect the repository to Zenodo, archive that GitHub release, and copy the newly generated DOI into `CITATION.cff` and the manuscript.
9. Rebuild the paper, confirm that its PDF and release commit point to the same artifact version, and re-check the active MVA page limit.
