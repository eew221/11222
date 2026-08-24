# Pre-submission release checklist

Complete these items in order. Do not invent values for any item.

1. Obtain organization approval for research use of the self-collected images and for publication of manually redacted qualitative crops; record the verified approval, authorization, or exemption basis in `DATA_AND_ETHICS.md`.
2. Complete and freeze the local Figure 5 manual redaction review. Do not upload raw frames, redaction overlays, or per-image labels.
3. Completed: original project code is released under the root MIT `LICENSE`. Keep third-party software and restricted data outside this grant.
4. Verify that no restricted originals, image overlays, model weights, or prediction caches have entered the Git history.
5. Add only the final non-image split manifests, aggregate tables, frozen annotation CSVs, and checksums that the data owner authorizes for distribution.
6. `CITATION.cff` and `.zenodo.json` record v0.4.1. Insert its exact commit hash into the release notes after creating the final commit.
7. Create a GitHub release from the `v0.4.1` tag after it is pushed.
8. Connect the repository to Zenodo, archive that GitHub release, and copy the newly generated DOI into `CITATION.cff` and the manuscript.
9. Rebuild the paper, confirm that its PDF and release commit point to the same artifact version, and re-check the active MVA page limit.
