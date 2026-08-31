# Pre-submission release checklist

Complete these items in order. Do not invent values for any item.

1. Obtain organization approval for research use of the self-collected images and for publication of manually redacted qualitative crops; record the verified approval, authorization, or exemption basis in `DATA_AND_ETHICS.md`.
2. Complete and freeze the local Figure 5 manual redaction review. Do not upload raw frames, redaction overlays, or per-image labels.
3. Completed: original project code is released under the root MIT `LICENSE`. Keep third-party software and restricted data outside this grant.
4. Verify that no restricted originals, image overlays, model weights, or prediction caches have entered the Git history.
5. Add only the final non-image split manifests, aggregate tables, frozen annotation CSVs, and checksums that the data owner authorizes for distribution.
6. Completed: `CITATION.cff` and `.zenodo.json` identify the v0.4.4 software release associated with manuscript v38; the stable Zenodo concept DOI is `10.5281/zenodo.22141330`.
7. Completed: GitHub release `v0.4.4` is published from its immutable tag.
8. Completed: Zenodo archives the v0.4.4 tag as a versioned record; record the generated version DOI on the GitHub release page.
9. Local build completed: manuscript v38 was rebuilt and checked. The accompanying public release is v0.4.4; its immutable tag identifies the archived commit.
