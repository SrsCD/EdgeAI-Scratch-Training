# GitHub publishing checklist

- [ ] Review the initial `git status --short` and `git diff --cached --stat` before committing.
- [ ] Confirm the only staged `.pt` is `models/distribution/best.pt`, stored as a Git LFS pointer. Run `git lfs ls-files` after staging. The original run checkpoint must remain ignored.
- [ ] Confirm no `data/raw`, `data/interim`, `data/experimental`, `data/processed`, webcam recordings, event evidence, SQLite databases, credentials, or source photos are staged.
- [ ] Verify both checkpoint SHA-256 values in the README and webcam script. Do not substitute the pretrained root `yolo11n.pt`.
- [ ] Review the source and model license recommendation, select and add a code `LICENSE`, and resolve any required notices before public publication.
- [ ] Keep dataset images, annotations, and genuine prediction screenshots private until redistribution rights, attribution, and participant privacy are reviewed.
- [ ] Run the unit suite, `pip check`, and the Stage 5G self-test on a supported setup.
- [ ] Have a teammate clone with Git LFS and follow the setup guide before announcing the repository as runnable.
- [ ] Create the GitHub repository only after the above review, then add the remote and push; no remote is configured by this preparation.

For a private team repository, still review data rights and privacy before sharing assets. The source and aggregate metrics can be reviewed without the excluded dataset.
