# Contributing

1. Open an issue or draft pull request describing the change and its test plan.
2. Work on a branch from the current main branch. Keep the fixed class ID order unless a separate model version and migration are agreed.
3. Add or update focused tests when behavior changes. Run `python -m unittest discover -s tests -v`; run the Stage 5G synthetic self-test when inference changes.
4. Keep generated data, downloaded images, annotations, webcam recordings, SQLite files, credentials, and experimental checkpoints out of commits. The privacy-redacted distribution copy at `models/distribution/best.pt` is tracked by Git LFS.
5. Mark evaluation results as experimental when they use provisional labels. Do not tune on the held-out test split or claim deployment readiness from these scores.
6. Document dataset provenance, image and annotation permissions, review status, and group-aware split policy for any new data. Never add third-party media to a pull request without verified redistribution rights and any necessary consent.
7. Review `git status`, `git diff --cached --stat`, and `git diff --cached` before requesting review. Include the commands run and limitations in the pull request.

The source is separate from EdgeAI-Surveillance (Model A). Do not copy its data or weights into this project.
