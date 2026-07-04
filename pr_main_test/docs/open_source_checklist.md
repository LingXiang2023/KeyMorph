# Open Source Checklist

Use this before pushing the repository publicly.

- [x] Remove training-only scripts and stale experiment variants.
- [x] Remove machine-specific defaults from `config.py`.
- [x] Ignore generated outputs, local datasets, caches, and logs.
- [x] Document installation, data layout, checkpoint path, and smoke test.
- [ ] Decide whether to include `model_zoo/default/checkpoint.pt` in Git.
- [ ] Add the final project license.
- [ ] Replace or expand this README if you want a full paper-reproduction release.
- [ ] Add your own paper title and project metadata after they are ready.
- [ ] Run `python -m py_compile test.py config.py models/network.py utils/*.py`.
- [ ] Run a one-batch smoke test on the target GitHub version.
