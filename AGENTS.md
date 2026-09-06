# AGENTS.md

## Cursor Cloud specific instructions

This is a **research/experiment Python project** (no long-running server or web UI). It evaluates motion consistency in surveillance video using a 6-stage pipeline with a Dual-Stream PyTorch LSTM autoencoder. The main way to "run" it is to execute the experiment script and the test suite.

### Environment

- Python 3.12 with a virtualenv at `.venv/` (created by the startup/update script). Activate with `source .venv/bin/activate` or call binaries directly as `.venv/bin/python`.
- PyTorch is installed as the **CPU-only** build (from `https://download.pytorch.org/whl/cpu`); there is no GPU in this environment. The experiment runs fine on CPU in ~10s.
- `opencv-python` needs the system library `libGL.so.1`, which is already present in this VM. If `import cv2` ever fails with a `libGL` error on a fresh image, install `libgl1` via apt.

### Run the experiment (primary end-to-end task)

```bash
.venv/bin/python Data_set/run_full_experiment.py --epochs 25
```

- It reads the pre-committed motion streams `Data_set/processed_output/1_stream_A_frame_diff.npy` and `1_stream_B_optical_flow.npy` (no raw video needed), trains the two LSTM autoencoders, and writes models, CSVs, `run_summary.json`, and PNG plots into `Data_set/processed_output/stage23/`.
- Note: this **overwrites tracked files** in `Data_set/processed_output/stage23/`. Those regenerated model/plot outputs are checked into git; do not commit the churn unless intended (`git checkout -- Data_set/processed_output/stage23/` restores them). Results vary slightly run-to-run due to random weight init (R is ~0.94–0.96).

### Run the tests

```bash
cd Data_set && ../.venv/bin/python test_pipeline.py
```

- The suite is a standalone script (not pytest-driven despite the README mention). It runs synthetic-data unit tests. Test 7 (full pipeline on a real video) is **expected to be SKIPPED** because it points at a hardcoded Windows path (`D:\...\1.mp4`) that does not exist here.

### Lint

There is no configured linter (no ruff/flake8/black config in the repo).
