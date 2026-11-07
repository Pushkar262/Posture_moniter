# Seated Posture Monitor

CPU-only Python CLI: MediaPipe Tasks Pose Landmarker tracks one seated person and writes a JSON session summary. Python 3.9+ with a compatible MediaPipe wheel.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python posture_monitor.py --source 0 --duration 60 --out posture.json
# Video: python posture_monitor.py --source sitting.mp4 --duration 60 --out posture.json
# Optional desktop preview: add --preview
python posture_monitor.py --source 0 --duration 60 --out posture.json --preview
```

First run downloads Google's Pose Landmarker **lite** model to `~/.cache/seated-posture-monitor/`; use `--model PATH` for a local model. CPU delegate is explicit. No GPU, UI, web service, Docker, or training is required. Exit `0` means at least one usable pose was observed and JSON written; unreadable input, missing dependencies, or no detected person returns non-zero. Invalid frames are skipped and long gaps are excluded.

## Decisions

- **Scale invariant:** use pixel coordinates normalized by shoulder-midpoint-to-hip-midpoint torso length. Features: torso lean, head/shoulder vertical gap, head horizontal offset, and shoulder-height asymmetry.
- **Entry thresholds:** lean `≤14°`, neck-gap ratio `≥0.22`, head-offset ratio `≤0.18`, shoulder-tilt ratio `≤0.12`. Ratios are divided by torso length. Wider exit limits (`18°`, `0.18`, `0.23`, `0.16`) provide hysteresis. These are initial engineering heuristics, not clinical limits; validate on representative recordings.
- **Stability:** 7-frame median filtering; a candidate must persist 1 second before a state change. A `GOOD`→`BAD` episode counts as a break only after 2 seconds. One bad frame is ignored; gaps over 0.75 seconds reset state.
- **Score:** `posture_score` is good-posture seconds / pose-observed seconds × 100. JSON also includes coverage, unclassified time, break intervals, and all thresholds; the score is not model confidence.

## Limits and submission

Assumes one person, fixed camera, and visible head/shoulders/hips. A frontal camera measures lateral lean/alignment, not forward-back slouch; side/three-quarter view is better. This is not a medical diagnosis. Review tracking coverage before interpreting the score.