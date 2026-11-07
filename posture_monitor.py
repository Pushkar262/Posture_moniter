#!/usr/bin/env python3
"""CPU-only seated posture monitor using MediaPipe Pose.

Example:
    python posture_monitor.py --source 0 --duration 60 --out posture.json
    python posture_monitor.py --source session.mp4 --duration 60 --out posture.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class Thresholds:
    """Geometric thresholds; ratios are normalized by shoulder-to-hip length."""

    min_visibility: float = 0.55
    enter_lean_deg: float = 14.0
    exit_lean_deg: float = 18.0
    enter_neck_gap_ratio: float = 0.22
    exit_neck_gap_ratio: float = 0.18
    enter_head_offset_ratio: float = 0.18
    exit_head_offset_ratio: float = 0.23
    enter_shoulder_tilt_ratio: float = 0.12
    exit_shoulder_tilt_ratio: float = 0.16
    smoothing_window: int = 7
    transition_seconds: float = 1.0
    break_min_seconds: float = 2.0
    max_observation_gap_seconds: float = 0.75


@dataclass
class BreakEvent:
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass
class Features:
    lean_deg: float
    neck_gap_ratio: float
    head_offset_ratio: float
    shoulder_tilt_ratio: float


@dataclass
class Landmark:
    """Small test-friendly landmark representation."""

    x: float
    y: float
    visibility: float = 1.0


def _point(landmarks: Sequence[Any], index: int, width: int, height: int,
           min_visibility: float) -> Optional[tuple[float, float]]:
    try:
        lm = landmarks[index]
        visibility = float(getattr(lm, "visibility", 1.0))
        x, y = float(lm.x), float(lm.y)
    except (IndexError, TypeError, ValueError, AttributeError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(visibility)):
        return None
    if visibility < min_visibility:
        return None
    return x * width, y * height


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extract_features(landmarks: Sequence[Any], image_shape: Sequence[int],
                     thresholds: Thresholds = Thresholds()) -> Optional[Features]:
    """Extract scale-normalized upper-body geometry from MediaPipe Pose landmarks.

    Landmark indices follow MediaPipe Pose: nose=0, ears=7/8, shoulders=11/12,
    hips=23/24. Coordinates are converted to pixels before geometry so non-square
    images are handled correctly. Missing/low-visibility points yield None.
    """
    if len(image_shape) < 2:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    if width <= 0 or height <= 0:
        return None

    left_shoulder = _point(landmarks, 11, width, height, thresholds.min_visibility)
    right_shoulder = _point(landmarks, 12, width, height, thresholds.min_visibility)
    left_hip = _point(landmarks, 23, width, height, thresholds.min_visibility)
    right_hip = _point(landmarks, 24, width, height, thresholds.min_visibility)
    if None in (left_shoulder, right_shoulder, left_hip, right_hip):
        return None

    # Type narrowing after the visibility checks above.
    assert left_shoulder and right_shoulder and left_hip and right_hip
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    torso_length = _distance(shoulder_mid, hip_mid)
    if torso_length < 1.0:
        return None

    # Prefer the ear midpoint (more robust for head centering); fall back to nose
    # when one/both ears are occluded, which is common with a frontal webcam.
    left_ear = _point(landmarks, 7, width, height, thresholds.min_visibility)
    right_ear = _point(landmarks, 8, width, height, thresholds.min_visibility)
    if left_ear is not None and right_ear is not None:
        head_point = _midpoint(left_ear, right_ear)
    else:
        head_point = _point(landmarks, 0, width, height, thresholds.min_visibility)
    if head_point is None:
        return None

    dx = abs(shoulder_mid[0] - hip_mid[0])
    dy = abs(shoulder_mid[1] - hip_mid[1])
    lean_deg = math.degrees(math.atan2(dx, max(dy, 1e-6)))
    neck_gap_ratio = (shoulder_mid[1] - head_point[1]) / torso_length
    head_offset_ratio = abs(head_point[0] - shoulder_mid[0]) / torso_length
    shoulder_tilt_ratio = abs(left_shoulder[1] - right_shoulder[1]) / torso_length

    values = (lean_deg, neck_gap_ratio, head_offset_ratio, shoulder_tilt_ratio)
    if not all(math.isfinite(value) for value in values):
        return None
    return Features(*values)


def is_upright(features: Features, thresholds: Thresholds,
               current_state: Optional[str]) -> bool:
    """Apply entry thresholds and wider exit thresholds (hysteresis)."""
    if current_state == "GOOD":
        lean = thresholds.exit_lean_deg
        neck = thresholds.exit_neck_gap_ratio
        head = thresholds.exit_head_offset_ratio
        tilt = thresholds.exit_shoulder_tilt_ratio
    else:
        lean = thresholds.enter_lean_deg
        neck = thresholds.enter_neck_gap_ratio
        head = thresholds.enter_head_offset_ratio
        tilt = thresholds.enter_shoulder_tilt_ratio
    return (
        features.lean_deg <= lean
        and features.neck_gap_ratio >= neck
        and features.head_offset_ratio <= head
        and features.shoulder_tilt_ratio <= tilt
    )


class TemporalPostureTracker:
    """Median smoothing + dwell-time state transitions to reject single bad frames."""

    def __init__(self, thresholds: Thresholds = Thresholds()) -> None:
        self.thresholds = thresholds
        self._window: deque[Features] = deque(maxlen=max(1, thresholds.smoothing_window))
        self.state: Optional[str] = None  # GOOD, BAD, or UNKNOWN
        self._candidate: Optional[str] = None
        self._candidate_seconds = 0.0
        self.last_timestamp: Optional[float] = None
        self.last_interval = 0.0
        self.pose_frames = 0
        self.pose_observed_seconds = 0.0
        self.good_seconds = 0.0
        self.bad_seconds = 0.0
        self.unclassified_seconds = 0.0
        self.breaks: list[BreakEvent] = []
        self._active_break_start: Optional[float] = None
        self._finished = False

    @staticmethod
    def _median_features(window: Iterable[Features]) -> Features:
        items = list(window)
        return Features(
            lean_deg=statistics.median(x.lean_deg for x in items),
            neck_gap_ratio=statistics.median(x.neck_gap_ratio for x in items),
            head_offset_ratio=statistics.median(x.head_offset_ratio for x in items),
            shoulder_tilt_ratio=statistics.median(x.shoulder_tilt_ratio for x in items),
        )

    def _close_break(self, end_time: float) -> None:
        if self._active_break_start is None:
            return
        start = self._active_break_start
        duration = max(0.0, end_time - start)
        if duration >= self.thresholds.break_min_seconds:
            self.breaks.append(BreakEvent(round(start, 3), round(end_time, 3), round(duration, 3)))
        self._active_break_start = None

    def _transition(self, new_state: str, timestamp: float) -> None:
        old_state = self.state
        if old_state == new_state:
            return
        if old_state == "BAD":
            self._close_break(timestamp)
        if new_state == "BAD" and old_state == "GOOD":
            self._active_break_start = timestamp
        self.state = new_state
        self._candidate = None
        self._candidate_seconds = 0.0

    def update(self, timestamp: float, features: Features) -> None:
        if self._finished:
            raise RuntimeError("Cannot update a finished tracker")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            # Non-monotonic timestamps can occur in damaged files; ignore that frame.
            return

        self.pose_frames += 1
        self._window.append(features)
        smoothed = self._median_features(self._window)

        dt = 0.0
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            self.last_interval = dt
            if dt > self.thresholds.max_observation_gap_seconds:
                # Do not count missing time. Close a known bad episode at the last
                # observation and require a fresh stable run after a long gap.
                self._close_break(self.last_timestamp)
                self.state = None
                self._candidate = None
                self._candidate_seconds = 0.0
                self._window.clear()
                self._window.append(features)
                smoothed = features
                dt = 0.0
            else:
                self.pose_observed_seconds += dt
                if self.state == "GOOD":
                    self.good_seconds += dt
                elif self.state == "BAD":
                    self.bad_seconds += dt
                else:
                    self.unclassified_seconds += dt

        upright = is_upright(smoothed, self.thresholds, self.state)
        candidate = "GOOD" if upright else "BAD"
        if candidate == self._candidate:
            self._candidate_seconds += dt
        else:
            self._candidate = candidate
            self._candidate_seconds = 0.0

        if self._candidate_seconds >= self.thresholds.transition_seconds:
            transition_time = timestamp
            self._transition(candidate, transition_time)

        self.last_timestamp = timestamp

    def finish(self, frame_period: float = 0.0) -> None:
        if self._finished:
            return
        if self.last_timestamp is not None and frame_period > 0:
            dt = min(frame_period, self.thresholds.max_observation_gap_seconds)
            self.pose_observed_seconds += dt
            if self.state == "GOOD":
                self.good_seconds += dt
            elif self.state == "BAD":
                self.bad_seconds += dt
            else:
                self.unclassified_seconds += dt
        if self.state == "BAD" and self.last_timestamp is not None:
            end = self.last_timestamp + max(0.0, frame_period)
            self._close_break(end)
        self._finished = True

    def summary(self, session_duration: float, thresholds: Thresholds) -> dict[str, Any]:
        observed = max(0.0, self.pose_observed_seconds)
        good_pct = 100.0 * self.good_seconds / observed if observed > 0 else 0.0
        coverage = 100.0 * observed / session_duration if session_duration > 0 else 0.0
        return {
            "schema_version": 1,
            "method": "MediaPipe Tasks Pose Landmarker (lite, CPU) + scale-normalized geometry + temporal hysteresis",
            "session_duration_seconds": round(max(0.0, session_duration), 2),
            "pose_frames": self.pose_frames,
            "pose_observed_seconds": round(observed, 2),
            "tracking_coverage_percent": round(min(100.0, coverage), 1),
            "good_posture_seconds": round(self.good_seconds, 2),
            "non_upright_seconds": round(self.bad_seconds, 2),
            "unclassified_seconds": round(self.unclassified_seconds, 2),
            "good_posture_percent": round(min(100.0, max(0.0, good_pct)), 1),
            "break_count": len(self.breaks),
            "breaks": [asdict(event) for event in self.breaks],
            "posture_score": round(min(100.0, max(0.0, good_pct)), 1),
            "thresholds": asdict(thresholds),
            "interpretation": "posture_score equals the percentage of pose-observed time classified as settled upright posture; unclassified transition time remains in the denominator",
            "limitations": [
                "This is a heuristic posture estimate, not a medical or ergonomic diagnosis.",
                "A frontal camera estimates lateral lean and head/shoulder alignment; forward-back slouch is better assessed from a side or three-quarter view.",
                "Single-person tracking is assumed. Other people or severe occlusion can reduce tracking coverage.",
            ],
        }


def _parse_source(source: str) -> int | str:
    value = source.strip()
    if value.isdigit():
        return int(value)
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def resolve_model_path(model_arg: Optional[str]) -> Path:
    """Use a caller-supplied task model or cache Google's official lite model."""
    if model_arg:
        path = Path(model_arg).expanduser()
        if not path.is_file():
            raise RuntimeError(f"Pose model file not found: {path}")
        return path

    cache_path = Path.home() / ".cache" / "seated-posture-monitor" / "pose_landmarker_lite.task"
    if cache_path.is_file() and cache_path.stat().st_size > 1_000_000:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".download")
    try:
        with urllib.request.urlopen(POSE_MODEL_URL, timeout=90) as response, temp_path.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        if temp_path.stat().st_size <= 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small; the model download may be incomplete.")
        os.replace(temp_path, cache_path)
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "Could not download MediaPipe's Pose Landmarker lite model. "
            "Check internet access or download pose_landmarker_lite.task manually "
            "and pass it with --model PATH."
        ) from exc
    return cache_path


def run_monitor(args: argparse.Namespace) -> dict[str, Any]:
    # Imports are delayed so --help and unit tests work without MediaPipe installed.
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install with: python -m pip install -r requirements.txt"
        ) from exc

    source = _parse_source(args.source)
    if isinstance(source, str) and not Path(source).is_file():
        raise RuntimeError(f"Video source does not exist or is not a file: {source}")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {args.source}")
    try:
        model_path = resolve_model_path(args.model)
    except Exception:
        capture.release()
        raise

    thresholds = Thresholds(
        min_visibility=args.min_visibility,
        transition_seconds=args.transition_seconds,
        break_min_seconds=args.break_min_seconds,
    )
    tracker = TemporalPostureTracker(thresholds)
    is_camera = isinstance(source, int)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if not is_camera else 0.0
    if not math.isfinite(fps) or fps <= 1.0:
        fps = 30.0
    frame_period = 1.0 / fps
    start_wall = time.monotonic()
    frame_index = 0
    last_timestamp = 0.0
    last_task_timestamp_ms = -1
    frames_read = 0
    bad_reads = 0
    total_read_errors = 0
    preview_active = bool(args.preview)
    pose = None

    try:
        # Tasks VIDEO mode tracks sequential frames and runs synchronously on CPU.
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        pose = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

        while True:
            if is_camera:
                elapsed_wall = time.monotonic() - start_wall
                if elapsed_wall >= args.duration:
                    break

            try:
                ok, frame = capture.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None:
                bad_reads += 1
                total_read_errors += 1
                if is_camera:
                    # A transient camera read failure is skipped, not fatal.
                    if bad_reads >= 50:
                        break
                    time.sleep(0.02)
                    continue
                # At EOF OpenCV's frame position reaches the end. If a decoder
                # reports a transient failure before EOF, try advancing a few
                # frames rather than terminating on the first damaged packet.
                total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                current_frame = float(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
                if total_frames > 0 and current_frame < total_frames - 1 and bad_reads < 5:
                    continue
                break

            bad_reads = 0
            frames_read += 1
            frame_index += 1
            if getattr(frame, "ndim", 0) != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                continue

            if is_camera:
                timestamp = time.monotonic() - start_wall
            else:
                media_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                candidate_ts = media_ms / 1000.0
                if not math.isfinite(candidate_ts) or candidate_ts <= last_timestamp:
                    candidate_ts = frame_index / fps
                timestamp = candidate_ts
                if timestamp > args.duration:
                    break
            last_timestamp = timestamp

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_timestamp_ms = max(last_task_timestamp_ms + 1, int(round(timestamp * 1000)))
                last_task_timestamp_ms = frame_timestamp_ms
                result = pose.detect_for_video(mp_image, frame_timestamp_ms)
                pose_sets = getattr(result, "pose_landmarks", None) or getattr(result, "landmarks", None)
                landmarks = pose_sets[0] if pose_sets else None
                features = extract_features(landmarks, frame.shape, thresholds) if landmarks else None
            except Exception:
                # Bad/corrupt frames and occasional inference exceptions are isolated.
                features = None

            if features is not None:
                tracker.update(timestamp, features)

            if preview_active:
                try:
                    label = tracker.state or "CALIBRATING"
                    color = (40, 180, 40) if label == "GOOD" else (30, 150, 230)
                    cv2.putText(frame, f"Posture: {label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, 2, cv2.LINE_AA)
                    cv2.imshow("Seated Posture Monitor (press q to stop)", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except Exception as exc:
                    print(f"Warning: preview unavailable; continuing headless ({exc})", file=sys.stderr)
                    preview_active = False

        if tracker.pose_frames == 0:
            raise RuntimeError(
                "No usable pose observation: check the camera/video, lighting, framing, and that one person is visible."
            )

        if is_camera:
            session_duration = min(args.duration, max(0.0, time.monotonic() - start_wall))
        else:
            session_duration = min(args.duration, max(last_timestamp, frame_period if frames_read else 0.0))
        tracker.finish(frame_period=frame_period)
        summary = tracker.summary(session_duration, thresholds)
        summary["source"] = args.source
        summary["frames_read"] = frames_read
        summary["read_errors"] = total_read_errors
        _atomic_write_json(Path(args.out), summary)
        return summary
    finally:
        capture.release()
        if pose is not None:
            try:
                pose.close()
            except Exception:
                pass
        if preview_active:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track seated upper-body posture from a webcam or video and write a JSON session summary."
    )
    parser.add_argument("--source", required=True,
                        help="Camera index (for example 0) or path to a video file")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Maximum session/video duration in seconds (default: 60)")
    parser.add_argument("--out", default="posture.json", help="Output JSON path (default: posture.json)")
    parser.add_argument("--model", default=None, help="Optional local Pose Landmarker .task file; otherwise the lite model is cached automatically")
    parser.add_argument("--preview", action="store_true", help="Optional live preview; omitted by default for headless runs")
    parser.add_argument("--min-visibility", type=float, default=0.55,
                        help="Minimum MediaPipe landmark visibility in [0,1] (default: 0.55)")
    parser.add_argument("--transition-seconds", type=float, default=1.0,
                        help="Candidate must persist this long before state changes (default: 1.0)")
    parser.add_argument("--break-min-seconds", type=float, default=2.0,
                        help="Count a non-upright episode as a break at or above this duration (default: 2.0)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a finite number greater than zero")
    if not 0.0 <= args.min_visibility <= 1.0:
        parser.error("--min-visibility must be between 0 and 1")
    if (not math.isfinite(args.transition_seconds) or not math.isfinite(args.break_min_seconds)
            or args.transition_seconds < 0 or args.break_min_seconds < 0):
        parser.error("transition and break durations must be finite and cannot be negative")

    try:
        summary = run_monitor(args)
    except KeyboardInterrupt:
        print("Stopped by user before a complete run.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "output": args.out,
        "posture_score": summary["posture_score"],
        "good_posture_percent": summary["good_posture_percent"],
        "break_count": summary["break_count"],
        "pose_frames": summary["pose_frames"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())