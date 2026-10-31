"""Low-latency live capture for IMX708/Picamera2 and USB/OpenCV cameras."""

from __future__ import annotations

import threading
import time

import cv2


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if all(hasattr(value, key) for key in ("x", "y", "width", "height")):
        return [int(value.x), int(value.y), int(value.width), int(value.height)]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _boottime_to_monotonic_offset() -> float:
    """Seconds to add to a libcamera CLOCK_BOOTTIME timestamp."""
    clock_id = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_id is None or not hasattr(time, "clock_gettime"):
        return 0.0
    return time.monotonic() - time.clock_gettime(clock_id)


class _LatestFrameBase:
    def _publish(self, captured_at, frame, metadata=None) -> None:
        if self.undistorter is not None:
            frame = self.undistorter.apply(frame)
        with self.condition:
            self.frame_index += 1
            self.latest_metadata = _json_safe(metadata or {})
            self.latest = (
                self.frame_index, float(captured_at), frame,
                dict(self.latest_metadata),
            )
            self.condition.notify_all()

    def read_latest(self, after_index: int = -1, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (
                (self.latest is None or self.latest[0] <= after_index)
                and not self.stopped
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.latest is None or self.latest[0] <= after_index:
                if self.error:
                    raise RuntimeError(self.error)
                return None
            return self.latest


class LatestFrameCamera(_LatestFrameBase):
    """Continuously drain a UVC camera and retain only its newest frame."""

    backend = "opencv"

    def __init__(
        self, index: int, width: int, height: int, fps: int, undistorter=None,
        exposure_offset_s: float = 0.0,
    ):
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open OpenCV camera {index}")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or float(fps)
        self.undistorter = undistorter
        self.exposure_offset_s = float(exposure_offset_s)
        self.settings = {"backend": self.backend, "index": int(index)}
        self.condition = threading.Condition()
        self.latest = None
        self.latest_metadata = {}
        self.frame_index = -1
        self.error = None
        self.stopped = False
        self.thread = threading.Thread(
            target=self._capture_loop, name="opencv-camera", daemon=True
        )
        self.thread.start()

    def _capture_loop(self) -> None:
        while True:
            with self.condition:
                if self.stopped:
                    return
            started_ns = time.monotonic_ns()
            ok, frame = self.capture.read()
            finished_ns = time.monotonic_ns()
            if not ok:
                with self.condition:
                    self.error = "camera read failed"
                    self.stopped = True
                    self.condition.notify_all()
                return
            captured_at = (
                (started_ns + finished_ns) * 0.5e-9 - self.exposure_offset_s
            )
            self._publish(captured_at, frame, {"CaptureTimestampBasis": "uvc_read_midpoint"})

    def close(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.capture.release()
        self.thread.join(timeout=2.0)


class LatestFramePicamera2(_LatestFrameBase):
    """Capture full-FoV IMX708 frames with libcamera exposure timestamps."""

    backend = "picamera2"

    def __init__(
        self, index: int, width: int, height: int, fps: int, undistorter=None,
        exposure_offset_s: float = 0.0, *, sensor_width: int = 2304,
        sensor_height: int = 1296, sensor_bit_depth: int = 10,
        ae: bool = True, exposure_us: int = 1000, analogue_gain: float = 1.0,
        exposure_value: float = -0.75, awb: bool = True,
        red_gain: float = 1.0, blue_gain: float = 1.0,
        lens_position: float = 0.0, brightness: float = 0.0,
        contrast: float = 1.3, saturation: float = 1.0,
        sharpness: float = 1.0,
    ) -> None:
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is unavailable; install python3-picamera2 and create "
                "the venv with --system-site-packages"
            ) from exc
        if min(width, height, sensor_width, sensor_height, fps) <= 0:
            raise ValueError("camera dimensions and FPS must be positive")
        self.picam2 = Picamera2(index)
        duration_us = int(round(1_000_000 / fps))
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            sensor={
                "output_size": (sensor_width, sensor_height),
                "bit_depth": sensor_bit_depth,
            },
            controls={"FrameDurationLimits": (duration_us, duration_us)},
            buffer_count=4,
            queue=False,
        )
        self.picam2.configure(config)
        supported = set(self.picam2.camera_controls)
        requested = {
            "AeEnable": bool(ae),
            "AwbEnable": bool(awb),
            "FrameDurationLimits": (duration_us, duration_us),
            "Brightness": float(brightness),
            "Contrast": float(contrast),
            "Saturation": float(saturation),
            "Sharpness": float(sharpness),
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(lens_position),
        }
        if ae:
            requested["ExposureValue"] = float(exposure_value)
        else:
            if exposure_us >= duration_us:
                raise ValueError("manual exposure must be shorter than one frame")
            requested["ExposureTime"] = int(exposure_us)
            requested["AnalogueGain"] = float(analogue_gain)
        if not awb:
            requested["ColourGains"] = (float(red_gain), float(blue_gain))
        applied = {key: value for key, value in requested.items() if key in supported}
        self.picam2.set_controls(applied)
        self.picam2.start()

        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.undistorter = undistorter
        self.exposure_offset_s = float(exposure_offset_s)
        self.boottime_to_monotonic_s = _boottime_to_monotonic_offset()
        self.settings = {
            "backend": self.backend,
            "index": int(index),
            "size": [self.width, self.height],
            "sensor_mode": [int(sensor_width), int(sensor_height), int(sensor_bit_depth)],
            "fps": self.fps,
            "ae": bool(ae),
            "exposure_value": float(exposure_value) if ae else None,
            "exposure_us": None if ae else int(exposure_us),
            "analogue_gain": None if ae else float(analogue_gain),
            "awb": bool(awb),
            "colour_gains": None if awb else [float(red_gain), float(blue_gain)],
            "lens_position": float(lens_position),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "saturation": float(saturation),
            "sharpness": float(sharpness),
            "applied_controls": sorted(applied),
        }
        self.condition = threading.Condition()
        self.latest = None
        self.latest_metadata = {}
        self.frame_index = -1
        self.error = None
        self.stopped = False
        self.thread = threading.Thread(
            target=self._capture_loop, name="picamera2-imx708", daemon=True
        )
        self.thread.start()

    @staticmethod
    def exposure_midpoint_ns(metadata: dict) -> float | None:
        """libcamera timestamp is first-row exposure start, so add half exposure."""
        sensor_ns = metadata.get("SensorTimestamp")
        if not isinstance(sensor_ns, (int, float)):
            return None
        exposure_us = metadata.get("ExposureTime")
        return float(sensor_ns) + (
            float(exposure_us) * 500.0
            if isinstance(exposure_us, (int, float)) else 0.0
        )

    def _capture_loop(self) -> None:
        try:
            while True:
                with self.condition:
                    if self.stopped:
                        return
                fallback_started_ns = time.monotonic_ns()
                request = self.picam2.capture_request()
                try:
                    frame = request.make_array("main").copy()
                    metadata = dict(request.get_metadata())
                finally:
                    request.release()
                fallback_finished_ns = time.monotonic_ns()

                midpoint_ns = self.exposure_midpoint_ns(metadata)
                if midpoint_ns is not None:
                    captured_at = (
                        midpoint_ns * 1e-9
                        + self.boottime_to_monotonic_s
                        - self.exposure_offset_s
                    )
                    basis = "libcamera_sensor_exposure_midpoint"
                else:
                    captured_at = (
                        (fallback_started_ns + fallback_finished_ns) * 0.5e-9
                        - self.exposure_offset_s
                    )
                    basis = "capture_request_midpoint_fallback"
                metadata["CaptureTimestampBasis"] = basis
                metadata["CaptureMonotonicNs"] = round(captured_at * 1e9)
                self._publish(captured_at, frame, metadata)
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
                self.stopped = True
                self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=2.0)
        self.picam2.stop()
        self.picam2.close()


def add_camera_arguments(parser) -> None:
    """Add the live IMX708 settings validated with the tuning tool."""
    parser.add_argument(
        "--camera-backend", choices=("auto", "picamera2", "opencv"),
        default="auto", help="auto prefers Picamera2 and falls back to OpenCV",
    )
    parser.add_argument("--sensor-width", type=int, default=2304)
    parser.add_argument("--sensor-height", type=int, default=1296)
    parser.add_argument("--sensor-bit-depth", type=int, default=10)
    parser.set_defaults(camera_ae=True, camera_awb=True)
    parser.add_argument("--camera-ae", dest="camera_ae", action="store_true")
    parser.add_argument("--camera-no-ae", dest="camera_ae", action="store_false")
    parser.add_argument("--camera-exposure-us", type=int, default=1000)
    parser.add_argument("--camera-analogue-gain", type=float, default=1.0)
    parser.add_argument("--camera-ev", type=float, default=-0.75)
    parser.add_argument("--camera-awb", dest="camera_awb", action="store_true")
    parser.add_argument("--camera-no-awb", dest="camera_awb", action="store_false")
    parser.add_argument("--camera-red-gain", type=float, default=1.0)
    parser.add_argument("--camera-blue-gain", type=float, default=1.0)
    parser.add_argument("--camera-lens-position", type=float, default=0.0)
    parser.add_argument("--camera-brightness", type=float, default=0.0)
    parser.add_argument("--camera-contrast", type=float, default=1.3)
    parser.add_argument("--camera-saturation", type=float, default=1.0)
    parser.add_argument("--camera-sharpness", type=float, default=1.0)


def build_live_camera(args, undistorter=None, exposure_offset_s: float = 0.0):
    """Open a live camera. Explicit Picamera2 requests never fall back silently."""
    source = int(args.source)
    backend = getattr(args, "camera_backend", "auto")
    if backend in ("auto", "picamera2"):
        try:
            return LatestFramePicamera2(
                source, args.width, args.height, args.fps, undistorter,
                exposure_offset_s,
                sensor_width=args.sensor_width,
                sensor_height=args.sensor_height,
                sensor_bit_depth=args.sensor_bit_depth,
                ae=args.camera_ae,
                exposure_us=args.camera_exposure_us,
                analogue_gain=args.camera_analogue_gain,
                exposure_value=args.camera_ev,
                awb=args.camera_awb,
                red_gain=args.camera_red_gain,
                blue_gain=args.camera_blue_gain,
                lens_position=args.camera_lens_position,
                brightness=args.camera_brightness,
                contrast=args.camera_contrast,
                saturation=args.camera_saturation,
                sharpness=args.camera_sharpness,
            )
        except RuntimeError as exc:
            if backend == "picamera2":
                raise
            print(f"Picamera2 unavailable ({exc}); falling back to OpenCV")
    return LatestFrameCamera(
        source, args.width, args.height, args.fps, undistorter,
        exposure_offset_s,
    )
