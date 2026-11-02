"""IMX708 real-time frame acquisition and in-memory frame sampling.

Production input only:
    IMX708 -> Picamera2 -> FramePacket -> latest-frame buffer

No image is written to disk in this module.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    """An owned, read-only frame plus timing and camera metadata.

    If the supplied image is writable, the packet copies it so later changes to
    the caller's array cannot alter the frame while downstream code is running.
    """

    image: np.ndarray
    frame_id: int
    capture_monotonic_ns: int
    configured_fps: float
    source_fps: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if self.image.dtype != np.uint8:
            raise ValueError("image dtype must be uint8")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.capture_monotonic_ns < 0:
            raise ValueError("capture_monotonic_ns must be non-negative")
        if self.configured_fps <= 0:
            raise ValueError("configured_fps must be positive")
        if self.source_fps is not None and self.source_fps <= 0:
            raise ValueError("source_fps must be positive when provided")

        image = self.image
        if image.flags.writeable:
            image = image.copy()
        image.setflags(write=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "metadata", dict(self.metadata))


class FrameSource(Protocol):
    """Minimal interface consumed by the real-time processing loop."""

    def read_latest(self, timeout_s: float = 1.0) -> FramePacket | None:
        """Return the latest sampled frame, or None on timeout."""

    def close(self) -> None:
        """Release the underlying source."""


class FpsEstimator:
    """Estimate achieved capture FPS from recent monotonic timestamps."""

    def __init__(self, max_samples: int = 31) -> None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        self._timestamps: deque[int] = deque(maxlen=max_samples)
        self._fps: float | None = None

    def add(self, capture_monotonic_ns: int) -> float | None:
        if capture_monotonic_ns < 0:
            raise ValueError("capture_monotonic_ns must be non-negative")
        if self._timestamps and capture_monotonic_ns <= self._timestamps[-1]:
            self._timestamps.clear()
            self._fps = None
        self._timestamps.append(capture_monotonic_ns)
        if len(self._timestamps) < 2:
            return None
        elapsed_ns = self._timestamps[-1] - self._timestamps[0]
        if elapsed_ns <= 0:
            self._fps = None
            return None
        self._fps = (len(self._timestamps) - 1) * 1_000_000_000.0 / elapsed_ns
        return self._fps

    @property
    def fps(self) -> float | None:
        return self._fps


def should_sample(frame_id: int, frame_stride: int) -> bool:
    """Return whether a frame should enter the processing buffer."""
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    return frame_id % frame_stride == 0


def convert_capture_timestamp_ns(
    *,
    sensor_timestamp_ns: int | float | None,
    exposure_us: int | float | None,
    boottime_to_monotonic_offset_ns: int,
    exposure_offset_ns: int,
    receive_monotonic_ns: int,
    max_future_ns: int = 500_000_000,
    max_age_ns: int = 5_000_000_000,
) -> tuple[int, str, dict[str, Any]]:
    """Convert a libcamera boot-time timestamp into CLOCK_MONOTONIC time.

    libcamera reports ``SensorTimestamp`` in the CLOCK_BOOTTIME domain. The
    converted value is validated against the receive time; implausible values
    fall back to receive monotonic time and are marked as untrusted.
    """
    if receive_monotonic_ns < 0:
        raise ValueError("receive_monotonic_ns must be non-negative")
    if max_future_ns < 0 or max_age_ns <= 0:
        raise ValueError("timestamp validation windows are invalid")

    fallback_ns = receive_monotonic_ns + exposure_offset_ns
    diagnostics: dict[str, Any] = {
        "basis": "receive_monotonic",
        "sensor_timestamp_domain": (
            "CLOCK_BOOTTIME" if sensor_timestamp_ns is not None else None
        ),
        "boottime_to_monotonic_offset_ns": boottime_to_monotonic_offset_ns,
        "sensor_to_receive_delta_ns": None,
        "validated": False,
    }
    if not isinstance(sensor_timestamp_ns, (int, float)) or sensor_timestamp_ns <= 0:
        return fallback_ns, "receive_monotonic", diagnostics

    exposure_midpoint_ns = 0
    if isinstance(exposure_us, (int, float)) and exposure_us > 0:
        exposure_midpoint_ns = int(round(float(exposure_us) * 500.0))
    candidate_ns = (
        int(sensor_timestamp_ns)
        + exposure_midpoint_ns
        + boottime_to_monotonic_offset_ns
        + exposure_offset_ns
    )
    delta_ns = receive_monotonic_ns - candidate_ns
    diagnostics["sensor_to_receive_delta_ns"] = delta_ns
    if -max_future_ns <= delta_ns <= max_age_ns:
        diagnostics["basis"] = "libcamera_sensor_exposure_midpoint"
        diagnostics["validated"] = True
        return candidate_ns, "libcamera_sensor_exposure_midpoint", diagnostics

    diagnostics["basis"] = "receive_monotonic_fallback"
    return fallback_ns, "receive_monotonic_fallback", diagnostics


class LatestFrameBuffer:
    """A one-frame mailbox that discards stale frames instead of queuing them."""

    def __init__(self) -> None:
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._dropped_frames = 0

    def put(self, packet: FramePacket) -> None:
        """Store a frame, replacing and counting any frame still in the buffer."""
        while True:
            try:
                self._queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue
                with self._lock:
                    self._dropped_frames += 1

    def get_latest(self, timeout_s: float = 1.0) -> FramePacket | None:
        """Wait for a frame and return it, or None when the timeout expires."""
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    def clear(self) -> None:
        """Discard an unprocessed frame without shutting down the source."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            return


def _boottime_to_monotonic_offset_ns() -> int:
    """Convert Linux CLOCK_BOOTTIME timestamps into CLOCK_MONOTONIC time."""
    if not sys.platform.startswith("linux") or not hasattr(time, "CLOCK_BOOTTIME"):
        return 0
    clock_gettime_ns = getattr(time, "clock_gettime_ns", None)
    if clock_gettime_ns is None:
        return 0
    return time.monotonic_ns() - clock_gettime_ns(time.CLOCK_BOOTTIME)


class Picamera2FrameSource:
    """Capture IMX708 frames and publish only the latest sampled frame."""

    def __init__(
        self,
        *,
        width: int = 2304,
        height: int = 1296,
        fps: float = 30.0,
        frame_stride: int = 1,
        buffer_count: int = 4,
        exposure_offset_s: float = 0.0,
        close_timeout_s: float = 5.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if buffer_count < 1:
            raise ValueError("buffer_count must be at least 1")
        if close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")

        self.width = width
        self.height = height
        self.fps = float(fps)
        self.frame_stride = frame_stride
        self.buffer_count = buffer_count
        self.exposure_offset_ns = int(round(exposure_offset_s * 1_000_000_000))
        self.close_timeout_s = float(close_timeout_s)

        self._buffer = LatestFrameBuffer()
        self._fps_estimator = FpsEstimator()
        self._stop_event = threading.Event()
        self._capture_done = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._camera = None
        self._error: BaseException | None = None
        self._boot_offset_ns = 0
        self._next_frame_id = 0
        self._last_clock_diagnostics: dict[str, Any] = {}

    @property
    def dropped_frames(self) -> int:
        return self._buffer.dropped_frames

    @property
    def measured_fps(self) -> float | None:
        return self._fps_estimator.fps

    @property
    def last_clock_diagnostics(self) -> dict[str, Any]:
        with self._error_lock:
            return dict(self._last_clock_diagnostics)

    def start(self) -> None:
        """Start Picamera2 capture and the background acquisition thread."""
        with self._lifecycle_lock:
            if self._thread is not None:
                if self._thread.is_alive():
                    return
                raise RuntimeError("capture thread stopped unexpectedly; call close() first")
            try:
                from picamera2 import Picamera2
            except ImportError as exc:
                raise RuntimeError(
                    "Picamera2 is required for the production video stream; "
                    "install the Raspberry Pi system package python3-picamera2"
                ) from exc

            camera = Picamera2()
            try:
                duration_us = int(round(1_000_000 / self.fps))
                configuration = camera.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameDurationLimits": (duration_us, duration_us)},
                    buffer_count=self.buffer_count,
                    queue=False,
                )
                camera.configure(configuration)
                camera.start()
            except BaseException:
                try:
                    camera.close()
                except Exception:
                    pass
                raise

            self._camera = camera
            self._boot_offset_ns = _boottime_to_monotonic_offset_ns()
            self._fps_estimator = FpsEstimator()
            self._stop_event.clear()
            self._capture_done.clear()
            self._error = None
            self._last_clock_diagnostics = {}
            self._next_frame_id = 0
            self._buffer.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="imx708-capture",
                daemon=True,
            )
            self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                request = self._camera.capture_request()
                try:
                    metadata = dict(request.get_metadata() or {})
                    receive_monotonic_ns = time.monotonic_ns()
                    rgb = request.make_array("main")
                    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                finally:
                    request.release()

                frame_id = self._next_frame_id
                self._next_frame_id += 1
                capture_ns, _basis, clock_diagnostics = convert_capture_timestamp_ns(
                    sensor_timestamp_ns=metadata.get("SensorTimestamp"),
                    exposure_us=metadata.get("ExposureTime"),
                    boottime_to_monotonic_offset_ns=self._boot_offset_ns,
                    exposure_offset_ns=self.exposure_offset_ns,
                    receive_monotonic_ns=receive_monotonic_ns,
                )
                with self._error_lock:
                    self._last_clock_diagnostics = dict(clock_diagnostics)

                measured_fps = self._fps_estimator.add(capture_ns)
                if not should_sample(frame_id, self.frame_stride):
                    continue

                image.setflags(write=False)
                metadata["frame_id"] = frame_id
                metadata["configured_fps"] = self.fps
                metadata["measured_fps"] = measured_fps
                metadata["capture_clock"] = clock_diagnostics
                packet = FramePacket(
                    image=image,
                    frame_id=frame_id,
                    capture_monotonic_ns=capture_ns,
                    configured_fps=self.fps,
                    source_fps=measured_fps,
                    metadata=metadata,
                )
                self._buffer.put(packet)
        except BaseException as exc:
            if not self._stop_event.is_set():
                with self._error_lock:
                    self._error = exc
                self._stop_event.set()
        finally:
            self._capture_done.set()

    def read_latest(self, timeout_s: float = 1.0) -> FramePacket | None:
        packet = self._buffer.get_latest(timeout_s)
        if packet is not None:
            return packet
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("IMX708 capture failed") from error
        return None

    def close(self) -> None:
        """Stop capture, wait for the capture thread, then release Picamera2."""
        with self._lifecycle_lock:
            thread = self._thread
            camera = self._camera
            if thread is None and camera is None:
                return

            self._stop_event.set()
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass

            if thread is not None:
                thread.join(timeout=self.close_timeout_s)
                if not self._capture_done.is_set():
                    if camera is not None:
                        try:
                            camera.close()
                        except Exception:
                            pass
                    thread.join(timeout=1.0)
                if thread.is_alive() or not self._capture_done.is_set():
                    raise RuntimeError("capture thread did not stop cleanly")

            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            self._thread = None
            self._camera = None
            self._buffer.clear()

    def clear(self) -> None:
        self._buffer.clear()

    def __enter__(self) -> "Picamera2FrameSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def get_photo(source: FrameSource, timeout_s: float = 1.0) -> FramePacket | None:
    """Return the latest sampled frame from an already-running source."""
    return source.read_latest(timeout_s)