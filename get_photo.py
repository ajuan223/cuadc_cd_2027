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
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One in-memory frame and the metadata required by later stages."""

    image: np.ndarray
    frame_id: int
    capture_monotonic_ns: int
    source_fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameSource(Protocol):
    """Minimal interface consumed by the real-time processing loop."""

    def read_latest(self, timeout_s: float = 1.0) -> FramePacket | None:
        """Return the latest sampled frame, or None on timeout."""

    def close(self) -> None:
        """Release the underlying source."""


def should_sample(frame_id: int, frame_stride: int) -> bool:
    """Return whether a frame should enter the processing buffer."""
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    return frame_id % frame_stride == 0


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
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if buffer_count < 1:
            raise ValueError("buffer_count must be at least 1")
        self.width = width
        self.height = height
        self.fps = float(fps)
        self.frame_stride = frame_stride
        self.buffer_count = buffer_count
        self.exposure_offset_ns = int(round(exposure_offset_s * 1_000_000_000))

        self._buffer = LatestFrameBuffer()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera = None
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._boot_offset_ns = 0
        self._next_frame_id = 0

    @property
    def dropped_frames(self) -> int:
        return self._buffer.dropped_frames

    def start(self) -> None:
        """Start Picamera2 capture and the background acquisition thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is required for the production video stream; "
                "install the Raspberry Pi system package python3-picamera2"
            ) from exc

        camera = Picamera2()
        duration_us = int(round(1_000_000 / self.fps))
        configuration = camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (duration_us, duration_us)},
            buffer_count=self.buffer_count,
            queue=False,
        )
        camera.configure(configuration)
        camera.start()
        self._camera = camera
        self._boot_offset_ns = _boottime_to_monotonic_offset_ns()
        self._stop_event.clear()
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
                    rgb = request.make_array("main")
                    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                finally:
                    request.release()

                frame_id = self._next_frame_id
                self._next_frame_id += 1
                if not should_sample(frame_id, self.frame_stride):
                    continue

                metadata["frame_id"] = frame_id
                packet = FramePacket(
                    image=image,
                    frame_id=frame_id,
                    capture_monotonic_ns=self._capture_time_ns(metadata),
                    source_fps=self.fps,
                    metadata=metadata,
                )
                self._buffer.put(packet)
        except BaseException as exc:
            with self._error_lock:
                self._error = exc
            self._stop_event.set()

    def _capture_time_ns(self, metadata: dict[str, Any]) -> int:
        sensor_ns = metadata.get("SensorTimestamp")
        exposure_us = metadata.get("ExposureTime")
        if isinstance(sensor_ns, (int, float)):
            exposure_midpoint_ns = 0
            if isinstance(exposure_us, (int, float)):
                exposure_midpoint_ns = int(round(float(exposure_us) * 500.0))
            return (
                int(sensor_ns)
                + exposure_midpoint_ns
                + self._boot_offset_ns
                + self.exposure_offset_ns
            )
        return time.monotonic_ns() + self.exposure_offset_ns

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
        """Stop capture and release Picamera2 resources."""
        self._stop_event.set()
        camera = self._camera
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        self._camera = None
        self._thread = None
        self.clear()

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