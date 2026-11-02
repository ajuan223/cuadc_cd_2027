import threading

import numpy as np
import pytest

from get_photo import (
    FpsEstimator,
    FramePacket,
    LatestFrameBuffer,
    Picamera2FrameSource,
    convert_capture_timestamp_ns,
    get_photo,
    should_sample,
)


def packet(frame_id: int, fps: float | None = 30.0) -> FramePacket:
    return FramePacket(
        image=np.full((4, 5, 3), frame_id, dtype=np.uint8),
        frame_id=frame_id,
        capture_monotonic_ns=frame_id * 1_000_000,
        configured_fps=30.0,
        source_fps=fps,
        metadata={"synthetic": True},
    )


def test_should_sample_uses_frame_stride() -> None:
    assert [frame_id for frame_id in range(8) if should_sample(frame_id, 2)] == [0, 2, 4, 6]
    assert all(should_sample(frame_id, 1) for frame_id in range(4))


def test_should_sample_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        should_sample(-1, 1)
    with pytest.raises(ValueError, match="frame_stride"):
        should_sample(0, 0)


def test_frame_packet_owns_read_only_image() -> None:
    source = np.zeros((4, 5, 3), dtype=np.uint8)
    frame = FramePacket(
        image=source,
        frame_id=1,
        capture_monotonic_ns=10,
        configured_fps=30.0,
        source_fps=None,
    )

    source[:, :] = 255

    assert not frame.image.flags.writeable
    assert np.count_nonzero(frame.image) == 0
    with pytest.raises(ValueError, match="read-only"):
        frame.image[0, 0, 0] = 1


def test_frame_packet_validates_image_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="shape"):
        FramePacket(
            image=np.zeros((4, 5), dtype=np.uint8),
            frame_id=0,
            capture_monotonic_ns=0,
            configured_fps=30.0,
            source_fps=None,
        )
    with pytest.raises(ValueError, match="dtype"):
        FramePacket(
            image=np.zeros((4, 5, 3), dtype=np.float32),
            frame_id=0,
            capture_monotonic_ns=0,
            configured_fps=30.0,
            source_fps=None,
        )


def test_latest_frame_buffer_replaces_stale_frame() -> None:
    buffer = LatestFrameBuffer()
    buffer.put(packet(1))
    buffer.put(packet(2))

    latest = buffer.get_latest(timeout_s=0)

    assert latest is not None
    assert latest.frame_id == 2
    assert buffer.dropped_frames == 1
    assert latest.image.dtype == np.uint8
    assert latest.image.shape == (4, 5, 3)


def test_latest_frame_buffer_times_out_without_copying_images() -> None:
    buffer = LatestFrameBuffer()

    assert buffer.get_latest(timeout_s=0) is None
    assert buffer.dropped_frames == 0


def test_get_photo_reads_from_source_interface() -> None:
    class FakeSource:
        def __init__(self) -> None:
            self.closed = False

        def read_latest(self, timeout_s: float = 1.0):
            return packet(7)

        def close(self) -> None:
            self.closed = True

    source = FakeSource()

    result = get_photo(source, timeout_s=0.1)

    assert result is not None
    assert result.frame_id == 7
    assert source.closed is False


def test_fps_estimator_uses_measured_timestamps() -> None:
    estimator = FpsEstimator(max_samples=5)
    assert estimator.add(0) is None

    fps = estimator.add(33_333_333)

    assert fps == pytest.approx(30.0000003)
    assert estimator.fps == pytest.approx(fps)


def test_capture_timestamp_converts_boottime_to_monotonic() -> None:
    capture_ns, basis, diagnostics = convert_capture_timestamp_ns(
        sensor_timestamp_ns=1_000_000_000,
        exposure_us=1_000,
        boottime_to_monotonic_offset_ns=10_000_000_000,
        exposure_offset_ns=2_000_000,
        receive_monotonic_ns=11_100_000_000,
    )

    assert capture_ns == 11_002_500_000
    assert basis == "libcamera_sensor_exposure_midpoint"
    assert diagnostics["sensor_timestamp_domain"] == "CLOCK_BOOTTIME"
    assert diagnostics["validated"] is True
    assert diagnostics["sensor_to_receive_delta_ns"] == 97_500_000


def test_capture_timestamp_falls_back_when_sensor_time_is_implausible() -> None:
    capture_ns, basis, diagnostics = convert_capture_timestamp_ns(
        sensor_timestamp_ns=100_000_000_000,
        exposure_us=1_000,
        boottime_to_monotonic_offset_ns=0,
        exposure_offset_ns=0,
        receive_monotonic_ns=1_000_000_000,
    )

    assert capture_ns == 1_000_000_000
    assert basis == "receive_monotonic_fallback"
    assert diagnostics["validated"] is False


def test_close_before_start_is_idempotent() -> None:
    source = Picamera2FrameSource()

    source.close()
    source.close()
def test_close_wakes_blocked_capture_thread() -> None:
    class BlockingCamera:
        def __init__(self) -> None:
            self.request_entered = threading.Event()
            self.stop_called = threading.Event()
            self.close_called = False

        def capture_request(self):
            self.request_entered.set()
            self.stop_called.wait(timeout=2)
            raise RuntimeError("camera stopped")

        def stop(self) -> None:
            self.stop_called.set()

        def close(self) -> None:
            self.close_called = True

    source = Picamera2FrameSource(close_timeout_s=1.0)
    camera = BlockingCamera()
    source._camera = camera
    source._stop_event.clear()
    source._capture_done.clear()
    source._thread = threading.Thread(target=source._capture_loop)
    source._thread.start()
    assert camera.request_entered.wait(timeout=1)

    source.close()

    assert source._thread is None
    assert source._capture_done.is_set()
    assert camera.close_called is True
    assert source._error is None
