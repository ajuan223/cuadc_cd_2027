import numpy as np
import pytest

from get_photo import (
    FramePacket,
    LatestFrameBuffer,
    get_photo,
    should_sample,
)


def packet(frame_id: int) -> FramePacket:
    return FramePacket(
        image=np.full((4, 5, 3), frame_id, dtype=np.uint8),
        frame_id=frame_id,
        capture_monotonic_ns=frame_id * 1_000_000,
        source_fps=30.0,
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