"""下视相机的像素坐标到目标 GPS 坐标解算，并可从 MAVLink 读取飞控遥测。"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from camera_calibration import CameraCalibration, CameraUndistorter


DEFAULT_CAMERA_CALIBRATION = CameraCalibration.load()
CALIBRATION_WIDTH = DEFAULT_CAMERA_CALIBRATION.width
CALIBRATION_HEIGHT = DEFAULT_CAMERA_CALIBRATION.height
CAMERA_MATRIX = DEFAULT_CAMERA_CALIBRATION.matrix.copy()
DISTORTION = DEFAULT_CAMERA_CALIBRATION.distortion.copy()
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
MIN_GROUND_RAY_DOWN = 0.15
BOOT_TIME_MODULUS_MS = 1 << 32
BOOT_TIME_HALF_RANGE_MS = 1 << 31
BOOT_RESET_BACKWARD_MS = 2_000
REMOTE_BOOT_TIME_LIMIT_S = 100_000_000.0


# 对齐相机单调时钟与飞控时间并处理time_boot_ms回绕。
class BootTimeUnwrapper:
    """Expand MAVLink's uint32 boot milliseconds and detect a reboot."""

    def __init__(self) -> None:
        self.last_raw_ms: int | None = None
        self.wraps = 0

    def unwrap(self, raw_ms: int) -> tuple[float | None, bool]:
        raw_ms = int(raw_ms)
        if raw_ms <= 0:
            return None, False
        raw_ms &= BOOT_TIME_MODULUS_MS - 1
        reset = False
        wrapped = False
        if self.last_raw_ms is not None:
            backward = self.last_raw_ms - raw_ms
            if backward > BOOT_TIME_HALF_RANGE_MS:
                self.wraps += 1
                wrapped = True
            elif backward > BOOT_RESET_BACKWARD_MS:
                self.wraps = 0
                reset = True
        if wrapped or reset or self.last_raw_ms is None or raw_ms > self.last_raw_ms:
            self.last_raw_ms = raw_ms
        return (raw_ms + self.wraps * BOOT_TIME_MODULUS_MS) / 1000.0, reset


class TelemetryClock:
    """Map flight-controller boot time onto the Pi monotonic clock.

    TIMESYNC round trips are preferred.  A recent low-delay envelope of
    time_boot_ms packet arrivals is a fully offline fallback.
    """

    def __init__(
        self, arrival_window_s: float = 15.0, timesync_window_s: float = 5.0,
        minimum_samples: int = 3,
    ) -> None:
        self.arrival_window_s = float(arrival_window_s)
        self.timesync_window_s = float(timesync_window_s)
        self.minimum_samples = int(minimum_samples)
        self.arrival_offsets = deque()
        self.timesync_offsets = deque()
        self.last_boot_s: float | None = None

    @staticmethod
    def _prune(samples: deque, newest_local_s: float, window_s: float) -> None:
        while samples and newest_local_s - samples[0][0] > window_s:
            samples.popleft()

    def reset(self) -> None:
        self.arrival_offsets.clear()
        self.timesync_offsets.clear()
        self.last_boot_s = None

    def observe_boot(self, boot_s: float | None, received_local_s: float) -> None:
        if boot_s is None or not math.isfinite(boot_s):
            return
        self.last_boot_s = float(boot_s)
        received_local_s = float(received_local_s)
        self.arrival_offsets.append(
            (received_local_s, received_local_s - float(boot_s))
        )
        self._prune(self.arrival_offsets, received_local_s, self.arrival_window_s)
        self._prune(self.timesync_offsets, received_local_s, self.timesync_window_s)

    def observe_timesync(
        self, sent_local_ns: int, remote_response_ns: int,
        received_local_ns: int, maximum_rtt_s: float,
    ) -> bool:
        sent_local_s = int(sent_local_ns) * 1e-9
        received_local_s = int(received_local_ns) * 1e-9
        remote_s = int(remote_response_ns) * 1e-9
        rtt_s = received_local_s - sent_local_s
        if (
            not 0.0 <= rtt_s <= maximum_rtt_s
            or not 0.0 < remote_s < REMOTE_BOOT_TIME_LIMIT_S
        ):
            return False
        # 忽略无法直接映射time_boot_ms的Unix纪元纳秒响应。
        if self.last_boot_s is not None and abs(remote_s - self.last_boot_s) > 30.0:
            return False
        midpoint_s = (sent_local_s + received_local_s) * 0.5
        self.timesync_offsets.append(
            (received_local_s, midpoint_s - remote_s, rtt_s)
        )
        self._prune(self.timesync_offsets, received_local_s, self.timesync_window_s)
        return True

    def _offset_and_basis(self) -> tuple[float | None, str]:
        if len(self.timesync_offsets) >= self.minimum_samples:
            ordered = sorted(self.timesync_offsets, key=lambda sample: sample[2])
            count = max(self.minimum_samples, (len(ordered) + 1) // 2)
            offset = statistics.median(sample[1] for sample in ordered[:count])
            return float(offset), "timesync"
        if len(self.arrival_offsets) >= self.minimum_samples:
            ordered = sorted(sample[1] for sample in self.arrival_offsets)
            # 对最低延迟样本取中位数以抑制串口排队和异常包。
            count = max(self.minimum_samples, (len(ordered) + 4) // 5)
            return float(statistics.median(ordered[:count])), "boot-receive"
        return None, "receive"

    def local_to_boot(self, local_s: float) -> tuple[float | None, str]:
        self._prune(self.arrival_offsets, float(local_s), self.arrival_window_s)
        self._prune(self.timesync_offsets, float(local_s), self.timesync_window_s)
        offset, basis = self._offset_and_basis()
        return (None if offset is None else float(local_s) - offset), basis

    def status(self, local_s: float | None = None) -> dict:
        offset, basis = self._offset_and_basis()
        best_rtt = min(
            (sample[2] for sample in self.timesync_offsets), default=None
        )
        return {
            "time_basis": basis,
            "sync_locked": offset is not None,
            "clock_offset_s": None if offset is None else round(offset, 9),
            "boot_arrival_samples": len(self.arrival_offsets),
            "timesync_samples": len(self.timesync_offsets),
            "best_timesync_rtt_ms": (
                None if best_rtt is None else round(best_rtt * 1000.0, 3)
            ),
            "last_timesync_age_ms": (
                None if not self.timesync_offsets or local_s is None
                else round(max(0.0, float(local_s) - self.timesync_offsets[-1][0]) * 1000.0, 3)
            ),
        }


def scaled_camera_matrix(
    width: int, height: int, calibration: CameraCalibration | None = None,
) -> np.ndarray:
    return (calibration or DEFAULT_CAMERA_CALIBRATION).matrix_for_size(width, height)


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


# 定义单帧定位所需的飞机位姿和目标位置数据。
@dataclass(frozen=True)
class AircraftPose:
    latitude_deg: float
    longitude_deg: float
    height_agl_m: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.latitude_deg, self.longitude_deg, self.height_agl_m,
            self.roll_deg, self.pitch_deg, self.yaw_deg,
        )):
            raise ValueError("aircraft pose values must be finite")
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude must be within [-90, 90]")
        if not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude must be within [-180, 180]")
        if self.height_agl_m <= 0:
            raise ValueError("height_agl_m must be positive")

    @classmethod
    def from_mapping(cls, data: dict) -> "AircraftPose":
        return cls(
            float(data.get("latitude_deg", data.get("lat"))),
            float(data.get("longitude_deg", data.get("lon"))),
            float(data.get("height_agl_m", data.get("relative_alt_m"))),
            float(data.get("roll_deg", data.get("roll", 0.0))),
            float(data.get("pitch_deg", data.get("pitch", 0.0))),
            float(data.get("yaw_deg", data.get("yaw", data.get("heading_deg", 0.0)))),
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TargetLocation:
    latitude_deg: float
    longitude_deg: float
    north_m: float
    east_m: float
    ground_distance_m: float
    slant_range_m: float

    def as_dict(self) -> dict:
        values = asdict(self)
        return {
            key: round(float(value), 9 if key.endswith("_deg") else 4)
            for key, value in values.items()
        }


def offset_wgs84(latitude_deg: float, longitude_deg: float, north_m: float, east_m: float):
    """把局部 NED 中的北向/东向米制偏移换算为 WGS84 经纬度。"""
    latitude = math.radians(latitude_deg)
    sine = math.sin(latitude)
    denominator = math.sqrt(1.0 - WGS84_E2 * sine * sine)
    radius_east = WGS84_A / denominator
    radius_north = WGS84_A * (1.0 - WGS84_E2) / denominator**3
    cosine = math.cos(latitude)
    if abs(cosine) < 1e-9:
        raise ValueError("longitude is undefined at the poles")
    target_latitude = latitude + north_m / radius_north
    target_longitude = math.radians(longitude_deg) + east_m / (radius_east * cosine)
    longitude = (math.degrees(target_longitude) + 180.0) % 360.0 - 180.0
    return math.degrees(target_latitude), longitude


class TargetGeolocator:
    """将去畸变或原始像素射线与局部水平地面相交，得到目标经纬度。"""

    CAMERA_TO_BODY_NADIR = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    def __init__(
        self, width: int, height: int, rectified: bool,
        mount_roll_deg: float = 0.0, mount_pitch_deg: float = 0.0,
        mount_yaw_deg: float = 0.0,
        camera_offset_forward_m: float = 0.0,
        camera_offset_right_m: float = 0.0,
        camera_offset_down_m: float = 0.0,
        calibration: CameraCalibration | None = None,
    ) -> None:
        self.calibration = calibration or DEFAULT_CAMERA_CALIBRATION
        self.matrix = scaled_camera_matrix(width, height, self.calibration)
        self.inverse_matrix = np.linalg.inv(self.matrix)
        self.distortion = (
            None if rectified or not self.calibration.uses_distortion
            else self.calibration.distortion
        )
        mount = (
            _rz(math.radians(mount_yaw_deg))
            @ _ry(math.radians(mount_pitch_deg))
            @ _rx(math.radians(mount_roll_deg))
        )
        self.camera_to_body = mount @ self.CAMERA_TO_BODY_NADIR
        self.camera_origin_body = np.array((
            camera_offset_forward_m,
            camera_offset_right_m,
            camera_offset_down_m,
        ), dtype=np.float64)

    def camera_ray(self, pixel_x: float, pixel_y: float) -> np.ndarray:
        # 将像素坐标转换到归一化相机坐标并按需单点去畸变。
        if self.distortion is not None:
            point = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
            normalized = cv2.undistortPoints(point, self.matrix, self.distortion)[0, 0]
            ray = np.array((normalized[0], normalized[1], 1.0), dtype=np.float64)
        else:
            ray = self.inverse_matrix @ np.array((pixel_x, pixel_y, 1.0), dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def locate_pixel(self, pixel_x: float, pixel_y: float, pose: AircraftPose) -> TargetLocation:
        # 经相机系、机体系FRD和当地NED求地面交点。
        ray_body = self.camera_to_body @ self.camera_ray(pixel_x, pixel_y)
        body_to_ned = (
            _rz(math.radians(pose.yaw_deg))
            @ _ry(math.radians(pose.pitch_deg))
            @ _rx(math.radians(pose.roll_deg))
        )
        ray_ned = body_to_ned @ ray_body
        camera_origin_ned = body_to_ned @ self.camera_origin_body
        if ray_ned[2] < MIN_GROUND_RAY_DOWN:
            raise ValueError("camera ray is too close to the horizon for reliable localization")
        camera_height_agl = pose.height_agl_m - float(camera_origin_ned[2])
        if camera_height_agl <= 0:
            raise ValueError("camera origin is at or below the target plane")
        slant_range = camera_height_agl / float(ray_ned[2])
        north_m = float(camera_origin_ned[0] + ray_ned[0] * slant_range)
        east_m = float(camera_origin_ned[1] + ray_ned[1] * slant_range)
        latitude, longitude = offset_wgs84(
            pose.latitude_deg, pose.longitude_deg, north_m, east_m
        )
        return TargetLocation(
            latitude, longitude, north_m, east_m,
            math.hypot(north_m, east_m), slant_range,
        )

    def locate_box(self, box, pose: AircraftPose) -> TargetLocation:
        x1, y1, x2, y2 = box
        return self.locate_pixel((x1 + x2) / 2, (y1 + y2) / 2, pose)


class StaticPoseSource:
    """静态图片/地面调试使用的固定飞机位姿，不连接飞控。"""
    def __init__(self, pose: AircraftPose) -> None:
        self.pose = pose

    def pose_for_frame(self, _frame_index: int, _frame_timestamp=None):
        return self.pose

    def close(self) -> None:
        pass

    @staticmethod
    def timing_status() -> dict:
        return {"time_basis": "static", "sync_locked": True}


class TelemetryLog:
    """离线回放飞控 JSONL：按帧号选择当时的飞机位姿。"""

    def __init__(self, path: Path) -> None:
        samples = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            data = json.loads(line)
            if "frame" not in data:
                raise KeyError(f"{path}:{line_number}: missing frame")
            samples.append((int(data["frame"]), AircraftPose.from_mapping(data)))
        if not samples:
            raise ValueError(f"telemetry log is empty: {path}")
        samples.sort(key=lambda item: item[0])
        self.frames = [item[0] for item in samples]
        self.poses = [item[1] for item in samples]

    def pose_for_frame(self, frame_index: int, _frame_timestamp=None):
        index = bisect.bisect_right(self.frames, frame_index) - 1
        return self.poses[index] if index >= 0 else None

    def close(self) -> None:
        pass

    @staticmethod
    def timing_status() -> dict:
        return {"time_basis": "frame-replay", "sync_locked": True}


class MavlinkPoseSource:
    """后台读取飞控 MAVLink，并按曝光时刻对齐 GPS、姿态和离地高度。"""

    def __init__(
        self, endpoint: str, baud: int, max_age: float, height_source: str,
        timesync_rate_hz: float = 2.0, timesync_max_rtt_ms: float = 100.0,
        require_timesync: bool = False,
        max_position_age: float | None = None,
        max_attitude_age: float | None = None,
        max_distance_age: float | None = None,
        target_plane_altitude_above_home_m: float = 0.0,
    ) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError("install pymavlink to use --mavlink") from exc
        self.mavutil = mavutil
        self.connection = mavutil.mavlink_connection(endpoint, baud=baud)
        # 用HEARTBEAT确认飞控连接并取得系统和组件编号。
        heartbeat = self.connection.wait_heartbeat(timeout=10)
        if heartbeat is None:
            raise RuntimeError(f"no MAVLink heartbeat from {endpoint}")
        requested = [(30, 20), (33, 10)]
        if height_source != "relative-alt":
            requested.append((132, 10))
        for message_id, rate_hz in requested:
            # 请求姿态、全球位置和可选测距消息。
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0, message_id, int(1_000_000 / rate_hz), 0, 0, 0, 0, 0,
            )
        self.max_age = max_age
        if self.max_age <= 0:
            raise ValueError("max telemetry age must be positive")
        self.height_source = height_source
        self.target_plane_altitude_above_home_m = float(
            target_plane_altitude_above_home_m
        )
        if not math.isfinite(self.target_plane_altitude_above_home_m):
            raise ValueError("target-plane altitude above Home must be finite")
        self.position_max_age = min(self.max_age, float(self.max_age if max_position_age is None else max_position_age))
        self.attitude_max_age = min(self.max_age, float(self.max_age if max_attitude_age is None else max_attitude_age))
        self.distance_max_age = min(self.max_age, float(self.max_age if max_distance_age is None else max_distance_age))
        if min(self.position_max_age, self.attitude_max_age, self.distance_max_age) <= 0:
            raise ValueError("telemetry stream ages must be positive")
        self.timesync_rate_hz = float(timesync_rate_hz)
        self.timesync_max_rtt_s = float(timesync_max_rtt_ms) / 1000.0
        self.require_timesync = bool(require_timesync)
        if self.timesync_rate_hz < 0 or self.timesync_max_rtt_s <= 0:
            raise ValueError("TIMESYNC rate must be non-negative and maximum RTT positive")
        if self.require_timesync and self.timesync_rate_hz <= 0:
            raise ValueError("--require-timesync needs --timesync-rate-hz greater than zero")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.clock = TelemetryClock()
        self.boot_unwrappers: dict[str, BootTimeUnwrapper] = {}
        self.pending_timesync: dict[int, float] = {}
        self.last_timesync_request_s = -math.inf
        self.last_pose_timing: dict = self.clock.status()
        self.dropped_out_of_order = 0
        self.dropped_invalid = 0
        self.dropped_wrong_source = 0
        self.timesync_send_failures = 0
        self.timesync_send_error: str | None = None
        self.receive_error: str | None = None
        self.positions = deque(maxlen=300)
        self.attitudes = deque(maxlen=600)
        self.distances = deque(maxlen=300)
        self.thread = threading.Thread(target=self._receive, name="mavlink-telemetry", daemon=True)
        self.thread.start()

    def _send_timesync_request(self) -> None:
        # 用TIMESYNC同步两端运行时钟。
        if self.timesync_rate_hz <= 0:
            return
        now_s = time.monotonic()
        if now_s - self.last_timesync_request_s < 1.0 / self.timesync_rate_hz:
            return
        sent_ns = time.monotonic_ns()
        if not self._send_timesync(0, sent_ns):
            self.last_timesync_request_s = sent_ns * 1e-9
            return
        self.pending_timesync[sent_ns] = sent_ns * 1e-9
        self.last_timesync_request_s = sent_ns * 1e-9
        for key, sent_s in tuple(self.pending_timesync.items()):
            if self.last_timesync_request_s - sent_s > 5.0:
                self.pending_timesync.pop(key, None)

    def _send_timesync(self, tc1: int, ts1: int) -> bool:
        try:
            self.connection.mav.timesync_send(int(tc1), int(ts1))
            return True
        except Exception as exc:
            with self.lock:
                self.timesync_send_failures += 1
                self.timesync_send_error = f"{type(exc).__name__}: {exc}"
            return False

    def _from_target(self, message, require_component: bool = False) -> bool:
        """Reject packets from unrelated MAVLink systems/components."""
        connection = getattr(self, "connection", None)
        target_system = int(getattr(connection, "target_system", 0) or 0)
        target_component = int(getattr(connection, "target_component", 0) or 0)
        try:
            source_system = int(message.get_srcSystem())
            source_component = int(message.get_srcComponent())
        except (AttributeError, TypeError, ValueError):
            return target_system == 0
        if target_system and source_system != target_system:
            return False
        return not (
            require_component and target_component
            and source_component != target_component
        )

    def _handle_timesync(self, message, received_ns: int) -> None:
        if not self._from_target(message, require_component=True):
            with self.lock:
                self.dropped_wrong_source = getattr(self, "dropped_wrong_source", 0) + 1
            return
        tc1, ts1 = int(message.tc1), int(message.ts1)
        if tc1 == 0:
            # 回传请求方时间并写入本机单调时钟。
            self._send_timesync(time.monotonic_ns(), ts1)
            return
        if ts1 not in self.pending_timesync:
            return
        self.pending_timesync.pop(ts1, None)
        with self.lock:
            self.clock.observe_timesync(
                ts1, tc1, received_ns, self.timesync_max_rtt_s
            )

    def _boot_time_locked(self, message_type: str, raw_ms: int) -> float | None:
        unwrapper = self.boot_unwrappers.setdefault(message_type, BootTimeUnwrapper())
        boot_s, reset = unwrapper.unwrap(raw_ms)
        if reset:
            # 飞控重启后清空全部旧插值样本。
            self.positions.clear()
            self.attitudes.clear()
            self.distances.clear()
            self.clock.reset()
            self.boot_unwrappers.clear()
            unwrapper = self.boot_unwrappers.setdefault(message_type, BootTimeUnwrapper())
            boot_s, _ = unwrapper.unwrap(raw_ms)
        return boot_s

    def _append_sample_locked(self, history: deque, sample: tuple) -> None:
        if any(
            value is not None and not math.isfinite(float(value))
            for value in sample
        ):
            self.dropped_invalid = getattr(self, "dropped_invalid", 0) + 1
            return
        if (
            sample[0] is not None and history and history[-1][0] is not None
            and sample[0] < history[-1][0]
        ):
            self.dropped_out_of_order += 1
            return
        history.append(sample)

    def _receive(self) -> None:
        """飞控接收线程：持续取包，视觉推理阻塞时也不会停止更新遥测缓存。"""
        wanted = ["GLOBAL_POSITION_INT", "ATTITUDE", "DISTANCE_SENSOR", "TIMESYNC"]
        while not self.stop_event.is_set():
            try:
                self._receive_once(wanted)
            except Exception as exc:
                with self.lock:
                    self.receive_error = f"{type(exc).__name__}: {exc}"
                print(f"MAVLink telemetry stopped: {self.receive_error}", flush=True)
                break

    def _receive_once(self, wanted) -> None:
        """解析一次飞控消息，并分别写入位置、姿态和测距时间序列。"""
        self._send_timesync_request()
        message = self.connection.recv_match(type=wanted, blocking=True, timeout=0.25)
        if message is None:
            return
        received_ns = time.monotonic_ns()
        if not self._from_target(
            message, require_component=message.get_type() == "TIMESYNC"
        ):
            with self.lock:
                self.dropped_wrong_source += 1
            return
        if message.get_type() == "TIMESYNC":
            self._handle_timesync(message, received_ns)
            return
        received_s = received_ns * 1e-9
        with self.lock:
            boot_s = self._boot_time_locked(
                message.get_type(), getattr(message, "time_boot_ms", 0)
            )
            self.clock.observe_boot(boot_s, received_s)
            if message.get_type() == "GLOBAL_POSITION_INT":
                if int(message.lat) == 0 and int(message.lon) == 0:
                    return
                heading = float(message.hdg) / 100.0 if int(message.hdg) != 65535 else None
                latitude = float(message.lat) / 1e7
                longitude = float(message.lon) / 1e7
                if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
                    self.dropped_invalid += 1
                    return
                self._append_sample_locked(self.positions, (
                    boot_s, received_s,
                    latitude, longitude,
                    float(message.relative_alt) / 1000.0, heading,
                ))
            elif message.get_type() == "ATTITUDE":
                self._append_sample_locked(self.attitudes, (
                    boot_s, received_s,
                    math.degrees(message.roll), math.degrees(message.pitch),
                    math.degrees(message.yaw),
                ))
            elif (
                message.get_type() == "DISTANCE_SENSOR"
                and message.current_distance > 0
                and int(message.orientation) == 25  # MAV_SENSOR_ROTATION_PITCH_270
            ):
                self._append_sample_locked(self.distances, (
                    boot_s, received_s, float(message.current_distance) / 100.0
                ))

    @staticmethod
    def _window(samples, timestamp: float, max_age: float, time_index: int = 0):
        if not samples:
            return None
        before = after = None
        # 反向查找最新附近样本以避免复制遥测队列。
        for sample in reversed(samples):
            sample_time = sample[time_index]
            if sample_time is None:
                continue
            if sample_time <= timestamp:
                before = sample
                break
            after = sample
        if before is not None and after is not None:
            before_age = timestamp - before[time_index]
            after_age = after[time_index] - timestamp
            if before_age <= max_age and after_age <= max_age:
                span = after[time_index] - before[time_index]
                ratio = 0.0 if span <= 0 else before_age / span
                return before, after, ratio
        nearest = min(
            (sample for sample in (before, after) if sample is not None),
            key=lambda sample: abs(sample[time_index] - timestamp),
            default=None,
        )
        if nearest is None or abs(nearest[time_index] - timestamp) > max_age:
            return None
        return nearest, nearest, 0.0

    @classmethod
    def _select_window(cls, samples, local_s, boot_s, max_age):
        if boot_s is not None:
            result = cls._window(samples, boot_s, max_age, time_index=0)
            if result is not None:
                return result, "time_boot_ms", boot_s, 0
        result = cls._window(samples, local_s, max_age, time_index=1)
        return result, "receive_monotonic", local_s, 1

    @staticmethod
    def _alignment_ms(window, timestamp, time_index):
        if window is None:
            return None
        return round(min(
            abs(window[0][time_index] - timestamp),
            abs(window[1][time_index] - timestamp),
        ) * 1000.0, 3)

    @staticmethod
    def _stream_health(samples, now_s: float) -> dict:
        if not samples:
            return {"samples": 0, "rate_hz": None, "last_receive_age_ms": None}
        count = min(len(samples), 20)
        first_s = samples[-count][1]
        last_s = samples[-1][1]
        span_s = last_s - first_s
        return {
            "samples": len(samples),
            "rate_hz": (
                None if count < 2 or span_s <= 0
                else round((count - 1) / span_s, 2)
            ),
            "last_receive_age_ms": round(max(0.0, now_s - last_s) * 1000.0, 3),
        }

    @staticmethod
    def _linear(first: float, second: float, ratio: float) -> float:
        return first + (second - first) * ratio

    @staticmethod
    def _angle(first: float, second: float, ratio: float) -> float:
        delta = (second - first + 180.0) % 360.0 - 180.0
        return first + delta * ratio

    def pose_for_frame(self, _frame_index: int, frame_timestamp=None):
        """按相机曝光时刻插值飞控数据，组合出该帧对应的 AircraftPose。"""
        timestamp = float(frame_timestamp) if frame_timestamp is not None else time.monotonic()
        with self.lock:
            frame_boot_s, clock_basis = self.clock.local_to_boot(timestamp)
            position_window, position_basis, position_query, position_index = self._select_window(
                self.positions, timestamp, frame_boot_s,
                getattr(self, "position_max_age", self.max_age),
            )
            attitude_window, attitude_basis, attitude_query, attitude_index = self._select_window(
                self.attitudes, timestamp, frame_boot_s,
                getattr(self, "attitude_max_age", self.max_age),
            )
            distance_window, distance_basis, distance_query, distance_index = self._select_window(
                self.distances, timestamp, frame_boot_s,
                getattr(self, "distance_max_age", self.max_age),
            )
            timing = self.clock.status(timestamp)
            timing["dropped_out_of_order"] = getattr(self, "dropped_out_of_order", 0)
            timing["dropped_invalid"] = getattr(self, "dropped_invalid", 0)
            timing["dropped_wrong_source"] = getattr(self, "dropped_wrong_source", 0)
            timing["timesync_send_failures"] = getattr(self, "timesync_send_failures", 0)
            timing["timesync_send_error"] = getattr(self, "timesync_send_error", None)
            timing["receive_error"] = getattr(self, "receive_error", None)
            health_now_s = time.monotonic()
            timing["stream_health"] = {
                "position": self._stream_health(self.positions, health_now_s),
                "attitude": self._stream_health(self.attitudes, health_now_s),
                "distance": self._stream_health(self.distances, health_now_s),
            }
            timing.update({
                "frame_monotonic_s": round(timestamp, 9),
                "frame_boot_s": None if frame_boot_s is None else round(frame_boot_s, 6),
                "position_basis": position_basis,
                "attitude_basis": attitude_basis,
                "distance_basis": distance_basis,
                "position_alignment_ms": self._alignment_ms(position_window, position_query, position_index),
                "attitude_alignment_ms": self._alignment_ms(attitude_window, attitude_query, attitude_index),
                "distance_alignment_ms": self._alignment_ms(distance_window, distance_query, distance_index),
            })
            if self.require_timesync:
                if clock_basis != "timesync":
                    timing["rejection_reason"] = "TIMESYNC not locked"
                    self.last_pose_timing = timing
                    return None
                unsynchronised_streams = []
                if position_basis != "time_boot_ms":
                    unsynchronised_streams.append("position")
                if attitude_basis != "time_boot_ms":
                    unsynchronised_streams.append("attitude")
                uses_distance = self.height_source == "distance-sensor" or (
                    self.height_source == "auto" and distance_window is not None
                )
                if uses_distance and distance_basis != "time_boot_ms":
                    unsynchronised_streams.append("distance")
                if unsynchronised_streams:
                    timing["rejection_reason"] = (
                        "source timestamps unavailable for "
                        + ", ".join(unsynchronised_streams)
                    )
                    self.last_pose_timing = timing
                    return None
            self.last_pose_timing = timing
        if position_window is None or attitude_window is None:
            return None
        position_a, position_b, position_ratio = position_window
        attitude_a, attitude_b, attitude_ratio = attitude_window
        latitude = self._linear(position_a[2], position_b[2], position_ratio)
        longitude = self._linear(position_a[3], position_b[3], position_ratio)
        relative_altitude = self._linear(position_a[4], position_b[4], position_ratio)
        heading = None
        if position_a[5] is not None and position_b[5] is not None:
            heading = self._angle(position_a[5], position_b[5], position_ratio)
        roll = self._linear(attitude_a[2], attitude_b[2], attitude_ratio)
        pitch = self._linear(attitude_a[3], attitude_b[3], attitude_ratio)
        yaw = self._angle(attitude_a[4], attitude_b[4], attitude_ratio)
        if heading is not None and not math.isfinite(yaw):
            yaw = heading
        distance_valid = distance_window is not None
        if self.height_source == "distance-sensor" and not distance_valid:
            return None
        if self.height_source == "distance-sensor" or (
            self.height_source == "auto" and distance_valid
        ):
            # 将机体向下测距的斜距转换为垂直AGL。
            distance_a, distance_b, distance_ratio = distance_window
            slant_distance = self._linear(distance_a[2], distance_b[2], distance_ratio)
            height = slant_distance * math.cos(math.radians(roll)) * math.cos(math.radians(pitch))
            height_source_used = "distance-sensor"
        else:
            # 从相对Home高度扣除靶面高程得到相对靶面的AGL。
            height = (
                relative_altitude
                - getattr(self, "target_plane_altitude_above_home_m", 0.0)
            )
            height_source_used = "relative-alt"
        if height <= 0:
            return None
        with self.lock:
            self.last_pose_timing["height_source_used"] = height_source_used
            self.last_pose_timing["target_plane_altitude_above_home_m"] = (
                getattr(self, "target_plane_altitude_above_home_m", 0.0)
                if height_source_used == "relative-alt" else None
            )
        try:
            return AircraftPose(latitude, longitude, height, roll, pitch, yaw)
        except ValueError as exc:
            with self.lock:
                self.dropped_invalid = getattr(self, "dropped_invalid", 0) + 1
                self.last_pose_timing["dropped_invalid"] = self.dropped_invalid
                self.last_pose_timing["rejection_reason"] = str(exc)
            return None

    def timing_status(self) -> dict:
        with self.lock:
            return dict(self.last_pose_timing)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.connection.close()


def distance_metres(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    mean_latitude = math.radians((lat_a + lat_b) / 2)
    north = math.radians(lat_b - lat_a) * WGS84_A
    east = math.radians(lon_b - lon_a) * WGS84_A * math.cos(mean_latitude)
    return math.hypot(north, east)


# 按空间距离合并多帧观测并在命中数达标后确认目标。
@dataclass
class _Cluster:
    cluster_id: int
    latitude: float
    longitude: float
    weight: float
    hits: int = 0
    last_frame: int | None = None
    last_seen_unix: float | None = None
    last_seen_monotonic_s: float | None = None
    label_weights: dict[str, float] = field(default_factory=dict)
    track_ids: set[int] = field(default_factory=set)


class LocationClusterer:
    """Confidence-weight repeated GPS observations inside a configurable radius."""

    def __init__(self, radius_m: float = 3.0, minimum_hits: int = 3) -> None:
        if radius_m <= 0 or minimum_hits <= 0:
            raise ValueError("cluster radius and minimum hits must be positive")
        self.radius_m = radius_m
        self.minimum_hits = minimum_hits
        self.clusters: list[_Cluster] = []

    def add(
        self, key: str | None, location: TargetLocation, confidence: float,
        frame_index: int | None = None, timestamp_unix: float | None = None,
        timestamp_monotonic: float | None = None,
        label_confidence: float | None = None, track_id: int | None = None,
    ) -> dict:
        candidates = [
            (distance_metres(item.latitude, item.longitude, location.latitude_deg, location.longitude_deg), item)
            for item in self.clusters
            if frame_index is None or item.last_frame != frame_index
        ]
        tracked_candidates = [
            candidate for candidate in candidates
            if track_id is not None and track_id in candidate[1].track_ids
        ]
        if tracked_candidates:
            candidates = tracked_candidates
        distance, cluster = min(candidates, default=(math.inf, None), key=lambda item: item[0])
        weight = max(float(confidence), 1e-6)
        if cluster is None or distance > self.radius_m:
            cluster = _Cluster(
                len(self.clusters) + 1,
                location.latitude_deg, location.longitude_deg, 0.0,
            )
            self.clusters.append(cluster)
        total = cluster.weight + weight
        cluster.latitude = (cluster.latitude * cluster.weight + location.latitude_deg * weight) / total
        cluster.longitude = (cluster.longitude * cluster.weight + location.longitude_deg * weight) / total
        cluster.weight = total
        if frame_index is None or cluster.last_frame != frame_index:
            cluster.hits += 1
        cluster.last_frame = frame_index
        cluster.last_seen_unix = timestamp_unix
        cluster.last_seen_monotonic_s = timestamp_monotonic
        if track_id is not None:
            cluster.track_ids.add(int(track_id))
        if key:
            label_weight = max(float(label_confidence if label_confidence is not None else confidence), 1e-6)
            cluster.label_weights[key] = cluster.label_weights.get(key, 0.0) + label_weight
        return self._as_dict(cluster)

    def _as_dict(self, cluster: _Cluster) -> dict:
        label = max(cluster.label_weights, key=cluster.label_weights.get) if cluster.label_weights else None
        return {
            "cluster_id": cluster.cluster_id,
            "label": label,
            "latitude_deg": round(cluster.latitude, 9),
            "longitude_deg": round(cluster.longitude, 9),
            "hits": cluster.hits,
            "confirmed": cluster.hits >= self.minimum_hits,
            "last_seen_unix": cluster.last_seen_unix,
            "last_seen_monotonic_s": cluster.last_seen_monotonic_s,
        }

    def results(self, confirmed_only: bool = False) -> list[dict]:
        results = [self._as_dict(cluster) for cluster in self.clusters]
        if confirmed_only:
            results = [item for item in results if item["confirmed"]]
        return sorted(results, key=lambda item: (-item["hits"], item["cluster_id"]))


def add_localization_arguments(parser: argparse.ArgumentParser) -> None:
    """向完整流程入口加入坐标解算、飞控串口、时间同步和聚类参数。"""
    parser.add_argument("--drone-latitude", type=float)
    parser.add_argument("--drone-longitude", type=float)
    parser.add_argument("--height-agl", type=float)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--telemetry-jsonl", type=Path)
    parser.add_argument("--mavlink", help="serial device or UDP endpoint, e.g. /dev/ttyAMA0 or udpin:0.0.0.0:14550")
    parser.add_argument("--mavlink-baud", type=int, default=57600)
    parser.add_argument("--max-telemetry-age", type=float, default=1.0)
    parser.add_argument("--max-position-age", type=float, default=0.25)
    parser.add_argument("--max-attitude-age", type=float, default=0.15)
    parser.add_argument("--max-distance-age", type=float, default=0.25)
    parser.add_argument("--timesync-rate-hz", type=float, default=2.0, help="offline MAVLink TIMESYNC request rate; 0 disables requests")
    parser.add_argument("--timesync-max-rtt-ms", type=float, default=100.0, help="reject slower TIMESYNC round trips")
    parser.add_argument("--require-timesync", action="store_true", help="only localize after a valid MAVLink TIMESYNC lock")
    parser.add_argument("--camera-latency-ms", type=float, default=0.0, help="calibrated residual offset applied to the exposure midpoint")
    parser.add_argument("--height-source", choices=("auto", "relative-alt", "distance-sensor"), default="relative-alt")
    parser.add_argument(
        "--target-plane-altitude-above-home-m", type=float, default=0.0,
        help=("target surface elevation relative to Home; subtracted from "
              "GLOBAL_POSITION_INT.relative_alt"),
    )
    parser.add_argument("--camera-mount-roll", type=float, default=0.0)
    parser.add_argument("--camera-mount-pitch", type=float, default=0.0)
    parser.add_argument("--camera-mount-yaw", type=float, default=0.0)
    parser.add_argument("--camera-offset-forward-m", type=float, default=0.0)
    parser.add_argument("--camera-offset-right-m", type=float, default=0.0)
    parser.add_argument("--camera-offset-down-m", type=float, default=0.0)
    parser.add_argument("--cluster-radius-m", type=float, default=3.0)
    parser.add_argument("--min-location-hits", type=int, default=3)
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--final-json", type=Path, help="write the final clustered target list")
    parser.add_argument("--run-id", help="stable flight/run identifier written to every output record")
    parser.add_argument("--require-telemetry", action="store_true", help="stop instead of recognition-only mode when telemetry is absent")


def build_pose_source(args):
    """按命令行配置选择固定值、日志回放或实时 MAVLink 飞控数据源。"""
    static_values = (args.drone_latitude, args.drone_longitude, args.height_agl)
    enabled = int(args.telemetry_jsonl is not None) + int(args.mavlink is not None) + int(any(value is not None for value in static_values))
    if enabled > 1:
        raise ValueError("choose only one telemetry source: static pose, JSONL, or MAVLink")
    if args.telemetry_jsonl is not None:
        return TelemetryLog(args.telemetry_jsonl.resolve())
    if args.mavlink is not None:
        return MavlinkPoseSource(
            args.mavlink, args.mavlink_baud, args.max_telemetry_age,
            args.height_source, args.timesync_rate_hz,
            args.timesync_max_rtt_ms, args.require_timesync,
            args.max_position_age, args.max_attitude_age, args.max_distance_age,
            args.target_plane_altitude_above_home_m,
        )
    if any(value is not None for value in static_values):
        if any(value is None for value in static_values):
            raise ValueError("--drone-latitude, --drone-longitude and --height-agl are required together")
        return StaticPoseSource(AircraftPose(
            args.drone_latitude, args.drone_longitude, args.height_agl,
            args.roll_deg, args.pitch_deg, args.yaw_deg,
        ))
    return None
