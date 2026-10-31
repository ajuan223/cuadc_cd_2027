"""数字任务完整入口：靶标检测、数字 OCR、飞控读取、坐标解算与结果记录。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import cv2

from app import (
    DigitOCR, IMAGE_SUFFIXES, WeightedVote, build_target_detector, open_capture,
    parser as base_parser, source_value,
)
from detector import padded_crop
from camera_runtime import build_live_camera
from camera_calibration import CameraCalibration
from target_tracking import BoxTrackManager, limit_detections
from digit_geometry import extract_digit_strips
from localization import (
    CameraUndistorter, LocationClusterer, TargetGeolocator,
    add_localization_arguments, build_pose_source,
)


# 逐帧保存识别、飞控位姿和定位，并汇总已确认靶标。
def write_jsonl(handle, payload: dict) -> None:
    if handle is not None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def write_final_json(path: Path | None, clusterer: LocationClusterer, args, pose_source=None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "digit_recognition",
        "schema_version": 2,
        "run_id": args._run_id,
        "generated_at_unix": time.time(),
        "generated_at_monotonic_s": time.monotonic(),
        "mission_duration_s": round(time.monotonic() - args._run_started_monotonic, 3),
        "wall_clock_trusted": False,
        "coordinate_model": "pixel->normalized_camera->body_FRD->local_NED->flat_ground->WGS84",
        "localization_config": {
            "camera_attitude": "rigid-body",
            "camera_mount_deg": [args.camera_mount_roll, args.camera_mount_pitch, args.camera_mount_yaw],
            "camera_offset_body_frd_m": [
                args.camera_offset_forward_m,
                args.camera_offset_right_m,
                args.camera_offset_down_m,
            ],
            "camera_calibration": args._camera_calibration.as_dict(),
            "live_camera_settings": getattr(args, "_camera_settings", None),
            "height_source": args.height_source,
            "target_plane_altitude_above_home_m": args.target_plane_altitude_above_home_m,
            "ocr_confidence_threshold": args.ocr_conf,
            "ocr_mode": args.ocr_mode,
            "ocr_threads": args.ocr_threads,
            "ocr_fast_confidence": max(args.ocr_fast_conf, args.ocr_conf),
            "early_reject_enabled": not args.no_early_reject,
            "camera_latency_ms": (
                args.camera_latency_ms
                if getattr(args, "_camera_latency_applied", False) else 0.0
            ),
            "camera_latency_applied": getattr(args, "_camera_latency_applied", False),
            "timesync_rate_hz": args.timesync_rate_hz,
            "timesync_max_rtt_ms": args.timesync_max_rtt_ms,
            "require_timesync": args.require_timesync,
            "maximum_stream_age_s": {
                "position": args.max_position_age,
                "attitude": args.max_attitude_age,
                "distance": args.max_distance_age,
            },
            "cluster_radius_m": args.cluster_radius_m,
            "minimum_location_hits": args.min_location_hits,
        },
        "confirmed_targets": clusterer.results(confirmed_only=True),
        "all_clusters": clusterer.results(),
        "telemetry_timing": (
            pose_source.timing_status()
            if pose_source is not None and hasattr(pose_source, "timing_status")
            else None
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"final_targets={path.resolve()}")


def process_frame(
    frame, detector, ocr, votes, tracker, geolocator, pose_source, clusterer,
    args, strip_directory, frame_index, log_handle, frame_timestamp=None,
    camera_metadata=None,
):
    """处理一张选中的推理帧，并完成数字识别到 GPS 定位的整条链路。"""
    # 检测靶标并为跨帧目标分配稳定编号。
    observation_monotonic = (
        float(frame_timestamp) if frame_timestamp is not None else time.monotonic()
    )
    observation_unix = observation_monotonic + args._wall_minus_monotonic
    annotated = frame.copy()
    records = []
    detections = limit_detections(detector.infer(frame), args.max_targets)
    track_ids = tracker.assign(detections)
    for stale_id in set(votes) - tracker.active_ids:
        votes.pop(stale_id, None)
    prepared = []
    for track_id, detection in zip(track_ids, detections):
        # 旋正靶标、提取白色数字面板并按所选模式执行OCR。
        crop = padded_crop(frame, detection.box, args.padding)
        geometry_info = {}
        strips = extract_digit_strips(crop, geometry_info, early_reject=not args.no_early_reject)
        number, score, raw = None, 0.0, "no strip"
        ocr_stats = {}
        if strips:
            if strip_directory is not None:
                cv2.imwrite(
                    str(strip_directory / f"frame_{frame_index:06d}_track_{track_id}.jpg"),
                    strips[0],
                )
            if ocr is not None:
                result = ocr.recognize(strips)
                number, score, raw = result["number"], result["score"], result["raw"]
                ocr_stats = result.get("ocr_stats", {})
            else:
                raw = "rectified"
        accepted = number is not None and score >= args.ocr_conf
        voted = votes.setdefault(
            track_id, WeightedVote(args.vote_window)
        ).add(number if accepted else None, score)
        record = {
            "track_id": track_id,
            "box": [round(float(value), 2) for value in detection.box],
            "target_pixel": [
                round(float((detection.box[0] + detection.box[2]) / 2), 2),
                round(float((detection.box[1] + detection.box[3]) / 2), 2),
            ],
            "detection_confidence": round(detection.confidence, 6),
            "number": number,
            "ocr_confidence": round(score, 6),
            "voted_number": voted,
            "accepted": accepted,
            "raw_text": raw,
            "geometry": geometry_info,
            "ocr_stats": ocr_stats,
        }
        prepared.append((track_id, detection, number, score, voted, accepted, record))

    # 读取曝光时刻对应的GPS、高度和姿态。
    pose_error = None
    try:
        pose = (
            pose_source.pose_for_frame(frame_index, observation_monotonic)
            if pose_source is not None else None
        )
    except ValueError as exc:
        pose = None
        pose_error = str(exc)
    telemetry_timing = (
        pose_source.timing_status()
        if pose_source is not None and hasattr(pose_source, "timing_status")
        else {"time_basis": "none", "sync_locked": False}
    )
    telemetry_timing["camera_latency_ms"] = (
        args.camera_latency_ms
        if getattr(args, "_camera_latency_applied", False) else 0.0
    )
    telemetry_timing["camera_latency_applied"] = getattr(
        args, "_camera_latency_applied", False
    )
    if pose_error is not None:
        telemetry_timing["rejection_reason"] = pose_error
    for track_id, detection, number, score, voted, accepted, record in prepared:
        location_text = None
        location_color = (0, 180, 255)
        if pose is not None and accepted:
            try:
                # 将检测框中心从像素坐标解算到WGS84坐标。
                raw_location = geolocator.locate_box(detection.box, pose)
                location_key = voted or number
                # 聚合同类近邻观测并确认稳定目标。
                stable = clusterer.add(
                    location_key, raw_location,
                    detection.confidence * max(score if number else 0.25, 0.25),
                    frame_index=frame_index,
                    timestamp_unix=observation_unix,
                    timestamp_monotonic=observation_monotonic,
                    label_confidence=score if number else None,
                    track_id=track_id,
                )
                record["raw_location"] = raw_location.as_dict()
                record["stable_location"] = stable
                location_color = (0, 210, 0) if stable["confirmed"] else (0, 180, 255)
                location_text = (
                    f"{'GPS' if stable['confirmed'] else 'GPS?'} "
                    f"{stable['latitude_deg']:.7f},"
                    f"{stable['longitude_deg']:.7f} n={stable['hits']}"
                )
            except ValueError as exc:
                record["location_error"] = str(exc)
        elif pose is None:
            record["location_status"] = "telemetry unavailable"
        else:
            record["location_status"] = "candidate rejected"

        x1, y1, x2, y2 = map(int, detection.box)
        color = (0, 210, 0) if accepted else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, f"T{track_id} num={number or '--'} p={score:.2f} vote={voted or '--'}",
            (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, color, 2, cv2.LINE_AA,
        )
        if location_text:
            cv2.putText(
                annotated, location_text,
                (x1, min(annotated.shape[0] - 8, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, location_color, 1, cv2.LINE_AA,
            )
        records.append(record)
        print(
            f"frame={frame_index} track={track_id} target={detection.confidence:.3f} "
            f"number={number} score={score:.3f} vote={voted} "
            f"location={record.get('stable_location')}"
        )
    # 保存本帧完整记录。
    write_jsonl(log_handle, {
        "schema_version": 2,
        "record_type": "frame",
        "run_id": args._run_id,
        "task": "digit_recognition",
        "frame": frame_index,
        "timestamp_unix": observation_unix,
        "wall_clock_trusted": False,
        "capture_monotonic_ns": round(observation_monotonic * 1_000_000_000),
        "mission_elapsed_s": round(observation_monotonic - args._run_started_monotonic, 6),
        "aircraft_pose": pose.as_dict() if pose else None,
        "camera_metadata": camera_metadata,
        "telemetry_timing": telemetry_timing,
        "targets": records,
    })
    return annotated


def run(args) -> int:
    """初始化各模块，并按图片、USB 摄像头或视频文件选择运行循环。"""
    # 初始化视觉模型、飞控数据源、跟踪器和坐标聚类器。
    args._run_id = args.run_id or uuid.uuid4().hex
    args._run_started_monotonic = time.monotonic()
    args._wall_minus_monotonic = time.time() - args._run_started_monotonic
    cv2.setNumThreads(max(1, args.opencv_threads))
    calibration = CameraCalibration.load(args.camera_calibration)
    args._camera_calibration = calibration
    args._camera_settings = None
    detector = build_target_detector(args)
    ocr = None if args.no_ocr else DigitOCR.from_args(args)
    pose_source = build_pose_source(args)
    if pose_source is None and args.require_telemetry:
        raise RuntimeError("telemetry is required but no static, JSONL, or MAVLink source was provided")
    if pose_source is None:
        print("Localization waiting for telemetry: provide static pose, JSONL, or --mavlink")
    clusterer = LocationClusterer(args.cluster_radius_m, args.min_location_hits)
    votes = {}
    tracker = BoxTrackManager(args.track_max_age)
    strip_directory = args.save_strips
    if strip_directory:
        strip_directory.mkdir(parents=True, exist_ok=True)
    source = source_value(args.source)
    args._camera_latency_applied = isinstance(source, int)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES
    undistorter = CameraUndistorter(calibration) if args.undistort else None
    log_handle = None
    if args.log_jsonl:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_jsonl.open("w", encoding="utf-8")

    try:
        if is_image:
            # 处理静态图片输入。
            frame = cv2.imread(source)
            if frame is None:
                raise RuntimeError(f"Cannot read image: {source}")
            if undistorter:
                frame = undistorter.apply(frame)
            geolocator = TargetGeolocator(
                frame.shape[1], frame.shape[0], args.undistort,
                args.camera_mount_roll, args.camera_mount_pitch,
                args.camera_mount_yaw,
                args.camera_offset_forward_m, args.camera_offset_right_m,
                args.camera_offset_down_m, calibration,
            )
            annotated = process_frame(
                frame, detector, ocr, votes, tracker, geolocator, pose_source, clusterer,
                args, strip_directory, 0, log_handle,
            )
            output = args.output or Path("digit_located.jpg")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), annotated):
                raise OSError(f"failed to save {output}")
            print(f"saved={output.resolve()}")
            return 0

        if isinstance(source, int):
            # 持续采集USB相机最新帧并按间隔抽帧推理。
            camera = build_live_camera(
                args, None, exposure_offset_s=args.camera_latency_ms / 1000.0,
            )
            args._camera_settings = camera.settings
            writer, last_seen, last_inferred, geolocator = None, -1, -args.infer_every, None
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"Camera backend={camera.backend} {camera.width}x{camera.height} @ "
                f"{camera.fps:.1f} FPS settings={camera.settings}"
            )
            try:
                while True:
                    item = camera.read_latest(last_seen)
                    if item is None:
                        continue
                    frame_index, captured_at, frame, camera_metadata = item
                    last_seen = frame_index
                    if frame_index - last_inferred < args.infer_every:
                        continue
                    last_inferred = frame_index
                    if undistorter:
                        frame = undistorter.apply(frame)
                    if geolocator is None:
                        geolocator = TargetGeolocator(
                            frame.shape[1], frame.shape[0], args.undistort,
                            args.camera_mount_roll, args.camera_mount_pitch,
                            args.camera_mount_yaw,
                            args.camera_offset_forward_m, args.camera_offset_right_m,
                            args.camera_offset_down_m, calibration,
                        )
                    started = time.perf_counter()
                    shown = process_frame(
                        frame, detector, ocr, votes, tracker, geolocator, pose_source, clusterer,
                        args, strip_directory, frame_index, log_handle, captured_at,
                        camera_metadata,
                    )
                    cv2.putText(
                        shown, f"inference {(time.perf_counter() - started) * 1000:.0f} ms",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (30, 230, 230), 2,
                    )
                    if args.output:
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                max(1.0, camera.fps / args.infer_every),
                                (shown.shape[1], shown.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"cannot create output video: {args.output}")
                        writer.write(shown)
                    if not args.headless:
                        cv2.imshow("digit localization - q to quit", shown)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
            finally:
                camera.close()
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
            return 0

        # 处理已录制视频输入。
        capture = open_capture(source, args.width, args.height, args.fps)
        if isinstance(source, int):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_FPS, args.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        output_fps = capture.get(cv2.CAP_PROP_FPS) or args.fps
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        writer, frame_index = None, 0
        geolocator = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if undistorter:
                    frame = undistorter.apply(frame)
                if geolocator is None:
                    geolocator = TargetGeolocator(
                        frame.shape[1], frame.shape[0], args.undistort,
                        args.camera_mount_roll, args.camera_mount_pitch,
                        args.camera_mount_yaw,
                        args.camera_offset_forward_m, args.camera_offset_right_m,
                        args.camera_offset_down_m, calibration,
                    )
                shown = frame
                if frame_index % args.infer_every == 0:
                    started = time.perf_counter()
                    shown = process_frame(
                        frame, detector, ocr, votes, tracker, geolocator, pose_source, clusterer,
                        args, strip_directory, frame_index, log_handle,
                    )
                    cv2.putText(
                        shown, f"inference {(time.perf_counter() - started) * 1000:.0f} ms",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (30, 230, 230), 2,
                    )
                if args.output:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps, (shown.shape[1], shown.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot create output video: {args.output}")
                    writer.write(shown)
                if not args.headless:
                    cv2.imshow("digit localization - q to quit", shown)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        return 0
    finally:
        # 结束时写出汇总并关闭飞控串口与日志。
        write_final_json(args.final_json, clusterer, args, pose_source)
        if pose_source is not None:
            pose_source.close()
        if log_handle is not None:
            log_handle.close()


def parser():
    result = base_parser()
    result.description = __doc__
    result.add_argument("--opencv-threads", type=int, default=2)
    add_localization_arguments(result)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
