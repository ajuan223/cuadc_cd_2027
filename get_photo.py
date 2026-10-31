import cv2
import os


def extract_frames(video_path, output_dir, interval=1.0, jpg_quality=95):
    """
    从视频中按时间间隔抽帧

    video_path   : 视频文件路径
    output_dir   : 图片输出目录
    interval     : 抽帧间隔，单位：秒
    jpg_quality  : JPG质量，1~100
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频：", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("无法获取视频帧率")
        cap.release()
        return

    duration = frame_count / fps

    print(f"视频帧率：{fps:.2f} FPS")
    print(f"视频总帧数：{frame_count}")
    print(f"视频时长：{duration:.2f} 秒")
    print(f"抽帧间隔：{interval} 秒")

    frame_interval = max(1, int(round(fps * interval)))

    frame_index = 0
    save_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_interval == 0:
            filename = os.path.join(
                output_dir,
                f"frame_{save_index:06d}.jpg"
            )

            cv2.imwrite(
                filename,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
            )

            save_index += 1

        frame_index += 1

    cap.release()

    print(f"\n抽帧完成，共保存 {save_index} 张图片")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":

    video_path = "./video/DJI_20260906165924_0026_D_stabilized.mp4"
    output_dir = "./head1"

    # 每0.5秒抽一张
    extract_frames(
        video_path,
        output_dir,
        interval=0.1,
        jpg_quality=95
    )