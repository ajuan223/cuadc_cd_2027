# CUADC CD 2027

固定翼无人机视觉侦察与目标定位项目。

## 环境要求

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 与依赖，Python 版本固定为 3.11。

首次安装：

```powershell
uv sync
```

`uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 创建 `.venv` 并安装锁定版本的依赖。

## 运行第一步：单图靶标检测

```powershell
uv run python tools/step1_detect_image.py
```

## 运行第二步：裁剪检测到的靶标

```powershell
uv run python tools/step2_crop_target.py
```

## 运行第三步：数字识别

```powershell
uv run python tools/step3_recognize_digit.py
```

更详细的原理和结果见 `STEP1.md`、`STEP2.md` 和 `STEP3.md`。

## 运行第四步：图案识别

```powershell
uv run python tools/step4_recognize_pattern.py
```

## 依赖管理

安装新的运行依赖：

```powershell
uv add package-name
```

安装开发依赖：

```powershell
uv add --dev package-name
```

更新锁文件：

```powershell
uv lock
```

不要在项目中手工维护多份 requirements 文件；运行依赖统一记录在 `pyproject.toml`，精确版本记录在 `uv.lock`。
## 实时视频流输入

生产输入统一由 `get_photo.py` 提供：

- 唯一生产视频源为 IMX708 + Picamera2；
- 帧只保存在内存缓冲区中；
- 处理不过来时覆盖旧帧，不排队、不落盘；
- 不在本地保存抽帧、检测裁剪图或识别面板图；
- `frame_stride` 只控制内存中的采样频率；
- `FramePacket.image` 由帧对象拥有并设为只读，避免下游修改共享图像；
- `configured_fps` 是相机配置值，`source_fps` 是根据采集时间计算的实测值；
- `SensorTimestamp` 从 `CLOCK_BOOTTIME` 转换到单调时钟并做合理性校验，异常时标记回退；
- `close()` 会唤醒并等待采集线程退出，再释放 Picamera2。

`legacy/get_photo_extract_frames.py` 是交接时的旧抽帧脚本，会保存 JPG，不属于生产流程。