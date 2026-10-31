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