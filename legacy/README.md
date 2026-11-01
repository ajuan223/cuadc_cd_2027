# Legacy scripts

此目录保存交接时收到的旧脚本，仅供历史参考，不属于生产代码。

`get_photo_extract_frames.py` 会从视频中抽取 JPG 到磁盘，与当前生产约束冲突。新的实时输入实现位于仓库根目录的 `get_photo.py`，不会保存任何图片。