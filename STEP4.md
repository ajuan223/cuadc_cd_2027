# V1 第四步：识别图案靶

图案识别和数字识别使用两个独立模型，但它们共用同一个 `target_detector`。

```text
靶标检测框
→ 向外扩展并裁剪靶标
→ 提取中央白色图案面板
→ pattern_model 分类
→ 输出图案类别和置信度
```

## 图案类别

| 模型输出 | 含义 |
|---|---|
| `jiqiangbing` | 机枪兵 |
| `huojianbing` | 火箭兵 |
| `duoxuanyi` | 多旋翼 |
| `gudingyi` | 固定翼 |
| `kache` | 卡车 |
| `fangkongpao` | 防空炮 |
| `tanke` | 坦克 |
| `zhishengji` | 直升机 |
| `zhandouji` | 战斗机 |
| `yunshuji` | 运输机 |
| `zhenchaji` | 侦察机 |
| `hongzhaji` | 轰炸机 |
| `invalid` | 无效或无法识别的图案 |

## 运行

```powershell
uv run python tools\step4_recognize_pattern.py
```

默认使用实际飞采帧：

```text
testdata/pattern_yunshuji_frame.jpg
```

指定其他图片：

```powershell
uv run python tools\step4_recognize_pattern.py --image path\to\image.jpg
```

## 输出

提取出的中央图案面板保存在：

```text
artifacts/step4_pattern/
```

默认运行应当输出：

```text
class=yunshuji
score≈0.6304
accepted=True
```

## 已知问题

- 靶标检测模型仍可能把车辆或其他物体检测为 `target`。
- 图案几何可能把草地等背景区域误提取为白色面板。
- 图案分类模型可能给错误面板较高置信度。

因此后续必须继续训练 `target` 检测模型和图案分类模型，不能把当前单帧结果当作最终精度。