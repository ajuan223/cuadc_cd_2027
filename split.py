from pathlib import Path
from PIL import Image

# =========================
# 路径
# =========================
INPUT_DIR = Path(r"D:\workplace\cuadc_test\photo\output6")
OUTPUT_DIR = Path(r"D:\workplace\\cuadc_test\photo\split6")

# 4列 × 2行
COLS = 4
ROWS = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        print("output 文件夹中没有找到图片！")
        return

    number = 1

    for image_path in images:
        img = Image.open(image_path).convert("RGB")

        width, height = img.size

        tile_width = width // COLS
        tile_height = height // ROWS

        print(f"处理：{image_path.name}  {width}×{height}")

        for row in range(ROWS):
            for col in range(COLS):

                # 最后一列/最后一行直接延伸到边缘
                x1 = col * tile_width
                y1 = row * tile_height

                x2 = (col + 1) * tile_width if col < COLS - 1 else width
                y2 = (row + 1) * tile_height if row < ROWS - 1 else height

                crop = img.crop((x1, y1, x2, y2))

                # 001.jpg、002.jpg……
                output_name = f"{number:03d}{image_path.suffix.lower()}"

                crop.save(OUTPUT_DIR / output_name, quality=95)

                print(
                    f"  {output_name}: "
                    f"{crop.width}×{crop.height}"
                )

                number += 1

    print()
    print("================================")
    print("切分完成！")
    print(f"原始图片：{len(images)} 张")
    print(f"生成切片：{number - 1} 张")
    print(f"输出目录：{OUTPUT_DIR}")
    print("================================")


if __name__ == "__main__":
    main()