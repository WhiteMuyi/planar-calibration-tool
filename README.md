# Planar Calibration Tool

A desktop application for computing a **planar homography** between image/pixel coordinates and real-world ground-plane coordinates. Load an image or video frame, place control points, enter their known world positions, and export a calibration JSON that can be used downstream to map any pixel on the ground plane to a real-world (X, Y) coordinate.

一款用于计算图像像素坐标与真实世界地面平面坐标之间**平面单应矩阵（Homography）**的桌面标定工具。加载图片或视频帧，在画面上放置控制点并输入对应的真实世界坐标，即可导出可供下游使用的标定 JSON 文件，将地面上任意像素映射到真实世界的 (X, Y) 坐标。

---

## What it does / 功能概述

| Feature / 功能 | Description / 说明 |
|---|---|
| Image & video input | Load `.jpg/.png/.bmp/.tiff` images or `.mp4/.mov/.avi` videos — scrub frame-by-frame to pick the best calibration frame |
| 图片与视频输入 | 支持常见图片格式及视频格式，可逐帧浏览选取最佳标定帧 |
| Interactive point placement | Click anywhere on the canvas to add a control point; drag to reposition it | 
| 交互式控制点放置 | 在画布上点击即可添加控制点，拖拽可重新定位 |
| World coordinate entry | Enter each point's real-world (X, Y) directly in the table; supports meters, centimeters, millimeters, feet, or custom units |
| 真实世界坐标输入 | 在表格中直接输入每个控制点对应的真实坐标；支持米、厘米、毫米、英尺或自定义单位 |
| Homography computation | Computes both image→world and world→image 3×3 homography matrices via OpenCV |
| 单应矩阵计算 | 通过 OpenCV 计算图像→世界和世界→图像两个方向的 3×3 单应矩阵 |
| Fit metrics | Reports mean / median / RMS / max reprojection error (px), world-space error, and image coverage fraction |
| 拟合误差报告 | 输出均值/中位数/RMS/最大重投影误差（像素）、世界坐标误差及图像覆盖率 |
| Calibration overlay | Projects a configurable world-coordinate grid and axis indicators back onto the image for visual verification |
| 标定叠加层可视化 | 将可配置的世界坐标网格和坐标轴投影回图像，方便直观验证标定质量 |
| Condition warnings | Detects collinear points, low image coverage, and duplicate coordinates before computing |
| 条件预警 | 在计算前自动检测共线点、覆盖率不足和重复坐标等问题 |
| Project save / load | Save and reload the full working state (source path, points, calibration) as a `.project.json` |
| 项目保存与加载 | 将完整工作状态（源文件路径、控制点、标定结果）保存为 `.project.json` 并可重新加载 |
| Export calibration JSON | Export a compact, API-ready `.calibration.json` containing the homography matrices and metadata |
| 导出标定 JSON | 导出包含单应矩阵和元数据的精简 API-ready `.calibration.json` 文件 |
| Video rotation | Auto-detects video orientation; manual override with automatic control-point coordinate remapping |
| 视频旋转处理 | 自动检测视频方向；支持手动调整，控制点坐标自动随旋转重新映射 |

---

## Typical workflow / 典型使用流程

**English**

1. Open an image or video (`Open Image` / `Open Video`).
2. For video: scrub to the frame that best shows the ground plane, then click `Use Current Frame`.
3. Set the real-world unit (meters, cm, etc.) in the **Units** panel.
4. Click on known ground-plane locations in the canvas — a marker is added for each click.
5. In the **control point table**, enter the real-world `World X` and `World Y` for each marker (at least 4 non-collinear points required).
6. Click **Compute Calibration**. Inspect the reprojection errors and the grid overlay.
7. Click **Export Calibration JSON** to save the result for use in other pipelines.

**中文**

1. 打开图片或视频（`Open Image` / `Open Video`）。
2. 若为视频：拖动进度条找到最能清晰展示地面的帧，点击 `Use Current Frame`。
3. 在 **Units** 面板中选择真实世界单位（米、厘米等）。
4. 在画布上点击已知地面位置——每次点击会添加一个控制点标记。
5. 在**控制点表格**中填写每个标记对应的 `World X` 和 `World Y`（至少需要 4 个非共线点）。
6. 点击 **Compute Calibration**，检查重投影误差和网格叠加层。
7. 点击 **Export Calibration JSON** 将结果导出供其他流程使用。

---

## Canvas controls / 画布操作

| Action / 操作 | Input / 输入方式 |
|---|---|
| Pan / 平移 | Middle-click drag · Right-click drag · Ctrl+Left-click drag · **Trackpad two-finger swipe** |
| Zoom / 缩放 | Mouse scroll wheel · **Trackpad pinch gesture** |
| Add point / 添加控制点 | Left-click on canvas |
| Move point / 移动控制点 | Left-click and drag an existing marker |
| Select point / 选中控制点 | Left-click on a marker |

---

## Output format / 输出格式

The exported `.calibration.json` contains:

导出的 `.calibration.json` 包含以下字段：

```json
{
  "model_type": "planar_homography",
  "reference_width": 1920,
  "reference_height": 1080,
  "homography_image_to_world": [[...], [...], [...]],
  "homography_world_to_image": [[...], [...], [...]],
  "calibration_hull_image": [[x, y], ...],
  "fit_metrics_summary": {
    "num_points_used": 6,
    "rms_reprojection_error_px": 0.42,
    "max_reprojection_error_px": 0.81
  },
  "units": { "unit_name": "meters", "unit_symbol": "m" },
  "usage_notes": [...]
}
```

> **Important / 注意：** This calibration assumes all queried points lie on the **ground plane (Z = 0)**. Apply it only to images with the exact same resolution as the reference frame. For foot-position estimation, query the lowest visible foot–ground contact pixel, not the bounding-box center.
>
> 本标定假设所有查询点位于**地面平面（Z = 0）**。仅适用于与参考帧分辨率完全相同的图像。用于足部位置估计时，应查询脚与地面接触的最低像素点，而非边界框中心。

---

## Installation / 安装

```bash
# Clone the repository / 克隆仓库
git clone git@github.com:WhiteMuyi/planar-calibration-tool.git
cd planar-calibration-tool

# Create and activate a virtual environment / 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Run / 启动
python main_window.py
```

**Requirements / 依赖环境**

- Python 3.10+
- PySide6 ≥ 6.7
- OpenCV-Python ≥ 4.10
- NumPy ≥ 2.0
- SciPy ≥ 1.13

---

## Project structure / 项目结构

```
planar-calibration-tool/
├── main_window.py        # UI layer — PySide6 main window and all widgets
│                         # UI 层 —— PySide6 主窗口及所有控件
├── video_canvas.py       # Interactive QGraphicsView canvas with zoom/pan
│                         # 交互式画布，支持缩放与平移
├── calibration_core.py   # Pure-Python calibration engine (no UI dependency)
│                         # 纯 Python 标定计算引擎（不依赖 UI）
└── requirements.txt
```
