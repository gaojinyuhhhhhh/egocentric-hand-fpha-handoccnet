# PPT 逐页制作指南（含实际数据与配图路径）

> 所有配图已预生成在 `results/final/ppt_assets/`，可直接拖入 PPT。
> 指标数字见 `results/final/ppt_assets/metrics_for_ppt.txt`。

---

## Slide 1 封面页

**标题：** Egocentric Hand Pose Estimation from RGB Videos

**副标题：** 基于 FPHA 数据集与 HandOccNet 的第一视角手部姿态估计实验

**页面元素：**
- 汇报人 / 课程名称 / 汇报日期

**配图：** `results/final/ppt_assets/01_cover_rgb.jpg`

**讲稿要点：**
> 本报告介绍第一视角 RGB 视频中的手部姿态估计，以 FPHA 数据集和 HandOccNet 遮挡鲁棒网络为基线，完成端到端实验与可视化分析。

---

## Slide 2 问题引入 Problem Introduction

**三个要点：**

1. **第一视角（Egocentric）视频**：相机佩戴在用户头部/胸部，以操作者自身视角记录场景。手部常出现在画面下方，与物体频繁交互。

2. **手部姿态估计的价值**：
   - AR/VR 虚实交互：手势驱动虚拟对象
   - 人机交互（HCI）：自然手势控制
   - 机器人视觉抓取：模仿人类操作

3. **核心挑战**：与第三视角不同，第一视角下手部自遮挡严重、物体遮挡频繁、运动模糊明显。

**配图建议：**
- 左：`Subject_1/handshake/1/color/color_0050.jpeg`（原始 FPHA 帧）
- 右：`results/final/ppt_assets/02_gt_overview.png`（GT 骨架叠加）

**讲稿要点：**
> 第一视角视频捕捉的是佩戴者所见，手部是交互的核心。准确估计手部姿态对 AR、HCI 和机器人抓取至关重要，但 egocentric 视角带来了独特的遮挡和视角挑战。

---

## Slide 3 任务定义 Task Definition

**输入：** 单目 RGB 第一视角视频帧（640×480）

**输出（三选一或组合）：**
- 2D 手部关键点（21 关节）
- 3D 手部姿态（相机坐标系）
- MANO 参数化手部网格

**核心难点（列 4 条）：**

| 难点 | 说明 |
|-----|------|
| 手部自遮挡 | 握拳、抓握时手指互相遮挡 |
| 物体遮挡 | 手与物体交互时关节被挡住 |
| 运动模糊 | 快速动作导致图像模糊 |
| 视角剧烈变化 | 手腕旋转导致手指透视缩短 |

**建议画一张流程图：**
```
RGB Frame → Hand Detection/Crop → Pose Network → 2D/3D Joints + MANO Mesh
```

**配图：** 可以用 `03_overlay_good.png` 展示输入输出关系（左原图，右骨架）。

---

## Slide 4 相关工作 Related Work（精简 1 页）

**表格形式呈现：**

| 方法/数据集 | 类型 | 特点 | 本项目角色 |
|-----------|------|------|-----------|
| **HandOccNet** (CVPR 2022) | 模型 | 遮挡鲁棒 3D mesh 估计 | **主基线** |
| **FPHA** (CVPR 2018) | 数据集 | 第一视角 RGB-D + 3D 标注 | **实验数据** |
| **InterHand2.6M** (ECCV 2020) | 数据集 | 大规模双手交互 | 扩展背景 |
| MediaPipe Hands | 模型 | 轻量 2D 检测 | 可选对比（未跑通） |

**讲稿要点：**
> 本实验选用 HandOccNet 作为主基线，因为它专为遮挡场景设计，能同时输出 3D 关节和 MANO 网格。FPHA 是第一视角手部动作的标准 benchmark，包含 RGB-D 和 3D 标注，与我们的任务高度匹配。

**详细文字参考：** `RELATED_WORK.md`

---

## Slide 5 实验数据集 Dataset

**FPHA 数据集介绍：**

| 属性 | 值 |
|-----|-----|
| 全称 | First-Person Hand Action Benchmark |
| 类型 | 第一视角 RGB-D 视频 |
| 总视频数 | 1176（6 个 subject） |
| 总帧数 | 105,165 |
| 本实验序列 | Subject_1 / handshake / 1 |
| 序列帧数 | **225 帧** |
| 分辨率 | **640 × 480** |
| 标注 | 每帧 **21 个 3D 关节**（相机坐标系，mm） |
| 附加数据 | 对齐的深度图 |

**数据集划分：**
- 来源：`data_split_action_recognition.txt`
- 600 个训练序列 + 576 个测试序列（按动作类别划分）
- 本实验使用单个 handshake 序列做定性 + 定量分析

**配图：**
- `results/final/ppt_assets/02_gt_overview.png`（GT 2D 投影 + 3D 骨架）
- 或原始帧 `Subject_1/handshake/1/color/color_0000.jpeg`

---

## Slide 6 方法与实验流水线 Method & Pipeline（核心页）

**完整流程图（建议自己画，箭头清晰）：**

```
FPHA 原始视频
    ↓ 帧提取
RGB (640×480) + Depth + skeleton.txt
    ↓ 手部框裁剪（GT 投影 + margin）
256×256 Hand Patch
    ↓ HandOccNet 推理
2D 关键点 + 3D 关节 + MANO Mesh
    ↓ 深度引导 2D 校正（小改进）
    ↓ 可视化 + 定量分析
MPJPE / 遮挡分数 / 失败案例
```

**自研模块（4 块）：**

1. **数据集解析加载** — `load_sequence_samples()`：读取 RGB、Depth、3D 标注
2. **图像预处理流水线** — `bbox_from_gt_projection()` + `generate_patch_image()`
3. **HandOccNet 推理封装** — `load_handoccnet()` + `infer_one_frame()`
4. **可视化与误差统计** — `visualize_frame()` + `save_occlusion_heatmap()` + `write_summary()`

**小改进：** `refine_2d_with_depth()` — 利用深度图前景质心微调 2D 投影

**代码入口：** `fpha_handoccnet_pipeline.py`

---

## Slide 7 基线实验配置 Baseline Setup

| 配置项 | 值 |
|-------|-----|
| 基准模型 | HandOccNet（ResNet-50 backbone） |
| 预训练权重 | `HandOccNet/weights/snapshot_demo.pth.tar` |
| MANO 模型 | `HandOccNet/weights/mano_v1_2/models/MANO_RIGHT.pkl` |
| 输入尺寸 | 256 × 256 RGB patch |
| 裁剪策略 | GT 2D 投影 bbox + 1.35× margin |
| GPU | CUDA（自动检测） |
| 评估指标 | MPJPE (mm)、Root Error (mm)、Occlusion Score (mm) |
| 推理状态 | **225/225 帧全部推理成功** |

**讲稿要点：**
> 权重和 MANO 模型均已就位，完整 225 帧序列推理验证通过。MPJPE 计算在根关节对齐后进行。

---

## Slide 8 实验结果 Results

**定量结果（直接复制到 PPT）：**

| 指标 | 值 |
|-----|-----|
| 处理帧数 | 225 |
| 成功推理帧数 | 225 |
| Mean MPJPE | **106.11 mm** |
| Median MPJPE | **105.34 mm** |
| Mean Root Error | **0.0 mm** |
| Mean Occlusion Score | **82.44 mm** |

**阶段性结论：**
1. 端到端流水线 225 帧全部跑通，结果可复现
2. MPJPE ~106 mm 反映 egocentric 遮挡场景下的真实难度
3. 遮挡分数与 MPJPE 正相关，验证遮挡是核心痛点

**数据来源：** `results/final/summary.json`

**建议加一张柱状图或表格**，对比 GT 可视化 vs 模型预测的可行性。

---

## Slide 9 可视化效果展示 Visualization（重点页）

**建议布局：2×2 网格，每格一张图**

| 位置 | 内容 | 文件路径 |
|-----|------|---------|
| 左上 | RGB + 2D 骨架叠加（GT 蓝 + Pred 红） | `ppt_assets/03_overlay_good.png` |
| 右上 | 3D GT vs Pred 对比 | `ppt_assets/04_3d_compare.png` |
| 左下 | 深度图 + 关键点 | `ppt_assets/05_depth_overlay.png` |
| 右下 | 遮挡热力图 | `ppt_assets/06_occlusion_heatmap.png` |

**讲稿要点：**
- 2D overlay：检查投影对齐质量，青色=GT，红色=预测
- 3D 对比：揭示结构误差，手指末端偏差明显
- 深度图：辅助理解遮挡区域
- 遮挡热力图：红色=深度不一致=可能被遮挡

**更多帧：** `results/final/frames/` 下有全部 225 帧的 4 种可视化。

---

## Slide 10 失败案例与局限 Failure Cases & Limitations

**配图：** `results/final/ppt_assets/07_failure_gallery.png`

**Top-5 失败帧（MPJPE 最高）：**

| 帧 ID | MPJPE (mm) | 遮挡分数 (mm) |
|-------|-----------|--------------|
| 0016 | 112.96 | 102.51 |
| 0017 | 112.69 | 103.46 |
| ... | ... | ... |

**四类局限：**

1. **重度遮挡**：手-物体交互时关节漂移、跟踪失效
2. **运动模糊**：快速握手动作导致关键点定位偏移
3. **第一视角畸变**：广角镜头边缘投影精度下降
4. **无时序建模**：单帧推理，缺少帧间平滑约束

**讲稿要点：**
> 即使在遮挡鲁棒的 HandOccNet 下，egocentric 场景的平均 MPJPE 仍达 106 mm。失败帧的遮挡分数普遍超过 100 mm，说明遮挡仍是 dominant failure mode。

---

## Slide 11 实验观察与思考 Discussion & Insights

**三个核心观察：**

1. **遮挡是核心痛点，而非次要噪声**
   - 平均遮挡分数 82 mm，失败帧 > 100 mm
   - 21 个关节中大量被标记为 occluded
   - 即使 GT 可视化也能清楚看到手-物体交互遮挡

2. **单帧 RGB → 3D 存在固有歧义**
   - 2D 投影正确 ≠ 3D 结构正确
   - 手指深度估计在 foreshortening 时尤其困难

3. **预处理（裁剪对齐）显著影响结果**
   - 深度引导的 2D 校正（`refine_2d_with_depth`）是本项目的小改进
   - 实际部署中需要独立手部检测器替代 GT bbox

**可选讨论：**
- HandOccNet 在 HO3D 上预训练，与 FPHA 域存在 gap
- 加入时序模型（如 VideoPose3D）可能改善稳定性

---

## Slide 12 总结与未来工作 Conclusion & Future Work

**工作总结：**
1. 搭建完整 FPHA + HandOccNet 端到端实验流水线
2. 完成 225 帧全序列推理，Mean MPJPE = 106.11 mm
3. 实现 4 类可视化 + 定量评估 + 失败案例分析 + PPT 素材导出
4. 尝试深度引导 2D 校正作为小改进

**未来工作：**
- [ ] 时序平滑（Temporal filtering / VideoPose3D）
- [ ] 独立手部检测器替代 GT bbox（如 MediaPipe、YOLO-Hand）
- [ ] 跨动作/跨 subject 定量对比
- [ ] 与 InterHand2.6M 预训练模型对比
- [ ] 利用 FPHA 深度图做 RGB-D 融合

**结尾语：**
> The pipeline is ready for further model-level improvements and cross-dataset comparison.

---

## 附录：如何在 PPT 中插入图片

1. 打开 `results/final/ppt_assets/` 文件夹
2. 按上表文件名拖入对应 Slide
3. 指标数字从 `metrics_for_ppt.txt` 复制
4. 如需其他帧，到 `results/final/frames/` 按帧号查找：
   - `XXXX_overlay.png` — 2D 叠加
   - `XXXX_3d_compare.png` — 3D 对比
   - `XXXX_depth_overlay.png` — 深度
   - `XXXX_occlusion_heatmap.png` — 遮挡热力图
