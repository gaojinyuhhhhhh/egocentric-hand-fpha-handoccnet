# 第一视角 RGB 视频手部姿态估计（Egocentric Hand Pose Estimation）

本项目围绕 **FPHA 第一视角手部数据集** 与 **HandOccNet 遮挡鲁棒基线模型**，搭建了一套完整的实验流水线：数据加载 → 模型推理 → 可视化 → 定量评估 → 失败案例分析 → PPT 素材导出。

---

## 一、项目目标（对应老师要求）

| 老师要求 | 本项目交付 |
|---------|-----------|
| 相关工作调研 | `RELATED_WORK.md` |
| 基线方法与实验结果 | HandOccN et 在 FPHA 上 225 帧全序列推理 |
| 可视化输出 | 2D 骨架叠加、3D 对比、深度图、遮挡热力图、MANO 网格 |
| 失败案例与局限分析 | `failure_cases.json` + `failure_cases_gallery.png` |
| 小改进尝试 | 深度引导的 2D 投影校正（`refine_2d_with_depth`） |

---

## 二、快速开始

### 环境要求

- Python 3.8+
- PyTorch（支持 CUDA 更佳）
- OpenCV、NumPy、Matplotlib、tqdm

### 1. 冒烟测试（不跑模型，仅验证 GT 可视化）

```bash
cd /root/data
python fpha_handoccnet_pipeline.py --skip_inference --max_frames 2 --output_dir results/fpha_smoke
```

> **注意**：`results/fpha_smoke/` 只有 2 帧、无 MPJPE 指标，**不能作为最终汇报结果**。完整结果请看 `results/final/`。

### 2. 完整实验（HandOccNet 推理 + 全部可视化）

```bash
cd /root/data
python fpha_handoccnet_pipeline.py --output_dir results/final
```

默认处理 **整个序列的全部帧**（handshake/1 共 225 帧）。

### 3. 对已有结果补生成热力图 / PPT 素材

```bash
python postprocess_results.py --results_dir results/final
```

### 4. 生成演示视频

```bash
python generate_demo_video.py --results_dir results/final --fps 12
```

输出：
- `demo_overlay.mp4` — 2D GT/Pred 骨架叠加（约 19 秒，适合 PPT 嵌入）
- `demo_full.mp4` — 2×2 四宫格（overlay / 3D / 深度 / 遮挡热力图）

视频会同步复制到 `results/final/ppt_assets/`。

---

## 三、目录结构说明

```
/root/data/
│
├── README.md                          ← 本文件：项目总说明
├── RELATED_WORK.md                    ← 相关工作精简综述（PPT Slide 4 素材）
├── PPT_GUIDE.md                       ← 逐页 PPT 制作指南
│
├── fpha_handoccnet_pipeline.py        ← 【核心】主实验流水线脚本
├── postprocess_results.py             ← 后处理：遮挡热力图、失败案例图、PPT 素材
├── generate_demo_video.py             ← 从可视化帧生成演示 MP4 视频
│
├── HandOccNet/                        ← HandOccNet 官方代码（已打补丁）
│   ├── weights/
│   │   ├── snapshot_demo.pth.tar    ← 预训练权重（必需）
│   │   └── mano_v1_2/models/
│   │       └── MANO_RIGHT.pkl       ← MANO 手部模型（必需）
│   └── ...
│
├── Subject_1/                         ← FPHA 原始 RGB + 深度图
│   └── handshake/1/
│       ├── color/color_XXXX.jpeg      ← RGB 帧（640×480）
│       └── depth/depth_XXXX.png       ← 深度帧
│
├── Hand_pose_annotation_v1/           ← FPHA 3D 手部关节标注
│   └── Subject_1/handshake/1/
│       └── skeleton.txt               ← 每帧 21 个 3D 关节坐标
│
├── Subject_1_info.txt                 ← 数据集统计（动作数、帧数）
├── data_split_action_recognition.txt  ← 训练/测试划分
│
└── results/
    ├── final/                         ← 【最终汇报用】完整 225 帧实验结果
    ├── fpha_with_weights/             ← 同上（final 是指向它的符号链接）
    ├── fpha_smoke/                    ← 2 帧冒烟测试（不完整，勿用于汇报）
    ├── fpha_with_weights_fix/         ← 早期 1 帧调试用
    └── fpha_with_weights_refine/      ← 早期 overlay 校正验证用
```

---

## 四、实验流水线详解

```
FPHA 原始视频帧
    ↓
① 数据解析（load_sequence_samples）
    读取 RGB、Depth、skeleton.txt 中的 21 个 3D 关节
    ↓
② 预处理（bbox_from_gt_projection + generate_patch_image）
    根据 GT 投影生成手部裁剪框 → 256×256 输入 patch
    ↓
③ HandOccNet 推理（infer_one_frame）
    输出：2D 关键点、3D 关节、MANO 网格顶点
    ↓
④ 小改进：深度引导 2D 校正（refine_2d_with_depth）
    利用深度图前景质心微调 2D 投影，缓解裁剪偏移
    ↓
⑤ 可视化（visualize_frame + save_occlusion_heatmap）
    - *_overlay.png        RGB + GT/Pred 2D 骨架
    - *_3d_compare.png      3D GT vs Pred
    - *_depth_overlay.png   深度图 + 关键点
    - *_occlusion_heatmap.png  遮挡热力图
    - *_mesh.obj            MANO 网格
    ↓
⑥ 定量评估
    MPJPE、Root Error、遮挡分数 → results.json / summary.json
    ↓
⑦ 失败案例分析
    Top-10 最差帧 → failure_cases.json
    Top-5 可视化拼图 → failure_cases_gallery.png
    ↓
⑧ PPT 素材导出
    ppt_assets/ 文件夹，可直接插入幻灯片
```

---

## 五、最终实验结果（`results/final/`）

### 实验配置

| 项目 | 值 |
|-----|-----|
| 数据集 | FPHA Subject_1 / handshake / 1 |
| 帧数 | 225 |
| 分辨率 | 640 × 480 |
| 基线模型 | HandOccNet（CVPR 2022 预训练权重） |
| 评估指标 | MPJPE（mm）、Root Error（mm）、遮挡分数（mm） |

### 定量结果

| 指标 | 值 |
|-----|-----|
| Mean MPJPE | **106.11 mm** |
| Median MPJPE | **105.34 mm** |
| Mean Root Error | **0.0 mm**（已做根关节对齐） |
| Mean Occlusion Score | **82.44 mm** |

### 输出文件一览

| 文件/文件夹 | 内容 |
|------------|------|
| `summary.json` | 汇总指标 |
| `results.json` | 每帧 MPJPE、遮挡分数 |
| `failure_cases.json` | 误差最大的 10 帧 |
| `failure_cases_gallery.png` | 失败案例可视化拼图 |
| `gt_overview.png` | GT 2D/3D 总览 |
| `analysis_report.md` | 文字版分析报告 |
| `frames/` | 每帧 4 种可视化 PNG（共 900 张） |
| `XXXX_mesh.obj` | 每帧 MANO 网格（225 个） |
| `ppt_assets/` | **PPT 直接可用的配图 + 指标文本 + 演示视频** |
| `demo_overlay.mp4` | 2D 骨架叠加演示视频（225 帧，12 fps） |
| `demo_full.mp4` | 四宫格多视角演示视频 |

---

## 六、PPT 配图速查

做 PPT 时，直接从 `results/final/ppt_assets/` 取图：

| PPT 用途 | 文件 |
|---------|------|
| 封面背景 | `01_cover_rgb.jpg` |
| 数据集/GT 介绍 | `02_gt_overview.png` |
| 2D 骨架叠加 | `03_overlay_good.png` |
| 3D GT vs Pred | `04_3d_compare.png` |
| 深度分析 | `05_depth_overlay.png` |
| 遮挡热力图 | `06_occlusion_heatmap.png` |
| 失败案例 | `07_failure_gallery.png` |
| 指标数字 | `metrics_for_ppt.txt` |

更详细的逐页说明见 **`PPT_GUIDE.md`**。

---

## 七、已知局限与未来工作

1. **遮挡是主要误差来源**：平均遮挡分数 82 mm，与 MPJPE 106 mm 高度相关。
2. **单帧推理无时序约束**：相邻帧预测可能抖动，可加时序平滑。
3. **裁剪框依赖 GT 投影**：实际部署需独立手部检测器。
4. **HandOccNet 在 HO3D 上预训练**：与 FPHA 第一视角域存在 gap。
5. **MediaPipe 基线**：环境兼容性问题暂未跑通，可作为未来对比。

---

## 八、引用

**FPHA 数据集：**
```bibtex
@inproceedings{FirstPersonAction_CVPR2018,
  title={First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations},
  author={Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun},
  booktitle={CVPR}, year={2018}
}
```

**HandOccNet：**
```bibtex
@inproceedings{handoccnet2022,
  title={HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network},
  booktitle={CVPR}, year={2022}
}
```
