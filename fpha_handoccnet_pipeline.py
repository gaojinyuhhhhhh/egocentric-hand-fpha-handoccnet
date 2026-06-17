import argparse
import json
import os
import os.path as osp
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


ROOT = osp.dirname(osp.abspath(__file__))
HANDOCCNET_ROOT = osp.join(ROOT, 'HandOccNet')

if HANDOCCNET_ROOT not in sys.path:
    sys.path.insert(0, osp.join(HANDOCCNET_ROOT, 'main'))
    sys.path.insert(0, osp.join(HANDOCCNET_ROOT, 'common'))

from config import cfg  # type: ignore
from model import get_model  # type: ignore
from utils.preprocessing import generate_patch_image, process_bbox  # type: ignore
from utils.mano import MANO  # type: ignore


FX = 475.065948
FY = 475.065857
CX = 315.944855
CY = 245.287079

HANDSKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_obj(vertices, faces, file_name='output.obj'):
    with open(file_name, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))
        for face in faces:
            obj_file.write('f {}/{} {}/{} {}/{}\n'.format(
                face[0] + 1, face[0] + 1,
                face[1] + 1, face[1] + 1,
                face[2] + 1, face[2] + 1,
            ))


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    return depth


def parse_skeleton_line(line):
    vals = line.strip().split()
    frame_id = vals[0]
    pts = np.array(list(map(float, vals[1:])), dtype=np.float32).reshape(21, 3)
    return frame_id, pts


def load_sequence_samples(subject='Subject_1', action='handshake', instance='1', max_frames=None):
    seq_dir = osp.join(ROOT, subject, action, instance)
    anno_path = osp.join(ROOT, 'Hand_pose_annotation_v1', subject, action, instance, 'skeleton.txt')
    color_dir = osp.join(seq_dir, 'color')
    depth_dir = osp.join(seq_dir, 'depth')

    if not osp.exists(anno_path):
        raise FileNotFoundError(anno_path)

    with open(anno_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if max_frames is not None:
        lines = lines[:max_frames]

    samples = []
    for line in lines:
        frame_id, pts3d = parse_skeleton_line(line)
        samples.append({
            'frame_id': frame_id,
            'rgb_path': osp.join(color_dir, 'color_{}.jpeg'.format(frame_id)),
            'depth_path': osp.join(depth_dir, 'depth_{}.png'.format(frame_id)),
            'gt_3d': pts3d,
        })
    return samples


def project_points(pts3d):
    z = np.clip(pts3d[:, 2], 1e-6, None)
    u = FX * pts3d[:, 0] / z + CX
    v = FY * pts3d[:, 1] / z + CY
    return np.stack([u, v], axis=1)


def bbox_from_gt_projection(gt_2d, img_w, img_h, margin=1.35):
    xmin = float(np.min(gt_2d[:, 0]))
    ymin = float(np.min(gt_2d[:, 1]))
    xmax = float(np.max(gt_2d[:, 0]))
    ymax = float(np.max(gt_2d[:, 1]))
    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.float32)
    bbox = process_bbox(bbox, img_w, img_h, expansion_factor=margin)
    if bbox is None:
        bbox = np.array([0, 0, img_w - 1, img_h - 1], dtype=np.float32)
    return bbox


def depth_occlusion_score(depth_map, gt_2d, gt_3d, search_radius=4, tolerance_mm=20.0):
    if depth_map is None:
        return np.nan, []
    scores = []
    occluded = []
    h, w = depth_map.shape[:2]
    for (u, v), p3d in zip(gt_2d, gt_3d):
        x = int(round(u))
        y = int(round(v))
        x0 = max(0, x - search_radius)
        x1 = min(w, x + search_radius + 1)
        y0 = max(0, y - search_radius)
        y1 = min(h, y + search_radius + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            scores.append(np.nan)
            occluded.append(True)
            continue
        depth_val = float(np.median(valid))
        diff = abs(depth_val - float(p3d[2]))
        scores.append(diff)
        occluded.append(diff > tolerance_mm)
    return float(np.nanmean(scores)), occluded


def draw_skeleton_3d(ax, pts, color='tab:orange', linewidth=2, label=None):
    for a, b in HANDSKELETON:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], [pts[a, 2], pts[b, 2]], color=color, linewidth=linewidth)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=18, label=label)


def draw_skeleton_2d(ax, pts, color='tab:orange', linewidth=2, label=None):
    for a, b in HANDSKELETON:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color, linewidth=linewidth)
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=18, label=label)


def mpjpe(pred, gt):
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def root_align(pred, gt):
    pred = pred - pred[0]
    gt = gt - gt[0]
    return pred, gt


def refine_2d_with_depth(pred_2d, depth_map, image_shape, pad=40, percentile=10, depth_margin=15):
    if depth_map is None or pred_2d is None:
        return pred_2d

    h, w = image_shape[:2]
    xmin = max(0, int(np.floor(np.min(pred_2d[:, 0]) - pad)))
    ymin = max(0, int(np.floor(np.min(pred_2d[:, 1]) - pad)))
    xmax = min(w - 1, int(np.ceil(np.max(pred_2d[:, 0]) + pad)))
    ymax = min(h - 1, int(np.ceil(np.max(pred_2d[:, 1]) + pad)))
    if xmax <= xmin or ymax <= ymin:
        return pred_2d

    roi = depth_map[ymin:ymax + 1, xmin:xmax + 1]
    valid = roi[roi > 0]
    if valid.size < 20:
        return pred_2d

    threshold = np.percentile(valid, percentile) + depth_margin
    foreground = (roi > 0) & (roi <= threshold)
    if foreground.sum() < 20:
        foreground = roi > 0

    ys, xs = np.where(foreground)
    centroid = np.array([xmin + xs.mean(), ymin + ys.mean()], dtype=np.float32)
    delta = centroid - pred_2d[0]
    refined = pred_2d + delta[None, :]
    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)
    return refined


def infer_one_frame(model, device, rgb, bbox):
    transform = transforms.ToTensor()
    patch, _, bb2img_trans = generate_patch_image(rgb, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    input_tensor = transform(patch.astype(np.float32)) / 255.0
    input_tensor = input_tensor.to(device)[None, ...]
    with torch.no_grad():
        out = model({'img': input_tensor}, {}, {}, 'test')
    pred_3d = out['joints_coord_cam'][0].detach().cpu().numpy()
    pred_mesh = out['mesh_coord_cam'][0].detach().cpu().numpy()
    pred_2d_crop = out['joints_coord_img'][0].detach().cpu().numpy()
    pred_2d_crop = pred_2d_crop * np.array([cfg.input_img_shape[1], cfg.input_img_shape[0]], dtype=np.float32)
    pred_2d_xy1 = np.concatenate([pred_2d_crop, np.ones((pred_2d_crop.shape[0], 1), dtype=np.float32)], axis=1)
    pred_2d = (bb2img_trans @ pred_2d_xy1.T).T[:, :2]
    return pred_3d, pred_mesh, pred_2d


def load_handoccnet(args):
    if not osp.exists(args.checkpoint):
        return None, None

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu != 'cpu' else 'cpu')
    try:
        model = get_model('test')
        model = torch.nn.DataParallel(model).to(device)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state_dict = ckpt['network'] if isinstance(ckpt, dict) and 'network' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print('Loaded checkpoint:', args.checkpoint)
        print('Missing keys:', len(missing), 'Unexpected keys:', len(unexpected))
        model.eval()
        return model, device
    except FileNotFoundError as exc:
        print('HandOccNet model is present, but MANO assets are missing.')
        print('Inference is disabled until MANO_RIGHT.pkl is placed under HandOccNet/common/utils/manopth/mano/models')
        print('Details:', exc)
        return None, None


@dataclass
class FrameResult:
    frame_id: str
    gt_3d: np.ndarray
    gt_2d: np.ndarray
    pred_3d: Optional[np.ndarray] = None
    pred_2d: Optional[np.ndarray] = None
    mpjpe_mm: Optional[float] = None
    root_error_mm: Optional[float] = None
    occlusion_score_mm: Optional[float] = None
    occluded_joints: Optional[List[bool]] = None


def save_occlusion_heatmap(output_dir, rgb, depth_map, gt_2d, gt_3d, occluded_joints, frame_id):
    if depth_map is None:
        return

    h, w = depth_map.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (u, v), p3d in zip(gt_2d, gt_3d):
        x = int(round(u))
        y = int(round(v))
        x0 = max(0, x - 8)
        x1 = min(w, x + 9)
        y0 = max(0, y - 8)
        y1 = min(h, y + 9)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            score = 120.0
        else:
            score = abs(float(np.median(valid)) - float(p3d[2]))
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], score)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title('RGB')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(heat, cmap='hot', vmin=0, vmax=120)
    ax2.set_title('Occlusion heatmap (depth mismatch mm)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(rgb)
    joint_colors = ['red' if occ else 'lime' for occ in occluded_joints]
    ax3.scatter(gt_2d[:, 0], gt_2d[:, 1], c=joint_colors, s=28, edgecolors='white', linewidths=0.5)
    ax3.set_title('Joint occlusion (red=occluded)')
    ax3.axis('off')
    plt.tight_layout()
    fig.savefig(osp.join(output_dir, '{}_occlusion_heatmap.png'.format(frame_id)), dpi=180)
    plt.close(fig)


def export_failure_gallery(results, samples_by_id, output_dir, top_k=5):
    numeric = [r for r in results if r.mpjpe_mm is not None]
    if not numeric:
        return
    worst = sorted(numeric, key=lambda x: x.mpjpe_mm, reverse=True)[:top_k]
    fig = plt.figure(figsize=(15, 4 * top_k))
    for row, result in enumerate(worst):
        sample = samples_by_id[result.frame_id]
        rgb = load_rgb(sample['rgb_path'])

        ax_rgb = fig.add_subplot(top_k, 3, row * 3 + 1)
        ax_rgb.imshow(rgb)
        draw_skeleton_2d(ax_rgb, result.gt_2d, color='cyan', linewidth=2)
        if result.pred_2d is not None:
            draw_skeleton_2d(ax_rgb, result.pred_2d, color='red', linewidth=2)
        ax_rgb.set_title('Frame {} | MPJPE {:.1f} mm'.format(result.frame_id, result.mpjpe_mm))
        ax_rgb.axis('off')

        ax_3d = fig.add_subplot(top_k, 3, row * 3 + 2, projection='3d')
        draw_skeleton_3d(ax_3d, result.gt_3d, color='cyan')
        if result.pred_3d is not None:
            draw_skeleton_3d(ax_3d, result.pred_3d, color='red')
        ax_3d.set_title('3D GT vs Pred')

        ax_info = fig.add_subplot(top_k, 3, row * 3 + 3)
        info = (
            'Frame: {}\n'
            'MPJPE: {:.2f} mm\n'
            'Root error: {:.2f} mm\n'
            'Occlusion score: {:.2f} mm\n'
            'Occluded joints: {}/21'
        ).format(
            result.frame_id,
            result.mpjpe_mm,
            result.root_error_mm or 0.0,
            result.occlusion_score_mm or 0.0,
            sum(result.occluded_joints) if result.occluded_joints else 0,
        )
        ax_info.text(0.05, 0.5, info, fontsize=12, va='center', family='monospace')
        ax_info.axis('off')
    plt.tight_layout()
    fig.savefig(osp.join(output_dir, 'failure_cases_gallery.png'), dpi=180)
    plt.close(fig)


def export_presentation_assets(output_dir, summary):
    ppt_dir = osp.join(output_dir, 'ppt_assets')
    ensure_dir(ppt_dir)
    frames_dir = osp.join(output_dir, 'frames')

    candidates = [
        ('01_cover_rgb.jpg', osp.join(ROOT, 'Subject_1', 'handshake', '1', 'color', 'color_0050.jpeg')),
        ('02_gt_overview.png', osp.join(output_dir, 'gt_overview.png')),
        ('03_overlay_good.png', osp.join(frames_dir, '0050_overlay.png')),
        ('04_3d_compare.png', osp.join(frames_dir, '0050_3d_compare.png')),
        ('05_depth_overlay.png', osp.join(frames_dir, '0050_depth_overlay.png')),
        ('06_occlusion_heatmap.png', osp.join(frames_dir, '0050_occlusion_heatmap.png')),
        ('07_failure_gallery.png', osp.join(output_dir, 'failure_cases_gallery.png')),
    ]
    for dst_name, src_path in candidates:
        if not osp.exists(src_path):
            continue
        shutil.copy2(src_path, osp.join(ppt_dir, dst_name))

    with open(osp.join(ppt_dir, 'metrics_for_ppt.txt'), 'w') as f:
        f.write('FPHA HandOccNet 实验指标（可直接复制到 PPT）\n\n')
        for key, val in summary.items():
            f.write('{}: {}\n'.format(key, val))
        f.write('\n数据集: FPHA Subject_1/handshake/1\n')
        f.write('分辨率: 640 x 480\n')
        f.write('关节数: 21 (3D)\n')
        f.write('基线模型: HandOccNet (CVPR 2022)\n')


def visualize_frame(output_dir, sample, result, rgb, depth_map):
    ensure_dir(output_dir)
    frame_id = sample['frame_id']

    fig = plt.figure(figsize=(16, 8))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(rgb)
    draw_skeleton_2d(ax_img, result.gt_2d, color='cyan', label='GT')
    if result.pred_2d is not None:
        draw_skeleton_2d(ax_img, result.pred_2d, color='red', label='Pred')
    ax_img.set_title('2D overlay frame {}'.format(frame_id))
    ax_img.axis('off')
    ax_img.legend(loc='lower left')

    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    draw_skeleton_3d(ax_3d, result.gt_3d, color='cyan', label='GT')
    if result.pred_3d is not None:
        draw_skeleton_3d(ax_3d, result.pred_3d, color='red', label='Pred')
    ax_3d.set_title('3D compare frame {}'.format(frame_id))
    ax_3d.legend(loc='best')
    plt.tight_layout()
    fig.savefig(osp.join(output_dir, '{}_overlay.png'.format(frame_id)), dpi=180)
    plt.close(fig)

    if result.pred_3d is not None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        draw_skeleton_3d(ax, result.gt_3d, color='tab:cyan', label='GT')
        draw_skeleton_3d(ax, result.pred_3d, color='tab:red', label='Pred')
        ax.set_title('GT vs Pred 3D {}'.format(frame_id))
        ax.legend(loc='best')
        fig.savefig(osp.join(output_dir, '{}_3d_compare.png'.format(frame_id)), dpi=180)
        plt.close(fig)

    if depth_map is not None:
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(depth_map, cmap='gray')
        plt.scatter(result.gt_2d[:, 0], result.gt_2d[:, 1], s=16, c='cyan', label='GT 2D')
        if result.pred_2d is not None:
            plt.scatter(result.pred_2d[:, 0], result.pred_2d[:, 1], s=16, c='red', label='Pred 2D')
        plt.axis('off')
        plt.legend(loc='lower left')
        plt.tight_layout()
        fig.savefig(osp.join(output_dir, '{}_depth_overlay.png'.format(frame_id)), dpi=180)
        plt.close(fig)

    if result.occluded_joints is not None:
        save_occlusion_heatmap(
            output_dir, rgb, depth_map, result.gt_2d, result.gt_3d,
            result.occluded_joints, frame_id,
        )


def export_video(image_paths, output_path, fps=12):
    if not image_paths:
        return
    first = cv2.imread(image_paths[0])
    if first is None:
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        writer.write(frame)
    writer.release()


def write_summary(results, output_dir):
    ensure_dir(output_dir)
    numeric = [r for r in results if r.mpjpe_mm is not None]
    summary = {
        'num_frames': len(results),
        'num_inferred': len(numeric),
        'mpjpe_mm_mean': float(np.mean([r.mpjpe_mm for r in numeric])) if numeric else None,
        'mpjpe_mm_median': float(np.median([r.mpjpe_mm for r in numeric])) if numeric else None,
        'root_error_mm_mean': float(np.mean([r.root_error_mm for r in numeric])) if numeric else None,
        'occlusion_score_mm_mean': float(np.nanmean([r.occlusion_score_mm for r in results if r.occlusion_score_mm is not None])) if results else None,
    }
    with open(osp.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    failure_cases = sorted(
        [r for r in numeric if r.mpjpe_mm is not None],
        key=lambda x: x.mpjpe_mm,
        reverse=True,
    )[:10]
    with open(osp.join(output_dir, 'failure_cases.json'), 'w') as f:
        json.dump([
            {
                'frame_id': r.frame_id,
                'mpjpe_mm': r.mpjpe_mm,
                'root_error_mm': r.root_error_mm,
                'occlusion_score_mm': r.occlusion_score_mm,
                'occluded_joints': r.occluded_joints,
            }
            for r in failure_cases
        ], f, indent=2)

    with open(osp.join(output_dir, 'analysis_report.md'), 'w') as f:
        f.write('# FPHA HandOccNet Analysis\n\n')
        f.write('- Frames processed: {}\n'.format(summary['num_frames']))
        f.write('- Frames inferred: {}\n'.format(summary['num_inferred']))
        f.write('- Mean MPJPE: {}\n'.format(summary['mpjpe_mm_mean']))
        f.write('- Median MPJPE: {}\n'.format(summary['mpjpe_mm_median']))
        f.write('- Mean root error: {}\n'.format(summary['root_error_mm_mean']))
        f.write('- Mean occlusion score: {}\n\n'.format(summary['occlusion_score_mm_mean']))
        f.write('## Typical failure sources\n')
        f.write('- Severe self-occlusion or hand-object overlap in egocentric view.\n')
        f.write('- Thin finger articulation and extreme foreshortening.\n')
        f.write('- Crop misalignment when the hand is near the image border.\n')
        f.write('- Depth holes / missing depth pixels in low-texture regions.\n')


def save_gt_summary(samples, output_dir):
    ensure_dir(output_dir)
    first = samples[0]
    rgb = load_rgb(first['rgb_path'])
    gt_3d = first['gt_3d']
    gt_2d = project_points(gt_3d)

    fig = plt.figure(figsize=(16, 8))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(rgb)
    draw_skeleton_2d(ax_img, gt_2d, color='cyan', label='GT')
    ax_img.axis('off')
    ax_img.set_title('Ground truth 2D projection')

    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    draw_skeleton_3d(ax_3d, gt_3d, color='cyan', label='GT')
    ax_3d.set_title('Ground truth 3D pose')
    plt.tight_layout()
    fig.savefig(osp.join(output_dir, 'gt_overview.png'), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='Subject_1')
    parser.add_argument('--action', default='handshake')
    parser.add_argument('--instance', default='1')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit frames; default runs the full sequence')
    parser.add_argument('--checkpoint', default=osp.join(HANDOCCNET_ROOT, 'weights', 'snapshot_demo.pth.tar'))
    parser.add_argument('--mano_root', default=osp.join(HANDOCCNET_ROOT, 'weights', 'mano_v1_2'))
    parser.add_argument('--output_dir', default=osp.join(ROOT, 'results', 'fpha_handoccnet'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument('--export_video', action='store_true')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(osp.join(args.output_dir, 'frames'))

    if osp.exists(args.mano_root):
        cfg.mano_path = args.mano_root

    samples = load_sequence_samples(args.subject, args.action, args.instance, args.max_frames)
    print('Loaded {} FPHA frames from {}/{}/{}'.format(len(samples), args.subject, args.action, args.instance))

    save_gt_summary(samples, args.output_dir)

    can_infer = osp.exists(args.checkpoint) and not args.skip_inference
    model = None
    device = None
    if can_infer:
        model, device = load_handoccnet(args)
    else:
        print('Inference disabled because checkpoint is missing, or --skip_inference was used.')

    results = []
    vis_paths = []
    for sample in tqdm(samples):
        rgb = load_rgb(sample['rgb_path'])
        depth = load_depth(sample['depth_path'])
        gt_3d = sample['gt_3d']
        gt_2d = project_points(gt_3d)
        bbox = bbox_from_gt_projection(gt_2d, rgb.shape[1], rgb.shape[0])
        occlusion_score, occluded_joints = depth_occlusion_score(depth, gt_2d, gt_3d)

        result = FrameResult(
            frame_id=sample['frame_id'],
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            occlusion_score_mm=occlusion_score,
            occluded_joints=occluded_joints,
        )

        if model is not None:
            pred_3d, pred_mesh, pred_2d = infer_one_frame(model, device, rgb, bbox)
            pred_3d = pred_3d - pred_3d[0] + gt_3d[0]
            pred_2d = refine_2d_with_depth(pred_2d, depth, rgb.shape)
            result.pred_3d = pred_3d
            result.pred_2d = pred_2d
            result.mpjpe_mm = mpjpe(pred_3d, gt_3d)
            result.root_error_mm = float(np.linalg.norm(pred_3d[0] - gt_3d[0]))

            mesh_path = osp.join(args.output_dir, '{}_mesh.obj'.format(sample['frame_id']))
            try:
                save_obj(pred_mesh * np.array([1, -1, -1]), MANO().face, mesh_path)
            except Exception as exc:
                print('Skip mesh export for frame {}: {}'.format(sample['frame_id'], exc))

        visualize_frame(osp.join(args.output_dir, 'frames'), sample, result, rgb, depth)
        vis_paths.append(osp.join(args.output_dir, 'frames', '{}_overlay.png'.format(sample['frame_id'])))
        results.append(result)

    with open(osp.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump([
            {
                'frame_id': r.frame_id,
                'mpjpe_mm': r.mpjpe_mm,
                'root_error_mm': r.root_error_mm,
                'occlusion_score_mm': r.occlusion_score_mm,
                'occluded_joints': r.occluded_joints,
            }
            for r in results
        ], f, indent=2)

    write_summary(results, args.output_dir)

    samples_by_id = {s['frame_id']: s for s in samples}
    export_failure_gallery(results, samples_by_id, args.output_dir)

    with open(osp.join(args.output_dir, 'summary.json'), 'r') as f:
        summary = json.load(f)
    export_presentation_assets(args.output_dir, summary)

    if args.export_video:
        export_video(vis_paths, osp.join(args.output_dir, 'overlay.mp4'))

    print('Saved results to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()