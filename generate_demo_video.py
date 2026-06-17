"""Generate demo videos from existing pipeline visualization frames."""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np

ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, ROOT)

from fpha_handoccnet_pipeline import ensure_dir, export_video


def sorted_frame_ids(frames_dir):
    ids = []
    for name in os.listdir(frames_dir):
        if name.endswith('_overlay.png') and name.count('_') == 1:
            ids.append(name.replace('_overlay.png', ''))
    return sorted(ids, key=lambda x: int(x))


def make_grid_frame(paths, tile_w=640, tile_h=480):
    tiles = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (tile_w, tile_h))
        tiles.append(img)
    while len(tiles) < 4:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


def export_grid_video(frames_dir, frame_ids, output_path, fps=12, tile_w=640, tile_h=480):
    if not frame_ids:
        return
    first = make_grid_frame([
        osp.join(frames_dir, '{}_overlay.png'.format(frame_ids[0])),
        osp.join(frames_dir, '{}_3d_compare.png'.format(frame_ids[0])),
        osp.join(frames_dir, '{}_depth_overlay.png'.format(frame_ids[0])),
        osp.join(frames_dir, '{}_occlusion_heatmap.png'.format(frame_ids[0])),
    ], tile_w, tile_h)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for fid in frame_ids:
        frame = make_grid_frame([
            osp.join(frames_dir, '{}_overlay.png'.format(fid)),
            osp.join(frames_dir, '{}_3d_compare.png'.format(fid)),
            osp.join(frames_dir, '{}_depth_overlay.png'.format(fid)),
            osp.join(frames_dir, '{}_occlusion_heatmap.png'.format(fid)),
        ], tile_w, tile_h)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=osp.join(ROOT, 'results', 'final'))
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    frames_dir = osp.join(args.results_dir, 'frames')
    if not osp.isdir(frames_dir):
        raise FileNotFoundError('Missing frames dir: {}'.format(frames_dir))

    frame_ids = sorted_frame_ids(frames_dir)
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    overlay_paths = [osp.join(frames_dir, '{}_overlay.png'.format(fid)) for fid in frame_ids]
    overlay_out = osp.join(args.results_dir, 'demo_overlay.mp4')
    grid_out = osp.join(args.results_dir, 'demo_full.mp4')

    export_video(overlay_paths, overlay_out, fps=args.fps)
    export_grid_video(frames_dir, frame_ids, grid_out, fps=args.fps)

    ppt_dir = osp.join(args.results_dir, 'ppt_assets')
    ensure_dir(ppt_dir)
    for src in [overlay_out, grid_out]:
        if osp.exists(src):
            import shutil
            shutil.copy2(src, osp.join(ppt_dir, osp.basename(src)))

    print('Generated:')
    print('  {} ({} frames, {} fps, ~{:.1f}s)'.format(
        overlay_out, len(frame_ids), args.fps, len(frame_ids) / args.fps))
    print('  {} (2x2 multi-view)'.format(grid_out))
    print('Copied to: {}'.format(ppt_dir))


if __name__ == '__main__':
    main()
