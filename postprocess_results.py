"""Post-process existing pipeline results: add occlusion heatmaps, failure gallery, PPT assets."""

import argparse
import json
import os.path as osp
import sys

ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, ROOT)

from fpha_handoccnet_pipeline import (
    FrameResult,
    export_failure_gallery,
    export_presentation_assets,
    load_rgb,
    load_depth,
    load_sequence_samples,
    parse_skeleton_line,
    project_points,
    depth_occlusion_score,
    visualize_frame,
    ensure_dir,
)


def load_results_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def rebuild_frame_results(samples, records, subject, action, instance):
    results = []
    record_map = {r['frame_id']: r for r in records}
    for sample in samples:
        fid = sample['frame_id']
        rec = record_map.get(fid, {})
        gt_3d = sample['gt_3d']
        gt_2d = project_points(gt_3d)
        results.append(FrameResult(
            frame_id=fid,
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            mpjpe_mm=rec.get('mpjpe_mm'),
            root_error_mm=rec.get('root_error_mm'),
            occlusion_score_mm=rec.get('occlusion_score_mm'),
            occluded_joints=rec.get('occluded_joints'),
        ))
    return results


def add_heatmaps_only(results_dir, subject, action, instance, reinfer_worst=5):
    samples = load_sequence_samples(subject, action, instance, max_frames=None)
    records = load_results_json(osp.join(results_dir, 'results.json'))
    results = rebuild_frame_results(samples, records, subject, action, instance)
    frames_dir = osp.join(results_dir, 'frames')
    ensure_dir(frames_dir)

    samples_by_id = {s['frame_id']: s for s in samples}
    numeric = [r for r in results if r.mpjpe_mm is not None]
    if numeric and reinfer_worst > 0:
        from fpha_handoccnet_pipeline import (
            HANDOCCNET_ROOT, bbox_from_gt_projection, infer_one_frame, load_handoccnet, refine_2d_with_depth,
        )
        import argparse
        ckpt = osp.join(HANDOCCNET_ROOT, 'weights', 'snapshot_demo.pth.tar')
        if osp.exists(ckpt):
            args = argparse.Namespace(checkpoint=ckpt, gpu='0')
            model, device = load_handoccnet(args)
            if model is not None:
                worst_ids = {r.frame_id for r in sorted(numeric, key=lambda x: x.mpjpe_mm, reverse=True)[:reinfer_worst]}
                for result in results:
                    if result.frame_id not in worst_ids:
                        continue
                    sample = samples_by_id[result.frame_id]
                    rgb = load_rgb(sample['rgb_path'])
                    depth = load_depth(sample['depth_path'])
                    bbox = bbox_from_gt_projection(result.gt_2d, rgb.shape[1], rgb.shape[0])
                    pred_3d, _, pred_2d = infer_one_frame(model, device, rgb, bbox)
                    pred_3d = pred_3d - pred_3d[0] + result.gt_3d[0]
                    pred_2d = refine_2d_with_depth(pred_2d, depth, rgb.shape)
                    result.pred_3d = pred_3d
                    result.pred_2d = pred_2d

    for sample, result in zip(samples, results):
        rgb = load_rgb(sample['rgb_path'])
        depth = load_depth(sample['depth_path'])
        if result.occluded_joints is None:
            score, occluded = depth_occlusion_score(depth, result.gt_2d, result.gt_3d)
            result.occlusion_score_mm = score
            result.occluded_joints = occluded
        from fpha_handoccnet_pipeline import save_occlusion_heatmap
        save_occlusion_heatmap(
            frames_dir, rgb, depth, result.gt_2d, result.gt_3d,
            result.occluded_joints, sample['frame_id'],
        )

    export_failure_gallery(results, samples_by_id, results_dir)
    with open(osp.join(results_dir, 'summary.json'), 'r') as f:
        summary = json.load(f)
    export_presentation_assets(results_dir, summary)
    print('Post-processed:', results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=osp.join(ROOT, 'results', 'fpha_with_weights'))
    parser.add_argument('--subject', default='Subject_1')
    parser.add_argument('--action', default='handshake')
    parser.add_argument('--instance', default='1')
    args = parser.parse_args()
    add_heatmaps_only(args.results_dir, args.subject, args.action, args.instance)


if __name__ == '__main__':
    main()
