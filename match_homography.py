from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
from models.matching import Matching
from utils.common import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,read_image_with_homography,
                          rotate_intrinsics, rotate_pose_inplane, compute_pixel_error,
                          scale_intrinsics, weights_mapping, download_base_files, download_test_images)
from utils.preprocess_utils import torch_find_matches

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_homography', type=str, default='assets/coco_test_images_homo.txt',
        help='Path to the list of image pairs and corresponding homographies')
    parser.add_argument(
        '--input_dir', type=str, default='assets/coco_test_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_homo_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', default='coco_homo',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--min_matches', type=int, default=12,
        help="Minimum matches required for considering matching")
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_homography, 'r') as f:
        homo_pairs = f.readlines()

    if opt.max_length > -1:
        homo_pairs = homo_pairs[0:np.min([len(homo_pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(homo_pairs)
    download_base_files()
    download_test_images()
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    try:
        curr_weights_path = str(weights_mapping[opt.superglue])
    except:
        if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
            curr_weights_path = str(opt.superglue)
        else:
            raise ValueError("Given --superglue path doesn't exist or invalid")
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights_path': curr_weights_path,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))
    timer = AverageTimer(newline=True)
    for i, info in enumerate(homo_pairs):
        split_info = info.strip().split(' ')
        image_name = split_info[0]
        homo_info = list(map(lambda x: float(x), split_info[1:]))
        homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
        stem0 = Path(image_name).stem
        matches_path = output_dir / '{}_matches.npz'.format(stem0)
        eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
        viz_path = output_dir / '{}_matches.{}'.format(stem0, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_evaluation.{}'.format(stem0, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))
            continue
        image0, image1, inp0, inp1, scales0, homo_matrix = read_image_with_homography(input_dir / image_name, homo_matrix, device,
                                                opt.resize, 0, opt.resize_float)

        if image0 is None or image1 is None:
            print('Problem reading image pair: {}'.format(
                input_dir/ image_name))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, torch.from_numpy(homo_matrix).to(kp0_torch.device), dist_thresh=3, n_iters=3)
        ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
        gt_match_vec = np.ones((len(matches), ), dtype=np.int32) * -1
        gt_match_vec[ma_0] = ma_1
        corner_points = np.array([[0,0], [0, image0.shape[0]], [image0.shape[1], image0.shape[0]], [image0.shape[1], 0]]).astype(np.float32)
        if do_eval:
            if len(mconf) < opt.min_matches:
                out_eval = {'error_dlt': -1,
                            'error_ransac': -1,
                            'precision': -1,
                            'recall': -1
                            }
                #non matched points will not be considered for evaluation
                np.savez(str(eval_path), **out_eval)
                timer.update('eval')
                print('Skipping {} due to inefficient matches'.format(i))
                continue
            sort_index = np.argsort(mconf)[::-1][0:4]
            est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
            est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
            corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
            corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(1)
            corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), homo_matrix).squeeze(1)
            error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
            error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
            match_flag = (matches[ma_0] == ma_1)
            precision = match_flag.sum() / valid.sum()
            fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
            recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
            # Write the evaluation results to disk.
            out_eval = {'error_dlt': error_dlt,
                        'error_ransac': error_ransac,
                        'precision': precision,
                        'recall': recall
                        }
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem0),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))

    if opt.eval:
    # Collate the results into a final table and print to terminal.
        errors_dlt = []
        errors_ransac = []
        precisions = []
        recall = []
        matching_scores = []
        for info in homo_pairs:
            split_info = info.strip().split(' ')
            image_name = split_info[0]
            stem0 = Path(image_name).stem
            eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
            results = np.load(eval_path)
            if results['precision'] == -1:
                continue
            errors_dlt.append(results['error_dlt'])
            errors_ransac.append(results['error_ransac'])
            precisions.append(results['precision'])
            recall.append(results['recall'])
        thresholds = [5, 10, 25]
        aucs_dlt = pose_auc(errors_dlt, thresholds)
        aucs_ransac = pose_auc(errors_ransac, thresholds)
        aucs_dlt = [100.*yy for yy in aucs_dlt]
        aucs_ransac = [100.*yy for yy in aucs_ransac]
        prec = 100.*np.mean(precisions)
        rec = 100.*np.mean(recall)
        print('Evaluation Results (mean over {} pairs):'.format(len(homo_pairs)))
        print("For DLT results...")
        print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec))
        print("For homography results...")
        print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec))