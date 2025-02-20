import os
import cv2
os.environ['PYTORCH_JIT'] = '1'

import numpy as np
import torch
from constants import MEAN, STD
from romatch import roma_outdoor
from Roma_utils import sample, prepare_image_for_roma, generate_distinct_colors, make_matching_figure, \
    draw_points_and_lines_concat, calc_sampson_dist, load_tagged_points
import time
import sys
import pandas as pd

from romatch.models.matcher import RegressionMatcher


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()  # Ensure real-time output

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(output_dir):
    """Sets up logging to both terminal and a log file."""
    os.makedirs(output_dir + '/distances_original', exist_ok=True)
    log_file = open(f"{output_dir}/output.log", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    return log_file


def print_iter_times(iteration, times):
    print(f"i: {iteration}, im_read: {times['im_read'][-1]:.4f} seconds")
    print(f"i: {iteration}, match: {times['match'][-1]:.4f} seconds")
    print(f"i: {iteration}, sample: {times['sample'][-1]:.4f} seconds")
    print(f"i: {iteration}, pixel_coordinate: {times['pixel_coordinate'][-1]:.4f} seconds")
    print(f"i: {iteration}, gpu_cpu_copy: {times['gpu_cpu_copy'][-1]:.4f} seconds")
    print(f"i: {iteration}, fundamental: {times['fundamental'][-1]:.4f} seconds")
    print(f"i: {iteration}, total: {times['total'][-1]:.4f} seconds")


def run_and_time_roma(roma_model: RegressionMatcher,
                      imA_path: str,
                      imB_path: str,
                      run_kwargs: dict):
    times = {}
    start_cuda = torch.cuda.Event(enable_timing=True)
    end_cuda = torch.cuda.Event(enable_timing=True)
    start_start_time = time.perf_counter()

    start_time = time.perf_counter()
    img_A, img_A_ups, (W_A, H_A) = prepare_image_for_roma(imA_path,
                                                          mean=MEAN,
                                                          std=STD,
                                                          coarse_res=run_kwargs["coarse_res"],
                                                          upsample_res=run_kwargs["upsample_res"],
                                                          device=run_kwargs["device"])

    img_B, img_B_ups, (W_B, H_B) = prepare_image_for_roma(imB_path,
                                                          mean=MEAN,
                                                          std=STD,
                                                          coarse_res=run_kwargs["coarse_res"],
                                                          upsample_res=run_kwargs["upsample_res"],
                                                          device=run_kwargs["device"], )
    end_time = time.perf_counter()
    times['im_read'] = end_time - start_time

    # match
    if run_kwargs['store']:
        roma_model.extract_single_image(img_A, store=True)
        start_cuda.record()
        warp, certainty = roma_model.match(img_B,
                                           use_stored=True,
                                           device=run_kwargs['device'])

    else:
        start_cuda.record()
        warp, certainty = roma_model.match(img_A, img_B,
                                           img_A_ups, img_B_ups,
                                           device=run_kwargs['device'])
    end_cuda.record()
    torch.cuda.synchronize()
    times['match'] = start_cuda.elapsed_time(end_cuda) / 1000

    # sample
    start_cuda.record()
    matches, certainty = sample(warp, certainty,
                                sample_mode=run_kwargs['sample_mode'],
                                expansion_factor=run_kwargs['expansion_factor'],
                                reduction_factor=run_kwargs['reduction_factor'])
    end_cuda.record()
    torch.cuda.synchronize()
    times['sample'] = start_cuda.elapsed_time(end_cuda) / 1000

    start_cuda.record()
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    end_cuda.record()
    torch.cuda.synchronize()
    times['pixel_coordinate'] = start_cuda.elapsed_time(end_cuda) / 1000

    start_time = time.perf_counter()
    kptsA_array = kptsA.cpu().numpy()
    kptsB_array = kptsB.cpu().numpy()
    end_time = time.perf_counter()
    times['gpu_cpu_copy'] = end_time - start_time

    # Find a fundamental matrix
    start_time = time.perf_counter()
    F, mask = cv2.findFundamentalMat(
        kptsA_array,
        kptsB_array,
        ransacReprojThreshold=run_kwargs['ransacReprojThreshold'],
        method=cv2.USAC_MAGSAC,
        confidence=run_kwargs['confidence'],
        maxIters=run_kwargs['maxIters'])

    end_time = time.perf_counter()
    end_end_time = time.perf_counter()
    times['fundamental'] = end_time - start_time
    times['total'] = end_end_time - start_start_time

    # Filter inliers
    inliers_mask = mask.ravel().astype(bool)
    kptsA = kptsA_array[inliers_mask][0:16]
    kptsB = kptsB_array[inliers_mask][0:16]
    certainty = certainty[inliers_mask][0:16]
    return {'pointsA': kptsA, 'pointsB': kptsB, 'certainty': certainty, 'Fundamental': F, 'times': times}


def main():
    # path_ = "/vlad/couples_vis_vis"
    # output_dir = path_ + "_Results"  # "/app/code/yifat/data/Results_Finder/Results_roma_inliers_24_11_renamed_560"

    # TODO: on jetson
    # ds_path = "/vlad/couples_vis_vis"
    # compare_dir = f"/vlad/couples_vis_vis_Results"
    # img_path = f"{ds_path}"

    ds_path = "/home/vladislavs/finder-data/data/thermal_data/preprocessed"

    compare_dir = f"{ds_path}/tags/couples_vis_thermal" #_colorado_river_inverse_bigfixed"
    # compare_dir = "/home/vladislavs/finder-data/colmap-tags-exported/DJI_20240313133809_0002_V.hf5"

    img_path = f"{ds_path}/images/couples_vis_thermal" #_colorado_river_inverse_bigfixed"

    output_dir = f"/home/vladislavs/finder-data/roma_eval/16-02-25-fastest/"

    max_iter = 1e20

    save_results = 1
    compare_to_roma = 1
    compare_to_colmap = 0

    original_kwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       "coarse_res": (560, 560),
                       "upsample_res": (864, 864),
                       "upsample_preds": True,
                       "symmetric": True,
                       "sample_mode": 'threshold_balanced_approx',
                       "expansion_factor": 4.,
                       "reduction_factor": 1.,
                       "store": False,
                       "ransacReprojThreshold": 0.2,
                       "confidence": 0.99999,
                       "maxIters": 10_000,
                       "timing": True,
                       }

    run_kwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  "coarse_res": (420, 420),  # mul 14
                  "upsample_res": None,  # mul 16
                  "upsample_preds": False,
                  "symmetric": False,
                  "sample_mode": 'balanced_approx',
                  "expansion_factor": 1.5,
                  "reduction_factor": 2.,
                  "store": True,
                  "ransacReprojThreshold": 1.,
                  "confidence": 0.99,
                  "maxIters": 1_000,
                  "timing": False,
                  }

    run_kwargs = original_kwargs
    roma_model = roma_outdoor(**run_kwargs)

    times = {"im_read": [],
             "match": [],
             "sample": [],
             "pixel_coordinate": [],
             "gpu_cpu_copy": [],
             "fundamental": [],
             "total": []}

    if not (compare_to_roma ^ compare_to_colmap):
        raise AssertionError(f"Either compare_to_roma or compare_to_colmap has to be True, but not both")

    if save_results:
        os.makedirs(f"{output_dir}/df", exist_ok=True)
        setup_logging(f"{output_dir}/logs.log")

    print(run_kwargs)

    if compare_to_roma:
        results = {"query": [],
                   "reference": [],
                   "run_kwargs": [],
                   "error_sampson": [],
                   "certainty_top_16": []}

        image_files = sorted(
            [os.path.join(img_path, f) for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        )

        for idx in range(0, min(len(image_files), int(max_iter)), 2):

            imA_path = image_files[idx]
            imB_path = image_files[idx + 1]

            imA_name = os.path.splitext(os.path.basename(imA_path))[0]
            imB_name = os.path.splitext(os.path.basename(imB_path))[0]
            print(f"Processing pair: {imA_path}, {imB_path}")

            # Load tagged points
            txtA_path = os.path.join(compare_dir, f"{imA_name}.txt")
            txtB_path = os.path.join(compare_dir, f"{imB_name}.txt")

            tagged_points_A = load_tagged_points(txtA_path)
            tagged_points_B = load_tagged_points(txtB_path)
            if not tagged_points_A.any() or not tagged_points_B.any():
                print(f"Skipping pair, no tags found")
                continue

            results_dict = run_and_time_roma(roma_model, imA_path, imB_path, run_kwargs)
            [times[k].append(v) for k, v in results_dict['times'].items()]

            distances = calc_sampson_dist(tagged_points_A, tagged_points_B, results_dict['Fundamental'])

            results['query'].append(imA_path)
            results['reference'].append(imB_path)
            results['run_kwargs'].append({k: v for k, v in run_kwargs.items() if k != 'device'} if idx == 0 else None)
            results['error_sampson'].append(distances)
            results['certainty_top_16'].append(results_dict['certainty'].cpu().numpy())

            if save_results:
                keypointsA_filename = f"{output_dir}/{imA_name}.txt"
                keypointsB_filename = f"{output_dir}/{imB_name}.txt"
                np.savetxt(keypointsA_filename, results_dict['pointsA'], fmt='%.6f', delimiter=',', header="x,y")
                np.savetxt(keypointsB_filename, results_dict['pointsB'], fmt='%.6f', delimiter=',', header="x,y")

                output_name = f"{output_dir}/{idx}_{imA_name}_{imB_name}.png"
                colors = generate_distinct_colors(len(results_dict['pointsA']))
                colors = np.array(colors) / 255.0
                text = [
                    'RoMa',
                    'Matches: {}'.format(len(results_dict['pointsA'])),
                ]
                make_matching_figure(cv2.imread(imA_path),
                                     cv2.imread(imB_path),
                                     results_dict['pointsA'],
                                     results_dict['pointsB'],
                                     colors,
                                     text=text,
                                     dpi=75,
                                     path=output_name)

                # Compute epipolar lines for tagged points
                # (1) maps points from the first image to lines in the second image.
                epilines_B = cv2.computeCorrespondEpilines(tagged_points_A.reshape(-1, 1, 2), 1,
                                                           results_dict['Fundamental']).reshape(-1, 3)
                # (2) maps points from the second image to lines in the first image
                epilines_A = cv2.computeCorrespondEpilines(tagged_points_B.reshape(-1, 1, 2), 2,
                                                           results_dict['Fundamental']).reshape(-1, 3)
                imA = cv2.imread(imA_path)
                imB = cv2.imread(imB_path)

                # Draw concatenated image with points and epipolar lines
                draw_points_and_lines_concat(imA, imB, tagged_points_A, tagged_points_B, epilines_B, epilines_A,
                                             output_path=f"{output_dir}/{imA_name}_{imB_name}_PredLine2GTPoint.jpg")

            print_iter_times(idx, times)

        if save_results:
            results_df = pd.DataFrame(results)
            results_df.to_hdf(f"{output_dir}/df/results-cmp-RoMa-vis_thermal_no-inv.hf5", key='df', index=False)

    if compare_to_colmap:
        results = {"run_kwargs": [],
                   "distances_sampson": [],
                   "certainty_top_16": []}

        df_colmap: pd.DataFrame = pd.read_hdf(compare_dir, key='df')
        dset = df_colmap['dataset'].iloc[0]

        for idx, row in df_colmap.iterrows():
            imA_path = row['query']
            imB_path = row['reference']
            print(f"Processed pair: {imA_path}, {imB_path}")

            results_dict = run_and_time_roma(roma_model, imA_path, imB_path, run_kwargs)
            [times[k].append(v) for k, v in results_dict['times'].items()]

            tagged_points_A = np.array(row['query_points'])
            tagged_points_B = np.array(row['reference_points'])

            distances = calc_sampson_dist(tagged_points_A, tagged_points_B, results_dict['Fundamental'])
            results['run_kwargs'].append({k: v for k, v in run_kwargs.items() if k != 'device'} if idx == 0 else None)
            results['distances_sampson'].append(distances)
            results['certainty_top_16'].append(results_dict['certainty'].cpu().numpy())

            if save_results:
                imA_name = os.path.splitext(os.path.basename(imA_path))[0]
                imB_name = os.path.splitext(os.path.basename(imB_path))[0]

                keypointsA_filename = f"{output_dir}/{imA_name}.txt"
                keypointsB_filename = f"{output_dir}/{imB_name}.txt"

                np.savetxt(keypointsA_filename, results_dict['pointsA'], fmt='%.6f', delimiter=',', header="x,y")
                np.savetxt(keypointsB_filename, results_dict['pointsB'], fmt='%.6f', delimiter=',', header="x,y")

                output_name = f"{output_dir}/{idx}_{imA_name}_{imB_name}.png"
                colors = generate_distinct_colors(len(results_dict['pointsA']))
                colors = np.array(colors) / 255.0
                text = [f"RoMa-Matches: {len(results_dict['pointsA'])}-angle={abs(row['angle'])}"]
                make_matching_figure(cv2.imread(imA_path),
                                     cv2.imread(imB_path),
                                     results_dict['pointsA'],
                                     results_dict['pointsB'],
                                     colors,
                                     text=text,
                                     dpi=75,
                                     path=output_name)

                # Compute epipolar lines for tagged points
                # (1) maps points from the first image to lines in the second image.
                epilines_B = cv2.computeCorrespondEpilines(tagged_points_A.reshape(-1, 1, 2), 1,
                                                           results_dict['Fundamental']).reshape(-1, 3)
                # (2) maps points from the second image to lines in the first image
                epilines_A = cv2.computeCorrespondEpilines(tagged_points_B.reshape(-1, 1, 2), 2,
                                                           results_dict['Fundamental']).reshape(-1, 3)
                imA = cv2.imread(imA_path)
                imB = cv2.imread(imB_path)

                # Draw concatenated image with points and epipolar lines
                draw_points_and_lines_concat(imA, imB, tagged_points_A, tagged_points_B, epilines_B, epilines_A,
                                             output_path=f"{output_dir}/{imA_name}_{imB_name}_PredLine2GTPoint.jpg")

            print_iter_times(idx, times)

        results_df = df_colmap.join(pd.DataFrame(results), how='outer')
        results_df.to_hdf(f"{output_dir}/df/results-cmp-colmap-{dset}.hf5", key='df', index=False)

    print("========================================================")
    print(f"\tAVERAGE TIMES for {run_kwargs}")
    print(f"im_read: {np.mean(times['im_read'][10:]):.4f} +- {np.std(times['im_read'][10:]):.4f} seconds")
    print(f"match: {np.mean(times['match'][10:]):.4f} +- {np.std(times['match'][10:]):.4f} seconds")
    print(f"sample: {np.mean(times['sample'][10:]):.4f} +- {np.std(times['sample'][10:]):.4f} seconds")
    print(
        f"pixel_coordinate: {np.mean(times['pixel_coordinate'][10:]):.4f} +- {np.std(times['pixel_coordinate'][10:]):.4f} seconds")
    print(
        f"gpu_cpu_copy: {np.mean(times['gpu_cpu_copy'][10:]):.4f} +- {np.std(times['gpu_cpu_copy'][10:]):.4f} seconds")
    print(f"fundamental: {np.mean(times['fundamental'][10:]):.4f} +- {np.std(times['fundamental'][10:]):.4f} seconds")
    print(f"total: {np.mean(times['total'][10:]):.4f} +- {np.std(times['total'][10:]):.4f} seconds")

    # times_df = pd.DataFrame(times)
    # times_df.to_csv(f"{output_dir}/times.csv", index=False)


if __name__ == "__main__":
    main()
