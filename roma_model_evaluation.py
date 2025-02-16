import os
import cv2
import torch

from romatch import roma_outdoor
from Roma_utils import *
import time
import sys
import pandas as pd


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


if __name__ == "__main__":

    # path_ = "/vlad/couples_vis_vis"
    # output_dir = path_ + "_Results"  # "/app/code/yifat/data/Results_Finder/Results_roma_inliers_24_11_renamed_560"

    # ds_path = "/mnt/utah-server-new/Finder/data/images/26_12_renamed"
    ds_path = "/home/vladislavs/finder-data/data/images/26_12_renamed_subset"

    img_path = f"{ds_path}/images"
    compare_dir = f"{ds_path}/tags_original"
    output_dir = f"{ds_path}/tags_no_upsampling406_balanced-no-th_store_fransac_autocast-dino"

    os.makedirs(output_dir, exist_ok=True)
    setup_logging(f"{output_dir}/logs.log")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    max_iter = 1e20

    save_results = False
    compare_results = True
    store = True

    original_kwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       "coarse_res": (560, 560),
                       "upsample_res": (864, 864),
                       "symmetric": True,
                       "sample_mode": 'threshold_balanced',
                       "upsample_preds": True}

    run_kwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  "coarse_res": (406, 406),  # mul 14
                  "upsample_res": (608, 608),  # mul 16
                  "symmetric": False,
                  "sample_mode": 'balanced',
                  "upsample_preds": False}

    # run_kwargs = original_kwargs
    roma_model = roma_outdoor(**run_kwargs)

    image_files = sorted(
        [os.path.join(img_path, f) for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    )

    times = {"im_read": [],
             "match": [],
             "sample": [],
             "pixel_coordinate": [],
             "gpu_cpu_copy": [],
             "fundamental": [],
             "total": []}

    for i in range(0, min(len(image_files) - 1, max_iter), 2):

        imA_path = image_files[i]
        imB_path = image_files[i + 1]

        imA_name = os.path.splitext(os.path.basename(imA_path))[0]
        imB_name = os.path.splitext(os.path.basename(imB_path))[0]

        start_start_time = time.perf_counter()
        start_time = time.perf_counter()
        img_A, img_A_ups, (W_A, H_A) = prepare_image_for_roma(imA_path,
                                                              mean=mean,
                                                              std=std,
                                                              coarse_res=run_kwargs["coarse_res"],
                                                              upsample_res=run_kwargs["upsample_res"],
                                                              device=run_kwargs["device"])

        img_B, img_B_ups, (W_B, H_B) = prepare_image_for_roma(imB_path,
                                                              mean=mean,
                                                              std=std,
                                                              coarse_res=run_kwargs["coarse_res"],
                                                              upsample_res=run_kwargs["upsample_res"],
                                                              device=run_kwargs["device"], )
        end_time = time.perf_counter()
        times['im_read'].append(end_time - start_time)

        # match
        if store:
            roma_model.extract_single_image(img_A, store=True)
            start.record()
            warp, certainty = roma_model.match(img_B,
                                               use_stored=True,
                                               device=run_kwargs['device'])

        else:
            start.record()
            warp, certainty = roma_model.match(img_A, img_B,
                                               img_A_ups, img_B_ups,
                                               device=run_kwargs['device'])
        end.record()
        torch.cuda.synchronize()
        times['match'].append(start.elapsed_time(end) / 1000)

        # sample
        start_time = time.perf_counter()
        matches, certainty = roma_model.sample(warp, certainty)
        end_time = time.perf_counter()
        times['sample'].append(end_time - start_time)

        start_time = time.perf_counter()
        kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        end_time = time.perf_counter()
        times['pixel_coordinate'].append(end_time - start_time)

        start_time = time.perf_counter()
        kptsA_array = kptsA.cpu().numpy()
        kptsB_array = kptsB.cpu().numpy()
        end_time = time.perf_counter()
        times['gpu_cpu_copy'].append(end_time - start_time)

        # Find a fundamental matrix
        start_time = time.perf_counter()
        F, mask = cv2.findFundamentalMat(
            kptsA_array,
            kptsB_array,
            ransacReprojThreshold=0.2,
            method=cv2.USAC_MAGSAC,
            confidence=0.999999,
            maxIters=10000)
        end_time = time.perf_counter()
        end_end_time = time.perf_counter()
        times['fundamental'].append(end_time - start_time)
        times['total'].append(end_end_time - start_start_time)

        if save_results:
            inliers_mask = mask.ravel().astype(bool)
            # Filter inliers
            kptsA = kptsA_array[inliers_mask][0:16]
            kptsB = kptsB_array[inliers_mask][0:16]

            keypointsA_filename = f"{output_dir}/{imA_name}.txt"
            keypointsB_filename = f"{output_dir}/{imB_name}.txt"

            np.savetxt(keypointsA_filename, kptsA, fmt='%.6f', delimiter=',', header="x,y")
            np.savetxt(keypointsB_filename, kptsB, fmt='%.6f', delimiter=',', header="x,y")

            certainty = certainty[inliers_mask][0:16]

            output_name = f"{output_dir}/{i}_{imA_name}_{imB_name}.png"
            colors = generate_distinct_colors(len(kptsA))
            colors = np.array(colors) / 255.0
            text = [
                'RoMa',
                'Matches: {}'.format(len(kptsA)),
            ]
            make_matching_figure(cv2.imread(imA_path),
                                 cv2.imread(imB_path),
                                 kptsA,
                                 kptsB,
                                 colors,
                                 text=text,
                                 dpi=75,
                                 path=output_name)

        # Output the results
        print(f"Processed pair: {imA_path}, {imB_path}")
        # print(f"Fundamental matrix F:\n{F}")
        # print(f"Inlier mask:\n{mask}")

        print(f"i: {i}, im_read: {times['im_read'][-1]:.4f} seconds")
        print(f"i: {i}, match: {times['match'][-1]:.4f} seconds")
        print(f"i: {i}, sample: {times['sample'][-1]:.4f} seconds")
        print(f"i: {i}, pixel_coordinate: {times['pixel_coordinate'][-1]:.4f} seconds")
        print(f"i: {i}, gpu_cpu_copy: {times['gpu_cpu_copy'][-1]:.4f} seconds")
        print(f"i: {i}, fundamental: {times['fundamental'][-1]:.4f} seconds")
        print(f"i: {i}, total: {times['total'][-1]:.4f} seconds")

    times_df = pd.DataFrame(times)
    times_df.to_csv(f"{output_dir}/times.csv", index=False)
    print("========================================================")
    print(f"\tAVERAGE TIMES for {run_kwargs}")
    print(f"im_read: {np.mean(times['im_read']):.4f} +- {np.std(times['im_read']):.4f} seconds")
    print(f"match: {np.mean(times['match']):.4f} +- {np.std(times['match']):.4f} seconds")
    print(f"sample: {np.mean(times['sample']):.4f} +- {np.std(times['sample']):.4f} seconds")
    print(
        f"pixel_coordinate: {np.mean(times['pixel_coordinate']):.4f} +- {np.std(times['pixel_coordinate']):.4f} seconds")
    print(f"gpu_cpu_copy: {np.mean(times['gpu_cpu_copy']):.4f} +- {np.std(times['gpu_cpu_copy']):.4f} seconds")
    print(f"fundamental: {np.mean(times['fundamental']):.4f} +- {np.std(times['fundamental']):.4f} seconds")
    print(f"total: {np.mean(times['total']):.4f} +- {np.std(times['total']):.4f} seconds")
