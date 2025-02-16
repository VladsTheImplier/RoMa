import bisect
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from PIL import Image


def generate_distinct_colors(n):
    colormap = plt.cm.get_cmap('hsv', n)
    colors = [tuple((np.array(colormap(i)[:3]) * 255).astype(int)) for i in range(n)]
    return colors

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        
        
        # fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
        #                                     (fkpts0[i, 1], fkpts1[i, 1]),
        #                                     transform=fig.transFigure, c=color[i], linewidth=1)
        #                                 for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)
    
    # plt.savefig(str("tmp.png"), bbox_inches='tight', pad_inches=0)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def prepare_image_for_roma(img_path: str, *, mean: list, std: list,
                           coarse_res: tuple[int, int], upsample_res: tuple[int, int],
                           device: str | torch.device = 'cpu', dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """ Load image, prepare for model and return downscaled, upscaled, and original (W, H) """
    mean = torch.as_tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=dtype, device=device).view(-1, 1, 1)

    pil_img = Image.open(img_path)
    w, h = pil_img.size
    pil_img = pil_img.convert("RGB")

    small_pil_img = pil_img.resize(coarse_res, Image.BICUBIC)  # Maybe can change to Area when shrinking image
    big_pil_img = pil_img.resize(upsample_res, Image.BICUBIC)

    t_small_img = np.array(small_pil_img, dtype=np.float32).transpose((2, 0, 1)) / 255.
    t_small_img = torch.as_tensor(t_small_img, dtype=dtype, device=device)
    t_small_img = t_small_img.sub_(mean).div_(std).unsqueeze(0)

    t_big_img = np.array(big_pil_img, dtype=np.float32).transpose((2, 0, 1)) / 255.
    t_big_img = torch.as_tensor(t_big_img, dtype=dtype, device=device)
    t_big_img = t_big_img.sub_(mean).div_(std).unsqueeze(0)

    return t_small_img, t_big_img, (w, h)


def load_tagged_points(txt_path: str) -> np.ndarray:
    """Load tagged points from a .txt file."""
    if os.path.exists(txt_path):
        points = np.loadtxt(txt_path, delimiter=",")
        if points.ndim == 1:  # Handle single point case
            points = points[np.newaxis, :]
        return points

    return np.array([])


def to_homogenous(a: np.ndarray) -> np.ndarray:
    """ Transform given vector of coordinates to homogeneous coordinates by adding a 1 at the end.
        e.g: [x, y] --> [x, y, 1] """
    return np.concatenate((a.reshape(-1, 1), np.ones((1, 1))))


def calc_sampson_dist(points_a: np.ndarray, points_b: np.ndarray, F: np.ndarray) -> np.ndarray:
    results = []
    for a, b in zip(points_a, points_b):
        dist = cv2.sampsonDistance(to_homogenous(a), to_homogenous(b), F)
        results.append(dist)

    return np.array(results)


def draw_points_and_lines_concat(imA, imB, points_A_tagged, points_B_tagged, epilines_B, epilines_A, output_path):
    """Concatenate two images and draw points and epipolar lines."""
    # Concatenate images horizontally
    hA, wA, _ = imA.shape
    hB, wB, _ = imB.shape
    height = max(hA, hB)
    width = wA + wB
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    canvas[:hA, :wA, :] = imA
    canvas[:hB, wA:wA + wB, :] = imB

    # Dynamically set thickness and point size based on image size
    avg_dim = (wA + hA + wB + hB) / 4
    point_radius = max(5, int(avg_dim * 0.005))  # Minimum radius of 5
    line_thickness = max(2, int(avg_dim * 0.002))  # Minimum thickness of 2

    # Draw points and epipolar lines
    for idx, (pointA_tagged, epiB, pointB_tagged, epiA) in enumerate(
            zip(points_A_tagged, epilines_B, points_B_tagged, epilines_A)):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Point in imA
        xA_tagged, yA_tagged = map(int, pointA_tagged)
        cv2.circle(canvas, (xA_tagged, yA_tagged), point_radius, color, -1)
        cv2.putText(canvas, f"A{idx}", (xA_tagged + 5, yA_tagged - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Point in imB
        xB_tagged, yB_tagged = map(int, pointB_tagged)
        cv2.circle(canvas, (xB_tagged + wA, yB_tagged), point_radius, color, -1)
        cv2.putText(canvas, f"B{idx}", (xB_tagged + wA + 5, yB_tagged - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Epipolar line in imB
        aB, bB, cB = epiB
        # Calculate the endpoints for the epipolar line in imB
        if bB != 0:
            x0B, y0B = 0, int(-cB / bB)  # Line at x=0 (Left Edge of the Image)
            x1B, y1B = wB, int(-(aB * wB + cB) / bB)  # Line at x=wB (Right Edge of the Image)
        else:  # Special case when bB == 0 (horizontal line)
            x0B, y0B = int(-cB / aB), 0
            x1B, y1B = int(-cB / aB), height
        # Draw the epipolar line in imB (adjusting x-coordinates to account for canvas layout)
        cv2.line(canvas, (x0B + wA, y0B), (x1B + wA, y1B), color, line_thickness)

        # Epipolar line in imA
        aA, bA, cA = epiA
        # Calculate the endpoints for the epipolar line in imA
        if bA != 0:
            x0A, y0A = 0, int(-cA / bA)  # Line at x=0
            x1A, y1A = wA, int(-(aA * wA + cA) / bA)  # Line at x=wA
        else:  # Special case when bA == 0 (horizontal line)
            x0A, y0A = int(-cA / aA), 0
            x1A, y1A = int(-cA / aA), height
        # Draw the epipolar line in imA
        cv2.line(canvas, (x0A, y0A), (x1A, y1A), color, line_thickness)

    # Save the result
    cv2.imwrite(output_path, canvas)