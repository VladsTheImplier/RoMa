import bisect
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
