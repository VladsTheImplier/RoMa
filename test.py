from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt

from romatch import roma_outdoor
from romatch.models.encoders import CNNandDinov2

weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
}


def main():

    s = torch.cuda.Event(enable_timing=True)
    f = torch.cuda.Event(enable_timing=True)

    run_kwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  "coarse_res": (406, 406),  # mul 14
                  "upsample_res": (608, 608),  # mul 16
                  "symmetric": False,
                  "sample_mode": 'threshold_balanced',
                  "upsample_preds": False}

    x = torch.randn((1, 3, 406, 406)).half().cuda()

    # roma_model = roma_outdoor(**run_kwargs).eval().cuda()
    # trt_model = roma_model  #torch.compile(roma_model) #, backend="tensorrt")
    # w, c = trt_model.match(x, x)
    # m, c = trt_model.sample(w, c)

    dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                        map_location="cuda")
    encoder = CNNandDinov2(
        cnn_kwargs=dict(
            pretrained=False,
            amp=True),
        amp=True,
        use_vgg=True,
        dinov2_weights=dinov2_weights,
        amp_dtype=torch.float16,
    ).cuda()

    trace = torch.jit.trace(encoder.half(), x, strict=False)
    # NamedTuple()

    times = []
    for _ in range(100):
        s.record()
        w, c = trt_model.match(x, x)
        m, c = trt_model.sample(w, c)
        f.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(f))

    print(f"{np.mean(times)} ms +- {np.std(times)}")

if __name__ == '__main__':
    main()