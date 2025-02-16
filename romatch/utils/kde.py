import torch


def kde(x, std = 0.1, half = True):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density