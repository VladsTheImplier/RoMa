import warnings
import torch.nn as nn
import torch
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.models.tiny import TinyRoMa


def tiny_roma_v1_model(weights=None, freeze_xfeat=False, exact_softmax=False, xfeat=None):
    model = TinyRoMa(
        xfeat=xfeat,
        freeze_xfeat=freeze_xfeat,
        exact_softmax=exact_softmax)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def roma_model(resolution, upsample_preds, symmetric, sample_mode, timing, device=None, weights=None, dinov2_weights=None,
               attenuate_cert=True, amp: bool = True, amp_dtype: torch.dtype = torch.float16, **kwargs):
    # romatch weights and dinov2 weights are loaded seperately, as dinov2 weights are not parameters
    # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul TODO: these probably ruin stuff, should be careful
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]),
        decoder_dim,
        cls_to_coord_res ** 2 + 1,
        is_classifier=True,
        amp=amp,
        pos_enc=False,
        timing=timing,)

    coordinate_decoder = torch.jit.trace(func=coordinate_decoder.cuda(),
                           example_inputs=(torch.rand((1, 512, 29, 29), device='cuda', dtype=torch.float32),
                                           torch.rand((1, 512, 29, 29), device='cuda', dtype=torch.float16)))
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefinerCoarse(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefinerCoarse(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefinerCoarse(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefinerFine(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefinerFine(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )
    # FIXME: not faster, tracer also claims wrong results :(
    # cr16 = torch.jit.trace(func=conv_refiner['16'].cuda(),
    #                        example_inputs=(torch.rand((1, 512, 30, 30), device='cuda', dtype=torch.float16),
    #                                        torch.rand((1, 512, 30, 30), device='cuda', dtype=torch.float16),
    #                                        torch.rand((1, 2, 30, 30), device='cuda', dtype=torch.float32),
    #                                        torch.tensor(1.0, device='cuda')))
    #
    # conv_refiner['16'] = cr16

    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )

    gp16 = torch.jit.trace(func=gp16.cuda(), example_inputs=(torch.rand((1, 512, 29, 29), device='cuda', dtype=torch.float16),
                                     torch.rand((1, 512, 29, 29), device='cuda', dtype=torch.float16)))
    gps = nn.ModuleDict({"16": torch.jit.script(gp16)})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
    })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(torch.jit.script(coordinate_decoder),
                      gps,
                      proj,
                      conv_refiner,
                      detach=True,
                      amp=amp,
                      scales=["16", "8", "4", "2", "1"],
                      displacement_dropout_p=displacement_dropout_p,
                      gm_warp_dropout_p=gm_warp_dropout_p,
                      timing=timing)

    encoder = CNNandDinov2(
        cnn_kwargs=dict(
            pretrained=False,  # TODO: dubious. why? performance is similar with 'True'
            amp=amp),
        amp=amp,
        device=device,
        use_vgg=True,
        dinov2_weights=dinov2_weights,
        amp_dtype=amp_dtype,
        timing=timing,
    )
    h, w = resolution
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, upsample_preds=upsample_preds,
                                symmetric=symmetric, attenuate_cert=attenuate_cert,
                                sample_mode=sample_mode, timing=timing, **kwargs).to(device)
    matcher.load_state_dict(weights)
    matcher.show_dino()
    return matcher
