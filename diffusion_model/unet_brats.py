#-*- coding:utf-8 -*-
#
# Original code is here: https://github.com/openai/guided-diffusion
#
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .modules import *

NUM_CLASSES = 1
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    # from VQGAN
    return self

class model_ensemble(nn.Module):
    def __init__(self, model_3D, model2D_2, model2D_3, batch_size_2D_inference = 8, time_step = 1000, out_channels = 1, ntime_steps_2D = 1000, baseline = "3D"):
        super(model_ensemble, self).__init__()
        self.ntime_steps_2D = ntime_steps_2D
        self.model_3D = model_3D
        self.model2D_2 = model2D_2.netG
        self.model2D_3 = model2D_3.netG
        self.baseline = baseline
        # TODO change this from hard coding
        self.featue_c = 64
        # if model2D_3 is a DDP model, we need to access the module
        if hasattr(self.model2D_3, "module"):
            self.model2D_3 = self.model2D_3.module
            self.model2D_2 = self.model2D_2.module
            
        self.batch_size_2D_inference = batch_size_2D_inference
        self.time_step = time_step
        if (192 % batch_size_2D_inference != 0) or (152 % batch_size_2D_inference != 0):
            print("invalid batch size 2D",batch_size_2D_inference)
            raise ValueError("batch_size_2D_inference must be a factor of 192 and 152")
        self.out_c = out_channels
        self.model2D_2.denoise_fn.eval()
        self.model2D_3.denoise_fn.eval()
        for param in self.model2D_2.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_3.denoise_fn.parameters():
            param.requires_grad = False
        # This only works for pl.lightning.LightningModule
        # self.model2D_2.denoise_fn = self.model2D_2.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
        # self.model2D_3.denoise_fn = self.model2D_3.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
    def debug_viz(self, arr, name = "debug/viz"):
        import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                plt.imshow(arr[i,j,:,:])
                plt.savefig(f"{name}_{i}_{j}.png")
                
    def forward(self, x, timesteps):
        # x shape is (batch_size, c, 152, 192, 192)
        # print("debugging t for 3D:", timesteps)
        b, c, d, h, w = x.shape
        b_2D = self.batch_size_2D_inference
        # print(timesteps)
        with torch.no_grad():
            if self.baseline == "3D_only":
                # do not need to do 2D inference if we only use 3D model
                pass
            elif not self.baseline == "3D_feature":
                # no feature is calculated here
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D] = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                
                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D] = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
            else: 
                # 2D_features are calculated here
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                feature_2D_2 = torch.zeros(b*h, self.featue_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D], F = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    feature_2D_2[i:i+b_2D] = F[-2]

                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                feature_2D_2 = feature_2D_2.reshape(b, h, self.featue_c, d, w).permute(0, 2, 3, 1, 4)

                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                feature_2D_3 = torch.zeros(b*w, self.featue_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D], F = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    feature_2D_3[i:i+b_2D] = F[-1]

                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
                feature_2D_3 = feature_2D_3.reshape(b, w, self.featue_c, d, h).permute(0, 2, 3, 4, 1)

        if self.baseline == "2D":
            w2 = 0.0
            w3 = 1.0
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "merge":
            w2 = 0.5
            w3 = 0.5
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "TPDM":
            w2 = 0.5
            # randeomly pick one of the 2D model to use
            if np.random.rand() > w2:
                return out_2D_2
            else:
                return out_2D_3 
        elif self.baseline == "3D":
            # inference for 3D model
            x_3D = torch.cat([x, out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps)
            out_3D = ensemble_w * out_2D_2 + (1 - ensemble_w) * out_2D_3
            return out_3D
        elif self.baseline == "3D_only":
            # inference for 3D model only
            out_3D = self.model_3D(x, timesteps)
            return out_3D
        elif self.baseline == "3D_feature":
            # inference for 3D model
            x_3D = torch.cat([x, out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            feature_3D = torch.cat([feature_2D_2.detach(), feature_2D_3.detach()], dim = 1)
            x_3D_final = torch.cat([x_3D], dim = 1)
            
            ensemble_w = self.model_3D(x_3D_final, timesteps, feature_2D = feature_3D)
            out_3D = ensemble_w * out_2D_2 + (1 - ensemble_w) * out_2D_3
            return out_3D
        else:
            raise ValueError("baseline must be 2D, 3D or merge")


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_2D_feature = False
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.use_2D_feature = use_2D_feature
        if use_2D_feature:
            self.feature_2D_channels = [128]

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32
        self.use_fp16 = use_fp16
        print("use_fp16: ", self.use_fp16)
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1, padding_mode = 'replicate'))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for layers_idx in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch + (self.feature_2D_channels[0] if (self.use_2D_feature and layers_idx == 3) else 0),
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    dim_shift = True if (level == 4) else False
                    # print(level, dim_shift)
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            out_dim_shift_depth = dim_shift
                        )
                        if resblock_updown
                        # for the first up sample layer we need to have the same size as the last downsample layer
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, out_dim_shift_depth=dim_shift)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode = 'replicate')),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, feature_2D = None, y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.use_fp16): 
            
            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if self.num_classes is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + self.label_emb(y)

            h = x #.type(self.dtype)
            # if not self.use_2D_feature:
            for i, module in enumerate(self.input_blocks):
                if self.use_2D_feature and i == 3:
                    h = th.cat([h, feature_2D], dim=1)
                h = module(h, emb)
                hs.append(h)
            # raise
            
            h = self.middle_block(h, emb)
            for module in self.output_blocks:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb)
            h = h #.type(x.dtype)
            return self.out(h)


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=8,
    out_channels=4,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 192:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    if len(attention_resolutions) > 0:
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
