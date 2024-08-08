# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple,List

import numpy as np
import torch
import torch.nn as nn

from diffusers.models.attention_processor import Attention


from diffusers.utils import BaseOutput, is_torch_version
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor,if_enhance=False) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        enhance_feature_list=[]

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                enhance_feature_list.append(sample)
                sample = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if if_enhance:
            return sample,enhance_feature_list
        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        human_in_list:list = [512,256,128,128],
        cloth_in_list:list = [512,256,128,128],
        heads:list=[16,16,8,4]
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.human_fusion_blocks=nn.ModuleList([])
        self.cloth_fusion_blocks=nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            if not is_final_block:  # 3 Fusion Block
                cloth_fusion_block=Cloth_Fusion_Block(in_ch=prev_output_channel,cloth_in_ch=cloth_in_list[i],heads=heads[i],dim_head=32)
                self.cloth_fusion_blocks.append(cloth_fusion_block)

                human_fusion_block=Human_Fusion_Block(out_ch=prev_output_channel,human_in_ch=human_in_list[i],)  
                self.human_fusion_blocks.append(human_fusion_block)

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            
            
            prev_output_channel = output_channel


        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False


    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
        cloth_feature_list:List[torch.FloatTensor] = None,
        human_feature_list:List[torch.FloatTensor] = None,
        mask:torch.FloatTensor=None,

    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    
                    with torch.no_grad():
                        if mask.shape[-2]!=sample.shape[-2] or mask.shape[-1]!=sample.shape[-1]:
                            mask=torch.nn.functional.interpolate(mask,(sample.shape[-2],sample.shape[-1]))
                        if (mask.shape)==3:
                            mask=mask.unsqueeze(dim=1)
                    
                    if i <len(human_feature_list):
                        human_enhance= torch.utils.checkpoint.checkpoint(
                            create_custom_forward(self.human_fusion_blocks[i]), 
                            sample, 
                            human_feature_list[i],
                            use_reentrant=False,) 
                        cloth_enhance= torch.utils.checkpoint.checkpoint(
                                create_custom_forward(self.cloth_fusion_blocks[i]), 
                                sample, 
                                cloth_feature_list[i],
                                use_reentrant=False,)
                            
                        sample=sample+mask*cloth_enhance+(1-mask)*human_enhance


                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    with torch.no_grad():
                        if mask.shape[-2]!=sample.shape[-2] or mask.shape[-1]!=sample.shape[-1]:
                            mask=torch.nn.functional.interpolate(mask,(sample.shape[-2],sample.shape[-1]))
                        if (mask.shape)==3:
                            mask=mask.unsqueeze(dim=1)
                
                    if i <len(human_feature_list):
                        human_enhance= torch.utils.checkpoint.checkpoint(
                            create_custom_forward(self.human_fusion_blocks[i]), sample, human_feature_list[i]) 
                        cloth_enhance= torch.utils.checkpoint.checkpoint(
                            create_custom_forward(self.cloth_fusion_blocks[i]), sample, cloth_feature_list[i])
                        
                        sample=sample+mask*cloth_enhance+(1-mask)*human_enhance

                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)
            # up
            for i,up_block in enumerate(self.up_blocks):
                with torch.no_grad():
                    if mask.shape[-2]!=sample.shape[-2] or mask.shape[-1]!=sample.shape[-1]:
                        mask=torch.nn.functional.interpolate(mask,(sample.shape[-2],sample.shape[-1]))
                    if (mask.shape)==3:
                        mask=mask.unsqueeze(dim=1)
                
                if i <len(human_feature_list):
                    human_enhance= self.human_fusion_blocks[i](human_feature_list[i])
                    cloth_enhance=self.cloth_fusion_blocks[i](sample,cloth_feature_list[i])
                    sample=sample+mask*cloth_enhance+(1-mask)*human_enhance
                
                sample = up_block(sample, latent_embeds)
            
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Enhance_Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types[:-1]):  #only need three down_block
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)


        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        enhance_feature_list=[]

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                    enhance_feature_list.append(sample)
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                    enhance_feature_list.append(sample)

        else:
            # down
            for down_block in self.down_blocks:  #only three down_blocks
                sample = down_block(sample)
                enhance_feature_list.append(sample)

        
        return enhance_feature_list

class Cloth_Fusion_Block(nn.Module):
    def __init__(self,in_ch:int,cloth_in_ch:int,heads:int,dim_head:int) -> None:
        super().__init__()

        self.cloth_GroupNorm=nn.GroupNorm(32,cloth_in_ch)
        self.cloth_proj_in=nn.Linear(cloth_in_ch,cloth_in_ch)
        self.cloth_layer_norm=nn.LayerNorm(cloth_in_ch)

        self.in_GroupNorm=nn.GroupNorm(32,in_ch)
        self.in_proj_in=nn.Linear(in_ch,in_ch)
        self.in_layer_norm=nn.LayerNorm(in_ch)

        self.cloth_enhance_attn=Attention(in_ch,cross_attention_dim=cloth_in_ch,heads=heads,dim_head=dim_head,residual_connection=False)  #这里求出的注意力结果没有和残差连接相加
        
        nn.init.zeros_(self.cloth_enhance_attn.to_out[0].weight)
        nn.init.zeros_(self.cloth_enhance_attn.to_out[0].bias)


    def forward(self,in_feature:torch.Tensor,cloth_feature:torch.Tensor):
        
        batch_size,channel,height,width=in_feature.shape

        hidden_states=self.in_GroupNorm(in_feature)
        hidden_states=hidden_states.view(batch_size, channel, height * width).transpose(1, 2).contiguous()
        hidden_states=self.in_proj_in(hidden_states)
        hidden_states=self.in_layer_norm(hidden_states)

        cross_states=self.cloth_GroupNorm(cloth_feature)
        cross_states=cross_states.view(cross_states.shape[0], cross_states.shape[1], cross_states.shape[2] * cross_states.shape[3]).transpose(1, 2).contiguous()
        cross_states=self.cloth_proj_in(cross_states)
        cross_states=self.cloth_layer_norm(cross_states)

        
        cloth_enhance=self.cloth_enhance_attn(hidden_states,cross_states)
        cloth_enhance=cloth_enhance.transpose(-1, -2).reshape(batch_size, channel, height, width).contiguous()


        return cloth_enhance

class Human_Fusion_Block(nn.Module):
    def __init__(self,out_ch:int,human_in_ch:int) -> None:
        super().__init__()

        # 然后补充背景信息
        # human_enhance
        self.human_GroupNorm=nn.GroupNorm(32,human_in_ch)
        self.conv1=nn.Conv2d(human_in_ch,human_in_ch,kernel_size=3,padding=1)
        self.silu=nn.SiLU(inplace=True)
        self.conv2=nn.Conv2d(human_in_ch,out_ch,kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self,human_feature:torch.Tensor):
        human_feature=self.human_GroupNorm(human_feature)
        human_feature=self.conv1(human_feature)
        human_feature=self.silu(human_feature)
        human_feature=self.conv2(human_feature)
        return human_feature


