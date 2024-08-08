# File heavily based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
# We modified the file to adapt the inpainting pipeline to the virtual try-on task

import inspect
from typing import Callable, List, Optional, Union
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import PIL
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL  
from src.enhance_vae.AutoEncoder_KL import AutoencoderKL as Enhance_VAE
from src.human_unet.unet_2d_condition import UNet2DConditionModel as HumanUNet
from src.garment_unet.unet_2d_condition import UNet2DConditionModel as GarmentUNet
from src.garment_encoder.feature_extractor import Garment_Encoder
from src.pose_encoder.pose_encoder import ControlNetConditioningEmbedding as PoseEncoder
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL 

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer




class StableDiffusionTryOnPipeline(DiffusionPipeline):
    r"""
    Pipeline for text and posemap -guided image inpainting using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker"]

    def __init__(
            self,
            vae: Union[AutoencoderKL,Enhance_VAE],  # vae or enhance_vae
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            human_unet: HumanUNet,
            garment_unet:GarmentUNet,
            garment_encoder:Garment_Encoder,
            pose_encoder:PoseEncoder,
            scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker: bool = False,
            is_enhance_vae:bool=False,
    ):
        """
            Initializes the TryonPipeTime object.
            Args:
                vae (Union[AutoencoderKL, Enhance_VAE]): The VAE or Enhance VAE model.
                text_encoder (CLIPTextModel): The CLIP text encoder model.
                tokenizer (CLIPTokenizer): The CLIP tokenizer.
                human_unet (HumanUNet): The Human U-Net model.
                garment_unet (GarmentUNet): The Garment U-Net model.
                garment_encoder (Garment_Encoder): The Garment Encoder model.
                scheduler (Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]): The scheduler model.
                safety_checker (optional): The safety checker model. Defaults to None.
                feature_extractor (optional): The feature extractor model. Defaults to None.
                requires_safety_checker (bool, optional): Flag indicating if safety checker is required. Defaults to False.
                is_enhance_vae (bool, optional): Flag indicating if Enhance VAE is used. Defaults to False.
        """
        
        super().__init__()

        self.is_enhance_vae = is_enhance_vae

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "skip_prk_steps") and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate("skip_prk_steps not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(human_unet.config, "_diffusers_version") and version.parse(
            version.parse(human_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(human_unet.config, "sample_size") and human_unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(self.human_unet.config)
            new_config["sample_size"] = 64
            self.human_unet._internal_dict = FrozenDict(new_config)
            self.garment_unet._internal_dict= FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            human_unet=human_unet,
            garment_unet=garment_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            garment_encoder=garment_encoder,
            pose_encoder=pose_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae,self.garment_unet]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

  

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    
    def enhance_vae_decode_latents(self, latents, garment_enhance_list,human_enhance_list,mask):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents,garment_enhance_list,human_enhance_list,mask).sample
        image = (image / 2 + 0.5).clamp(0, 1).detach()
        return image
    
    def decode_latents(self, latents,):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).detach()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    
    def enhance_vae_encode_latents(self, mask, human_image,garment, dtype, device,):
        """
        Refines the latent representations of human image and garment using Enhance VAE encoder.

        Args:
        mask (torch.Tensor): Binary mask tensor.
        human_image (torch.Tensor): Human image tensor.
        garment (torch.Tensor): garment image tensor.
        dtype (torch.dtype): Data type for the tensors.
        device (torch.device): Device to perform computations on.

        Returns:
        tuple: A tuple containing the refined mask tensor, refined human latent representations, and refined garment latent representations.
        """
        
        mask = mask.to(device=device, dtype=dtype)
        garment = garment.to(device=device, dtype=dtype)
        human_image = human_image.to(device=device, dtype=dtype)
        human_image = (1-mask)*human_image
        human_enhance_list = self.vae.human_encode(human_image)
        garment_enhance_list = self.vae.garment_encode(garment)
        human_enhance_list = [ i.to(device=device, dtype=dtype) for i in human_enhance_list]
        garment_enhance_list = [ i.to(device=device, dtype=dtype) for i in garment_enhance_list]
        return mask, human_enhance_list,garment_enhance_list

    @torch.no_grad()
    def __call__(self,
            human_image: Union[torch.FloatTensor,] = None,
            garment:Optional[torch.FloatTensor]=None,
            pose:Optional[torch.FloatTensor]=None,
            mask_image: Optional[torch.FloatTensor] =None,
            prompt: Union[str, List[str]] = None,
            is_text_only: bool = False,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 2.5,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            **kwargs
    ):
        r"""
        Perform the try-on pipeline.
        Args:
            human_image (torch.FloatTensor, optional): The human image. Defaults to None.
            garment (torch.FloatTensor, optional): The garment image. Defaults to None.
            mask_image (torch.FloatTensor, optional): The mask image. Defaults to None.
            prompt (Union[str, List[str]], optional): The garmenting text description. Defaults to None.
            is_text_only (bool, optional): Whether to use only the garmenting text description. Defaults to False.
            height (int, optional): The height of the output image. Defaults to None.
            width (int, optional): The width of the output image. Defaults to None.
            num_inference_steps (int, optional): The number of inference steps. Defaults to 50.
            guidance_scale (float, optional): The scale for guidance. Defaults to 2.5.
            negative_prompt (Union[str, List[str]], optional): The negative image prompt. Defaults to None.
            num_images_per_prompt (int, optional): The number of images per prompt. Defaults to 1.
            eta (float, optional): The eta value. Defaults to 0.0.
            prompt_embeds (torch.FloatTensor, optional): The embedded prompt. Defaults to None.
            negative_prompt_embeds (torch.FloatTensor, optional): The embedded negative prompt. Defaults to None.
            generator (Union[torch.Generator, List[torch.Generator]], optional): The generator. Defaults to None.
            latents (torch.FloatTensor, optional): The latent variables. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to True.
            callback (Callable[[int, int, torch.FloatTensor], None], optional): The callback function. Defaults to None.
            callback_steps (int, optional): The number of steps between callbacks. Defaults to 1.
            **kwargs: Additional keyword arguments.
            Returns:
                Union[StableDiffusionPipelineOutput, Tuple[torch.Tensor, None]]: The output of the try-on pipeline.
    """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if human_image is None or garment is None:
            raise ValueError("`human_image` or `garment` input cannot be undefined.")
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4.1 Prepare Enhance VAE
        if self.is_enhance_vae and mask_image is not None and garment is not None:
            # mask, masked_image = prepare_mask_and_masked_image(image, mask_image,height,width)
            mask,human_enhance_list,garment_enhance_list=self.enhance_vae_encode_latents(
                mask_image,human_image,garment,prompt_embeds.dtype,device,
            )
        else:
            human_enhance_list = None
            garment_enhance_list = None
        
        mask,masked_human_image=prepare_mask_and_masked_image(human_image,mask_image,height=height,width=width)
        human_latents=self.vae.encode(masked_human_image.to(human_image.dtype)).latent_dist.sample(generator=generator)*self.vae.config.scaling_factor
        mask=torch.nn.functional.interpolate(mask,size=(human_latents.shape[2],human_latents.shape[3]))
        if do_classifier_free_guidance:
            human_latents=torch.cat([human_latents]*2)
            inpaint_mask=torch.cat([mask]*2)
        human_latents=torch.cat([inpaint_mask,human_latents],dim=1)
            

        #4.2 prepare garment latent and garment feature
        if is_text_only :   #无服装约束的情况
            garment = torch.zeros_like(human_image)
        garment_latent = self.vae.encode(garment).latent_dist.sample(generator=generator)*self.vae.config.scaling_factor
        garment_feature = self.garment_encoder(garment)

        #4.3 prepare pose feature
        pose_feature = self.pose_encoder(pose)
        #for classifier free guidance
        if do_classifier_free_guidance:
            pose_feature = torch.cat([pose_feature]*2)
        

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


                garment_input=torch.cat([latents,garment_latent],dim=1)
                latent_model_input = torch.cat([latent_model_input, human_latents], dim=1)
                    
                garment_hidden_states=self.garment_unet(sample=garment_input,timestep=t,
                                                     encoder_hidden_states=prompt_embeds[garment_input.shape[0]:],
                                                     condition_latent=garment_feature)
                self_attn_states=garment_hidden_states.self_attn_states
                
                #for classifier free guidance
                if do_classifier_free_guidance:
                    if not is_text_only:
                        self_attn_states=[[torch.cat([torch.zeros_like(j),j]) for j in i] for i in self_attn_states] 
                    else:
                        self_attn_states=[[torch.cat([torch.zeros_like(j),torch.zeros_like(j)]) for j in i] for i in self_attn_states] 
                else:
                    if not is_text_only:
                        pass
                    else:
                        self_attn_states=[[torch.zeros_like(j) for j in i] for i in self_attn_states]   
                # predict the noise residual
                noise_pred = self.human_unet(latent_model_input, t, 
                                             encoder_hidden_states=prompt_embeds,
                                             self_attn_states=self_attn_states,
                                             condition_latens=pose_feature).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.to(self.vae.dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 11. Post-processing
        if self.is_enhance_vae and mask_image is not None and garment is not None:
            image = self.enhance_vae_decode_latents(latents, garment_enhance_list,human_enhance_list,mask)
        else:
            image = self.decode_latents(latents)


        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
