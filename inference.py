import argparse
import logging
log = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
import os
import torch
import torch.nn.functional as F
import torchvision

from diffusers import DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
seed_everything(1024)

from src.dataset.dresscode import DressCodeDataset
from src.dataset.vitonhd import VitonHDDataset
from src.enhance_vae.AutoEncoder_KL import AutoencoderKL as Enhance_AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as Origin_AutoencoderKL
from src.human_unet.unet_2d_condition import UNet2DConditionModel as Human_Unet
from src.garment_encoder.feature_extractor import Garment_Encoder
from src.garment_unet.unet_2d_condition import UNet2DConditionModel as Garment_Unet  #
from src.vto_pipelines.tryon_pipe_stage2 import StableDiffusionTryOnPipeline

from PIL import Image




def parse_args():
    parser = argparse.ArgumentParser(description="VTO inference script.")
    parser.add_argument("--dataset", type=str,default='vitonhd', required=True, choices=["dresscode", "vitonhd"], help="dataset to use")

    parser.add_argument("--pretrained_model_name_or_path",type=str,default="zwpro/Free-TryOn",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--num_inference_steps", type=int, default=50, help="the number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="the guidance scale")

    parser.add_argument("--height", type=int, default=512, help="image height")
    parser.add_argument("--width", type=int, default=384, help="image width")

    parser.add_argument("--model_image",type=str,default=None,help="path to model image")
    parser.add_argument("--garment_image",type=str,default=None,help="path to garment image")
    parser.add_argument("--save_path",type=str,default=None,help="path to save image")
    parser.add_argument("--prompts",type=str,default=None,help="the text description of the garment_image")
    parser.add_argument("--device",type=int,default=0,help="the device to run the model")

    args = parser.parse_args()


    return args

class VTO:
    def __init__(self,                 
                pretrained_model_name_or_path:str=None,
                guidance_scale:float=2.5,
                dataset:str='vitonhd',
                height: int=512,
                width: int=384,
                num_inference_steps: int=50,
                device: int=0,
                ) -> None:
        """
            Initialize the Inference class with the provided parameters.
            Parameters:
                - pretrained_model_name_or_path (str): Name or path of the pretrained model.
                - guidance_scale (float): Scale for guidance.
                - dataset (str): Dataset name.
                - height (int): Height of the image.
                - width (int): Width of the image.
                - num_inference_steps (int): Number of inference steps.
                - device (int): Device number for CUDA.
            Returns:
                - None
        """
        super().__init__()

        self.guidance_scale=guidance_scale
        self.dataset=dataset
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.height=height
        self.width=width
        self.num_inference_steps=num_inference_steps
        self.device=torch.device(f"cuda:{device}")

        self.tensor_to_image=torchvision.transforms.ToPILImage()
        self.image_to_tensor=torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.height, self.width)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path,subfolder='scheduler')
        self.val_scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_name_or_path,subfolder='scheduler')
        self.val_scheduler.set_timesteps(50)
        
        self.vae = Origin_AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")
        
         #Human UNet
        if self.height == 512:
            self.human_unet = Human_Unet.from_pretrained(self.pretrained_model_name_or_path, subfolder=f"stage2/{self.dataset}/human_unet")
            #Garment UNet
            self.garment_unet=Garment_Unet.from_pretrained(self.pretrained_model_name_or_path, subfolder=f"stage2/{self.dataset}/garment_unet")
            #Garment Encoder
            self.garment_encoder=Garment_Encoder.from_pretrained(self.pretrained_model_name_or_path, subfolder=f"stage2/{self.dataset}/garment_encoder") 
        else:
            self.human_unet = Human_Unet.from_pretrained(self.pretrained_model_name_or_path, subfolder=f"stage2/{self.dataset}-1024/human_unet")
            #Garment UNet
            self.garment_unet=Garment_Unet.from_pretrained(self.pretrained_model_name_or_path, subfolder=f"stage2/{self.dataset}-1024/garment_unet")
            #Garment Encoder
            self.garment_encoder=Garment_Encoder.from_pretrained(self.pretrained_model_name_or_path, subfolder=f"stage2/{self.dataset}-1024/garment_encoder") 
           
        # Freeze vae and text_encoder
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)
        self.garment_unet.eval().requires_grad_(False)
        self.human_unet.eval().requires_grad_(False)
        self.garment_encoder.eval().requires_grad_(False)
        
    
        self.val_pipe = StableDiffusionTryOnPipeline(
                                vae=self.vae,
                                text_encoder=self.text_encoder,
                                human_unet=self.human_unet,
                                garment_unet=self.garment_unet,
                                tokenizer=self.tokenizer,
                                scheduler=self.val_scheduler,
                                garment_encoder=self.garment_encoder,
                                is_enhance_vae=False,
                            ).to(self.device)
    
    @torch.no_grad()
    def inference(self, model_img:str=None,garment_image:str=None,save_path:str=None,prompts:str=None,):
        """
        Perform inference using the provided model image, garment image or prompts, and save the generated image to the specified path.
        Parameters:
        - model_img (str): Path to the model image.
        - garment_image (str): Path to the garment image.
        - save_path (str): Path to save the generated image.
        - prompts (str): Prompts for the garment_image.
        Returns:
        - None
        """
        assert model_img is not None, "model img must be provided"
        assert garment_image is not None or prompts is not None, "either prompts or garment image must be provided"

        model_img=Image.open(model_img).convert('RGB')
        model_img=self.image_to_tensor(model_img).unsqueeze(dim=0).to(self.device)

        if garment_image is not None:
            garment_image=Image.open(garment_image).convert('RGB')
            garment_image=self.image_to_tensor(garment_image).unsqueeze(dim=0).to(self.device)

        if prompts is None:
            prompts=''

        gen_image=self.val_pipe(
            prompt=[prompts],
            human_image=model_img,
            mask_image=None,
            garment=garment_image,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=self.num_inference_steps,
            is_text_only=True if garment_image is None else False,
            negative_prompt=["incorrect clothing,unnatural clothing folds,residual clothing artifacts,confusing clothing patterns\
                            abnormal clothing confusion,distorted clothing,unrealistic fabric textures,poor garment rendering,\
                            misplaced clothing details,fused clothing elements,blurry clothing edges,\
                            blurry, cracks on skins, poor shirts, poor pants, strange holes, bad legs, missing legs, \
                            bad arms, missing arms, bad anatomy, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, \
                            extra crus,fused crus, worst feet, three feet, fused feet, fused thigh, three thighs, fused thigh, extra thigh, worst thigh, \
                            missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, \
                             cartoon, cg, 3d, unreal, animate"]
            ).images

        gen_image=self.tensor_to_image(gen_image.squeeze(dim=0))

        if save_path is None:
            gen_image.save('./result.png')
        else:
            gen_image.save(os.path.join(save_path,'result.png'))
        


if __name__=='__main__':
    args = parse_args()

    model=VTO(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                guidance_scale=args.guidance_scale,
                dataset=args.dataset,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                device=args.device,
                )

    model.inference(model_img=args.model_image,garment_image=args.garment_image,save_path=args.save_path,prompts=args.prompts)

