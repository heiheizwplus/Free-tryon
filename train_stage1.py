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
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
seed_everything(1024)

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from src.dataset.dresscode import DressCodeDataset
from src.dataset.vitonhd import VitonHDDataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as Origin_AutoencoderKL
from src.human_unet.unet_2d_condition import UNet2DConditionModel as Human_Unet
from src.garment_encoder.feature_extractor import Garment_Encoder
from src.garment_unet.unet_2d_condition import UNet2DConditionModel as Garment_Unet  #
from src.vto_pipelines.tryon_pipe_stage1 import StableDiffusionTryOnPipeline
from src.pose_encoder.pose_encoder import ControlNetConditioningEmbedding as PoseEncoder

import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description="VTO training script.")
    parser.add_argument("--dataset", type=str,default='vitonhd', required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot',default='/opt/data/private/HR-VITON', type=str, help='VitonHD dataroot')
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models. (Stable Diffusion V2.1 inpainting) ",
    )
    
    parser.add_argument("--output_dir",
        type=str,
        required=True,
        default='/root/vto_output',
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--garment_text_path", type=str, help="the file path of garment text description")

 
    parser.add_argument("--train_batch_size", 
        type=int, default=8, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument("--test_batch_size", 
        type=int, default=8, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=200, help="Total number of training epochs to perform.")

    parser.add_argument("--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--learning_rate",
        type=float,
        default=4e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument("--lr_warmup_steps", 
                        type=int, default=2000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--mixed_precision", 
        type=str,
        default='16-mixed',
        choices=["no", "16-mixed", "bf16-mixed"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use in the dataloaders.")
    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="Number of workers to use in the test dataloaders.")

    parser.add_argument("--num_inference_steps", type=int, default=50, help="the number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="the guidance scale")
    parser.add_argument("--height", type=int, default=512, help="the height of image")
    parser.add_argument("--width", type=int, default=384, help="the width of image")
    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--uncond_fraction", type=float, default=0.2, help="Fraction of unconditioned training samples")
    args = parser.parse_args()


    return args

class VTO(pl.LightningModule):
    def __init__(self, 
                pretrained_model_name_or_path:str=None,
                lr:float=1e-4,
                adam_beta1:float=0.9,
                adam_beta2:float=0.999,
                adam_weight_decay:float=1e-2,
                adam_epsilon:float=1e-08,
                warm_up:float=2000,
                uncond_fraction:float=0.2,
                output_dir:str=None,
                guidance_scale:float=2.5,
                dataset:str='vitonhd',
                dresscode_dataroot:str=None,
                vitonhd_dataroot:str=None,
                height: int=512,
                width: int=384,
                num_inference_steps: int=50
                ) -> None:
        
        """
        Initializes the model with the specified hyperparameters.
            Parameters:
                pretrained_model_name_or_path (str, optional): Path to the pretrained model.
                lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
                adam_beta1 (float, optional): Beta1 parameter for Adam optimizer. Default is 0.9.
                adam_beta2 (float, optional): Beta2 parameter for Adam optimizer. Default is 0.999.
                adam_weight_decay (float, optional): Weight decay for Adam optimizer. Default is 1e-2.
                adam_epsilon (float, optional): Epsilon parameter for Adam optimizer. Default is 1e-08.
                warm_up (float, optional): Number of warm-up steps. Default is 2000.
                uncond_fraction (float, optional): Fraction of unconditioned samples. Default is 0.2.
                output_dir (str, optional): Directory to save output files.
                guidance_scale (float, optional): Scale for guidance. Default is 2.5.
                dataset (str, optional): Name of the dataset to use. Default is 'vitonhd'.
                dresscode_dataroot (str, optional): Root directory for dress code dataset.
                vitonhd_dataroot (str, optional): Root directory for VITON-HD dataset.
                height (int, optional): Height of the input images. Default is 512.
                width (int, optional): Width of the input images. Default is 384.
                num_inference_steps (int, optional): Number of inference steps. Default is 50.
            Returns:
                None
        """
        super().__init__()

        self.lr=lr
        self.adam_beta1=adam_beta1
        self.adam_beta2=adam_beta2
        self.adam_weight_decay=adam_weight_decay
        self.adam_epsilon=adam_epsilon
        self.warm_up=warm_up
        self.uncond_fraction=uncond_fraction
        self.output_dir=output_dir
        self.guidance_scale=guidance_scale
        self.dataset=dataset
        self.dresscode_dataroot=dresscode_dataroot 
        self.vitonhd_dataroot=vitonhd_dataroot
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.height=height
        self.width=width
        self.num_inference_steps=num_inference_steps

        self.save_hyperparameters()

        # self.fid=FrechetInceptionDistance(normalize=True)
        self.ssim=StructuralSimilarityIndexMeasure(data_range=(0,1))

        self.tensor_to_image=torchvision.transforms.ToPILImage()


    def configure_model(self) -> None:
        
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path,subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path,subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path,subfolder="scheduler")
        self.val_scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_name_or_path,subfolder="scheduler")
        self.val_scheduler.set_timesteps(50)
        
        self.vae = Origin_AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")

        #Human UNet
        self.human_unet = Human_Unet.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet")
        #Garment UNet
        self.garment_unet=Garment_Unet.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet")

        #garment_unet ：Noise-laten + garment laten
        garment_new_channels = 4+4
        with torch.no_grad():
            # Replace the first conv layer of the unet with a new one with the correct number of input channels
            conv_new = torch.nn.Conv2d(
                in_channels=garment_new_channels,
                out_channels=self.garment_unet.conv_in.out_channels,
                kernel_size=3,
                padding=1,
            )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :4] = self.garment_unet.conv_in.weight.data[:,:4]  # Copy weights from old conv layer
            conv_new.weight.data[:, 4:] = self.garment_unet.conv_in.weight.data[:,5:]  # Copy weights from old conv layer
            conv_new.bias.data = self.garment_unet.conv_in.bias.data  # Copy bias from old conv layer

            self.garment_unet.conv_in = conv_new  # replace conv layer in unet
            self.garment_unet.config['in_channels'] = garment_new_channels  # update config

        #Garment Encoder
        self.garment_encoder=Garment_Encoder.from_config('./train_config/garment_encoder/config.json')  #os.getcwd(),
        vae_state_dict=list(self.vae.state_dict().keys())
        for i in self.garment_encoder.state_dict().keys():
            if i in vae_state_dict:
                self.garment_encoder.state_dict()[i].data=self.vae.state_dict()[i].data
        
        #Pose Encoder
        self.pose_encoder=PoseEncoder(conditioning_embedding_channels=320)

        # Freeze vae and text_encoder
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)
       
        # torch.compile
        self.vae=torch.compile(self.vae)
        self.text_encoder=torch.compile(self.text_encoder)
        self.human_unet=torch.compile(self.human_unet)  
        self.garment_unet=torch.compile(self.garment_unet) 
        self.garment_encoder=torch.compile(self.garment_encoder)
        self.pose_encoder=torch.compile(self.pose_encoder)

    
        self.val_pipe = StableDiffusionTryOnPipeline(
                                vae=self.vae,
                                text_encoder=self.text_encoder,
                                human_unet=self.human_unet,
                                garment_unet=self.garment_unet,
                                tokenizer=self.tokenizer,
                                scheduler=self.val_scheduler,
                                garment_encoder=self.garment_encoder,
                                pose_encoder = self.pose_encoder,
                                is_enhance_vae=False
                            ).to(self.device)


    def training_step(self,batch,batch_idx) :
        self.save_img_num=0
        mask = batch.get("agnostic_mask")
        masked_image_latents = self.img_to_laten((1-mask)*batch["image"]) * self.vae.config.scaling_factor
        mask = torch.nn.functional.interpolate(mask,(int(mask.shape[2]/8),int(mask.shape[3]/8)))
        latents = self.img_to_laten(batch["image"])* self.vae.config.scaling_factor

        noise = torch.randn_like(latents).to(self.device)
        target = noise
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(self.device)
        
        #garment text description
        uncond_mask_text = torch.rand(bsz, device=self.device) < self.uncond_fraction           
        text = batch["captions"]
        text = [t if not uncond_mask_text[i] else "" for i, t in enumerate(text)]
        encoder_hidden_states=self.compute_text_embedding(text)

        #garment image
        garment = batch['garment']
        garment_latents = self.img_to_laten(garment) * self.vae.config.scaling_factor
        uncond_mask_self_attn = torch.rand(bsz, device=self.device) < self.uncond_fraction
        uncond_mask_self_attn = uncond_mask_self_attn.view(-1, 1, 1)

        garment_feature=self.garment_encoder(garment)
        garment_input=torch.cat([noisy_latents,garment_latents],dim=1)
        garment_hidden_states=self.garment_unet(sample=garment_input,timestep=timesteps,
                                            encoder_hidden_states=encoder_hidden_states,condition_latent=garment_feature) 
        self_attn_states=garment_hidden_states.self_attn_states

        #random drop garment feature
        for i in range(len(self_attn_states)):
            for j in range(len(self_attn_states[i])):
                self_attn_states[i][j] = torch.where(
                    uncond_mask_self_attn, torch.zeros_like(self_attn_states[i][j]), self_attn_states[i][j])

        # pose feature
        pose_feature=self.pose_encoder(batch['densepose'])

        unet_input=torch.cat([noisy_latents,mask,masked_image_latents],dim=1)
        model_pred=self(unet_input,timesteps,encoder_hidden_states,self_attn_states,pose_feature)

        loss=F.mse_loss(model_pred,target)
        self.log('train/loss',loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log("global/step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('global/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def forward(self, unet_input, timesteps, encoder_hidden_states,self_attn_states,pose_condition) :
        return self.human_unet(unet_input,timesteps,encoder_hidden_states,self_attn_states=self_attn_states,condition_latens=pose_condition).sample
    

    @torch.no_grad()
    def validation_step(self, batch,batch_idx):
        save_path = os.path.join(self.output_dir, f"img_{self.global_step}")
        os.makedirs(save_path, exist_ok=True)

        mask_img = batch["agnostic_mask"]
        model_img = batch['image']
        garment = batch['garment']
        category = batch['category']
        prompts = batch["captions"]
        pose = batch["densepose"]
        
        generated_images = self.val_pipe(
            prompt=prompts,
            human_image=model_img,
            mask_image=mask_img,
            garment=garment,
            pose=pose,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=self.num_inference_steps,
            negative_prompt=["incorrect clothing,unnatural clothing folds,residual clothing artifacts,confusing clothing patterns\
                            abnormal clothing confusion,distorted clothing,unrealistic fabric textures,poor garment rendering,\
                            misplaced clothing details,fused clothing elements,blurry clothing edges,\
                            blurry, cracks on skins, poor shirts, poor pants, strange holes, bad legs, missing legs, \
                            bad arms, missing arms, bad anatomy, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, \
                            extra crus,fused crus, worst feet, three feet, fused feet, fused thigh, three thighs, fused thigh, extra thigh, worst thigh, \
                            missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, \
                             cartoon, cg, 3d, unreal, animate" for i in range(len(prompts))]
            ).images
        
        model_img=torch.clamp(model_img.detach()/2+0.5,0,1)
        garment=torch.clamp(garment.detach()/2+0.5,0,1)

        # self.fid.update(generated_images,real=False)
        # self.fid.update(model_img,real=True)
        # self.log('fid',self.fid,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        self.ssim.update(generated_images,model_img)
        self.log('ssim',self.ssim,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        
        for gen_image,name,category in zip(generated_images, batch["im_name"],batch['category']):
            os.makedirs(os.path.join(save_path, category),exist_ok=True)
            save_image=self.tensor_to_image(gen_image)
            name=name.split('.')[0]
            name=name+'.png'
            save_image.save(os.path.join(save_path, category, name))

    @torch.no_grad()
    def compute_text_embedding(self,text):
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(self.device)
        encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
        return encoder_hidden_states

    @torch.no_grad()
    def img_to_laten(self,img):
        latent=self.vae.encode(img).latent_dist.sample()
        return latent


    def configure_optimizers(self):
        params =[i  for i in (list(self.garment_unet.parameters()) +list(self.garment_encoder.parameters())+list(self.pose_encoder.parameters())
                              +list(self.human_unet.parameters())) if i.requires_grad==True ]
        optim = AdamW(params, lr=self.lr,betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,eps=self.adam_epsilon,)
        lambda_lr=lambda step: max(((self.global_step)/self.warm_up),5e-3) if (self.global_step)< self.warm_up else  1.0
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda_lr)
        return {'optimizer':optim,'lr_scheduler':{"scheduler":lr_scheduler,'monitor':'ssim','interval':'step','frequency':1}}



if __name__=='__main__':
    args = parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    outputlist = ['image', 'captions', 'category', 'im_name','garment','densepose','agnostic_mask']

    # Define datasets and dataloaders.
    if args.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='train',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(args.height, args.width),
            outputlist=tuple(outputlist),
            caption_filename=args.garment_text_path,
        )

        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(args.height, args.width),
            outputlist=tuple(outputlist),
            caption_filename=args.garment_text_path,
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='train',
            order='paired',
            caption_filename=args.garment_text_path,
            radius=5,
            size=(args.height, args.width),
            outputlist=tuple(outputlist),
        )

        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
            caption_filename=args.garment_text_path,
            radius=5,
            size=(args.height, args.width),
            outputlist=tuple(outputlist)
        )
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
        pin_memory=True,
    )

    model=VTO(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        dataset=args.dataset,
        dresscode_dataroot=args.dresscode_dataroot,
        vitonhd_dataroot=args.vitonhd_dataroot,
        lr=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        warm_up=args.lr_warmup_steps,
        uncond_fraction=args.uncond_fraction,
        output_dir=args.output_dir,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir,'checkpoint'), 
            save_top_k=-1, monitor="ssim",mode='min',filename="vto-{epoch:03d}-{ssim:.3f}")

    
    logger=WandbLogger(save_dir=args.output_dir,project='free-tryon-stage-2-512',name='train')
    trainer=pl.Trainer(
        accelerator='gpu',devices=args.local_rank,logger=logger,callbacks=[checkpoint_callback],
        default_root_dir=os.path.join(args.output_dir,'checkpoint'),
        strategy=DeepSpeedStrategy(logging_level=logging.INFO,allgather_bucket_size=5e8,reduce_bucket_size=5e8),
        precision=args.mixed_precision,num_sanity_val_steps=2,gradient_clip_val=args.max_grad_norm,
        check_val_every_n_epoch=10,  accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=200,max_epochs=args.num_train_epochs,
        profiler='simple',benchmark=True,) 
    

    trainer.fit(model,train_dataloader,test_dataloader) 
    wandb.finish()


