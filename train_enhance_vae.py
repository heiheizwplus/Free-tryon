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

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
seed_everything(1024)

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from src.dataset.dresscode import DressCodeDataset
from src.dataset.vitonhd import VitonHDDataset
from src.enhance_vae.AutoEncoder_KL import AutoencoderKL as Enhance_AutoencoderKL
from src.utils.vgg_loss import VGGLoss

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as Origin_AutoencoderKL
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

 
    parser.add_argument("--train_batch_size", 
        type=int, default=8, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument("--test_batch_size", 
        type=int, default=8, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--vgg_weight",type=float, default=0.8, help="the weight of vgg loss")

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

    parser.add_argument("--height", type=int, default=512, help="the height of image")
    parser.add_argument("--width", type=int, default=384, help="the width of image")
    parser.add_argument("--test_order", type=str, default="paired", choices=["unpaired", "paired"])

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
                output_dir:str=None,
                dataset:str='vitonhd',
                dresscode_dataroot:str=None,
                vitonhd_dataroot:str=None,
                height: int=512,
                width: int=384,
                vgg_weight=0.8,
                ) -> None:
        super().__init__()

        self.lr=lr
        self.adam_beta1=adam_beta1
        self.adam_beta2=adam_beta2
        self.adam_weight_decay=adam_weight_decay
        self.adam_epsilon=adam_epsilon
        self.warm_up=warm_up
        self.output_dir=output_dir
        self.dataset=dataset
        self.dresscode_dataroot=dresscode_dataroot 
        self.vitonhd_dataroot=vitonhd_dataroot
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.height=height
        self.width=width
        self.vgg_weight=vgg_weight

        self.save_hyperparameters()

        # self.fid=FrechetInceptionDistance(normalize=True)
        self.ssim=StructuralSimilarityIndexMeasure(data_range=(0,1))

        self.tensor_to_image=torchvision.transforms.ToPILImage()


    def configure_model(self) -> None:
        
        self.vae = Enhance_AutoencoderKL.from_config('./train_config/enhance_vae/config.json')
        self.origin_vae = Origin_AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")

        origin_state_dict = self.origin_vae.state_dict()

        for i in list(origin_state_dict.keys()):
            if 'encoder' in i and 'mid_block' not in i and 'down_blocks.3' not in i:
                i_str=i.split('.')[1:]
                i_str='.'.join(i_str)
                i_human='encoder_human.'+i_str
                i_cloth='encoder_cloth.'+i_str
                origin_state_dict[i_human]=origin_state_dict[i]
                origin_state_dict[i_cloth]=origin_state_dict[i]

        self.vae.load_state_dict(origin_state_dict,strict=False)  #

        del self.origin_vae

        self.criterion_vgg=VGGLoss()

        # only test using this code
        # self.vae.load_state_dict(torch.load('/root/Free-tryon/test_checkpoint/enhance-vae-hd/diffusion_pytorch_model.bin'))
     


    def training_step(self,batch,batch_idx) :
        self.save_img_num=0
        mask = batch["agnostic_mask"]
        garment = batch['garment']
        image = batch["image"]
        masked_human_image=image*(1-mask)

        model_pred=self(image, garment=garment,human=masked_human_image,mask=mask)

        loss_mse=F.l1_loss(model_pred,image)  
        loss_vgg=self.vgg_weight*self.criterion_vgg(model_pred,image) 
        loss=loss_mse+loss_vgg

        self.log('train/loss',loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log('train/mse',loss_mse, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log('train/vgg',loss_vgg, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log("global/step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('global/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def forward(self,image,garment,human,mask) :
        return self.vae(image, cloth=garment,human=human,mask=mask,).sample
    

    @torch.no_grad()
    def validation_step(self, batch,batch_idx):
        save_path = os.path.join(self.output_dir, f"img_{self.global_step}")
        os.makedirs(save_path, exist_ok=True)

        mask = batch["agnostic_mask"]
        image = batch['image']
        garment = batch['garment']
        masked_human_image=image*(1-mask)

        model_pred=self(image, garment=garment,human=masked_human_image,mask=mask)
    
        image=torch.clamp(image.detach()/2+0.5,0,1)
        garment=torch.clamp(garment.detach()/2+0.5,0,1)
        masked_human_image=torch.clamp(masked_human_image.detach()/2+0.5,0,1)
        model_pred=torch.clamp(model_pred.detach()/2+0.5,0,1)

        # self.fid.update(model_pred,real=False)
        # self.fid.update(image,real=True)
        # self.log('fid',self.fid,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        self.ssim.update(model_pred,image)
        self.log('ssim',self.ssim,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        
        for gen_image,name,category in zip(model_pred, batch["im_name"],batch['category']):
            os.makedirs(os.path.join(save_path, category),exist_ok=True)
            save_image=self.tensor_to_image(gen_image)
            name=name.split('.')[0]
            name=name+'.png'
            save_image.save(os.path.join(save_path, category, name))


    def configure_optimizers(self):
        params =[i  for i in (list(self.vae.parameters())) if i.requires_grad==True ]
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

    outputlist = ['image', 'category', 'im_name','garment','agnostic_mask']

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
        )

        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(args.height, args.width),
            outputlist=tuple(outputlist),
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='train',
            order='paired',
            radius=5,
            size=(args.height, args.width),
            outputlist=tuple(outputlist),
        )

        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
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
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        vgg_weight=args.vgg_weight,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir,'checkpoint'), 
            save_top_k=-1, monitor="ssim",mode='min',filename="vto-{epoch:03d}-{ssim:.3f}")

    
    logger=WandbLogger(save_dir=args.output_dir,project='free-tryon-vae-512',name='train')
    trainer=pl.Trainer(
        accelerator='gpu',devices=args.local_rank,logger=logger,callbacks=[checkpoint_callback],
        default_root_dir=os.path.join(args.output_dir,'checkpoint'),
        strategy=DeepSpeedStrategy(logging_level=logging.INFO,allgather_bucket_size=5e8,reduce_bucket_size=5e8),
        precision=args.mixed_precision,num_sanity_val_steps=2,gradient_clip_val=args.max_grad_norm,
        check_val_every_n_epoch=10,  accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=200,max_epochs=args.num_train_epochs,
        profiler='simple',benchmark=True,) 
    

    trainer.fit(model,train_dataloader,test_dataloader) 
    # trainer.test(model,test_dataloader)
    wandb.finish()


