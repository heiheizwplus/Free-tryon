# Free-tryon
This is the **official repository** for the [**paper**]() "*Free Try-On: Virtual Try-On 
without Garment-Agnostic Images and Warped Garments*".

Our model checkpoints trained on [VITON-HD](https://github.com/shadow2496/VITON-HD) (half-body) and [Dress Code](https://github.com/aimagelab/dress-code) (full-body) have been released.
* 🤗 [Hugging Face link](https://huggingface.co/zwpro/Free-TryOn) for ***checkpoints*** (stage1, stage, enhance-vae)

## Installation
1. Clone the repository
```sh
git clone https://github.com/heiheizwplus/Free-tryon
```
2. Create a conda environment and install the required packages

```sh
conda create -n vto python==3.10
conda activate vto
pip install -r requirements.txt
```

## Inference
The tryon model T2 inference command:
```sh
python inference.py --dataset [dresscode | vitonhd] --model_image <path> --garment_image <path>  --save_path <path>
```

```
--dataset                           
                            the pre-trained model weights that were obtained by training on dresscode or vitonhd 
--pretrained_model_name_or_path     
                            Path to pretrained model from huggingface.co/models.(default="zwpro/Free-TryOn")
--num_inference_steps       the number of inference steps(default=50)
--num_inference_steps       the guidance scale(default=2.5)
--height                    image height(default=512)
--width                     image width(default=384)
--model_image               path to model image
--garment_image             path to garment image(Optional)
--save_path                 path to save image
--prompts                   the text description of the garment_image(Optional)
--device                    the device to run the model(default=)
```

- Currently, only inference for the Try-On Model T2 is supported, and Enhance VAE is not supported.
- Inference is supported at both 512x384 and 1024x768 resolutions, with 1024x768 only supported using weights trained on the VITON-HD dataset, i.e., `--dataset vitonhd --height 1024 --width 768`.

## Data Preparation
You can directly download our pre-processed data from [BaiduYun](https://pan.baidu.com/s/1jhoe0g7j9dgVN2Ng-ZiJgQ?pwd=tve3).
### Dresscode
1. Download the [DressCode](https://github.com/aimagelab/dress-code) dataset
2. The text annotations for garment images in the DressCode dataset are stored in `data/dresscode/text.json`.
3. For traing or tesing the tryon model T1, you need to download the densepose files from the following
   link: [BaiduYun](https://pan.baidu.com/s/1jhoe0g7j9dgVN2Ng-ZiJgQ?pwd=tve3). Once downloaded,
   please
   extract the densepose files and place them in the dataset folder alongside the corresponding images.

Once the dataset is downloaded, the folder structure should look like this:
```
├── DressCode
|   ├── test_pairs_paired.txt
|   ├── test_pairs_unpaired.txt
|   ├── train_pairs.txt
│   ├── [dresses | lower_body | upper_body]
|   |   ├── test_pairs_paired.txt
|   |   ├── test_pairs_unpaired.txt
|   |   ├── train_pairs.txt
│   │   ├── images
│   │   │   ├── [013563_0.jpg | 013563_1.jpg | 013564_0.jpg | 013564_1.jpg | ...]
│   │   ├── masks
│   │   │   ├── [013563_1.png| 013564_1.png | ...]
│   │   ├── keypoints
│   │   │   ├── [013563_2.json | 013564_2.json | ...]
│   │   ├── label_maps
│   │   │   ├── [013563_4.png | 013564_4.png | ...]
│   │   ├── skeletons
│   │   │   ├── [013563_5.jpg | 013564_5.jpg | ...]
|   |   ├── image-densepose
│   │   │   ├── [013563_0.jpg | 013564_0.jpg | ...]
```

### VITON-HD

1. Download the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset
2. The text annotations for garment images in the VITON-HD dataset are stored in `data/vitonhd/text.json`.
3. For traing or tesing the tryon model T1, you need to download the densepose files from the following
   link: [BaiduYun](https://pan.baidu.com/s/1jhoe0g7j9dgVN2Ng-ZiJgQ?pwd=tve3). Once downloaded,
   please
   extract the densepose files and place them in the dataset folder alongside the corresponding images.

Once the dataset is downloaded, the folder structure should look like this:

```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train | test]
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── openpose_json
│   │   │   ├── [000006_00_keypoints.json | 000008_00_keypoints.json | ...]
│   │   ├── agnostic-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── image-densepose
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
```

## Model Training
### Enhance VAE Training
To train the Enhance VAE, run the following command:
```
python train_enhance_vae.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> 
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --output_dir <str>             output directory
```


### Training the Try-On Model T1 [Optional]
Training a new Try-On Model T1 is not required; you can directly use any existing pre-trained try-on model without additional training.

```
python train_stage1.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --garment_text_path ['data/dresscode/text.json', './data/vitonhd/text.json']
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --garment_text_path <str>       the file path of garment text description,options: ['data/dresscode/text.json', './data/vitonhd/text.json']
    --output_dir <str>             output directory
    ...

```

### Training the Try-On Model T2

#### Additional Data Preparation
Training the Try-On Model T2 requires using the Try-On Model T1 or any existing pre-trained try-on model to generate the corresponding training data.

##### Using Existing Pre-Trained Try-On Models
1. If you use existing pre-trained try-on models to generate the training data required for the Try-On Model T2, you only need to generate the corresponding try-on images based on the unpaired matches in the training set of the respective dataset and place them in the appropriate dataset directory.
2. For the VITON-HD dataset, we follow the provided `train_pairs.txt`. For the DressCode dataset, we construct the corresponding data pairs in `./data/dresscode/[dresses | lower_body | upper_body]/train_unpaired_pairs.txt`.

##### Using the Try-On Model T1
1. For the VITON-HD dataset, we follow the provided `train_pairs.txt`. For the DressCode dataset, we construct the corresponding data pairs in `./data/dresscode/[dresses | lower_body | upper_body]/train_unpaired_pairs.txt`. Please place these files in the corresponding dataset directory.
2. Run the following command to use the Try-On Model T1 to generate the training data required for the Try-On Model T2.

```
python test_stage1.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path>  --vitonhd_dataroot <path>  --output_dir <path>  --garment_text_path <path> ['data/dresscode/text.json', './data/vitonhd/text.json']  --prepare_data_for_t2 True
```

3. Please place the generated try-on images in the appropriate directory within the corresponding dataset.


After completing the additional data preparation, the structure of the corresponding data folders is as follows:

Dresscode Dataset
```
├── DressCode
|   ├── test_pairs_paired.txt
|   ├── test_pairs_unpaired.txt
|   ├── train_pairs.txt
│   ├── [dresses | lower_body | upper_body]
|   |   ├── test_pairs_paired.txt
|   |   ├── test_pairs_unpaired.txt
|   |   ├── train_pairs.txt
|   |   ├── train_pairs_unpaired.txt
│   │   ├── ...
|   |   ├── tryon_image
│   │   │   ├── [013563_0.png | 013564_0.png | ...]
│   │   ├── ...
```

VITON-HD Dataset
```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train]
│   │   ├── ...
│   │   ├── tryon_image
│   │   │   ├── [000006_00.png | 000008_00.png | ...]
```


#### Training the Try-On Model T2
To train the tryon model T2, run the following command:
```
python train_stage2.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --garment_text_path 'data/dresscode/text.json', './data/vitonhd/text.json']
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --garment_text_path <str>       the file path of garment text description,options: ['data/dresscode/text.json', './data/vitonhd/text.json']
    --output_dir <str>             output directory
    ...

```

## Model Testing
#### Testing the Try-On Model T1
You can run the following command to test the Try-On Model T1, which will by default download the pre-trained weights from Hugging Face.
```
python test_stage1.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --garment_text_path ['data/dresscode/text.json', './data/vitonhd/text.json'] --test_order ["unpaired", "paired"]  --vae_type ["enhance", "origin"]
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --garment_text_path <str>       the file path of garment text description,options: ['data/dresscode/text.json', './data/vitonhd/text.json']
    --output_dir <str>             output directory
    --test_order                   test setting, options: ['paired', 'unpaired']
    --vae_type                     options: ["enhance", "origin"],If 'enhance' use the enhanced vae, if 'origin' use the origin stable diffusion vae"
    ...
```


#### Testing the Try-On Model T2
You can run the following command to test the Try-On Model T2, which will by default download the pre-trained weights from Hugging Face.

```
python test_stage2.py --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --output_dir <path> --garment_text_path ['data/dresscode/text.json', './data/vitonhd/text.json'] --test_order ["unpaired", "paired"]  --vae_type ["enhance", "origin"]
```

```
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --garment_text_path <str>       the file path of garment text description,options: ['data/dresscode/text.json', './data/vitonhd/text.json']
    --output_dir <str>             output directory
    --test_order                   test setting, options: ['paired', 'unpaired']
    --vae_type                     options: ["enhance", "origin"],If 'enhance' use the enhanced vae, if 'origin' use the origin stable diffusion vae"
    ...
```

#### Metrics Calculation
After you have completed testing the Try-On Model T1 or T2, you can compute the metrics by running the following command:command:

```
python val_metrics.py --gen_folder <path> --dataset [dresscode | vitonhd] --dresscode_dataroot <path> --vitonhd_dataroot <path> --test_order [paired | unpaired] 
```

```
    --gen_folder <str>             Path to the generated images folder.
    --dataset <str>                dataset to use, options: ['dresscode', 'vitonhd']
    --dresscode_dataroot <str>     data root of dresscode dataset (required when dataset=dresscode)
    --vitonhd_dataroot <str>       data root of vitonhd dataset (required when dataset=vitonhd)
    --test_order <str>             test setting, options: ['paired', 'unpaired']
    --category <str>               category to test, options: ['all', 'lower_body', 'upper_body', 'dresses'] (default=all)
    --batch_size                   batch size (default=32)
    --workers                      number of workers (default=8)
    --height                       Height of the generated images
    --width                        Width of the generated images
```





