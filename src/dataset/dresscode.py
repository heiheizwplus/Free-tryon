# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, Literal

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from numpy.linalg import lstsq

from src.utils.labelmap import label_map
from src.utils.posemap import kpoint_to_heatmap


class DressCodeDataset(data.Dataset):
    def __init__(self,
                 dataroot_path: str,
                 phase: Literal['train', 'test'],
                 radius=5,
                 caption_filename: str = 'dresscode.json',
                 order: Literal['paired', 'unpaired'] = 'paired',
                 outputlist: Tuple[str] = ('im_name', 'garment', 'image', 'shape', 
                                           'tryon_image','densepose','category',
                                           'captions','agnostic_mask'),
                 category: Tuple[str] = ('dresses', 'upper_body', 'lower_body'),
                 size: Tuple[int, int] = (512, 384),
                 ):

        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []

        possible_outputs = ['im_name', 'garment', 'image', 'shape', 
                            'tryon_image','densepose','category',
                            'captions','agnostic_mask']

        assert all(x in possible_outputs for x in outputlist)

        if "captions" in self.outputlist:
            try:
                with open (caption_filename, 'r') as f:
                    self.captions_dict = json.load(f)
            except FileNotFoundError as e:
                print(f"File {caption_filename} not found. NO captions will be loaded.")

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                if order == 'paired':
                    filename = os.path.join(dataroot, f"{phase}_pairs.txt")
                else:
                    filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")

            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()

                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)
        
        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names


    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = dataroot.split('/')[-1]

        if "captions" in self.outputlist:  # Captions
            captions = self.captions_dict[c_name.split('.')[0]]
            if self.phase == 'train':
                # During training shuffle the captions
                random.shuffle(captions)
            # Join the captions into a single one
            captions = ", ".join(captions)
            captions ='a photo of a model wearing a' + captions

        if "tryon_image" in self.outputlist:
            if self.phase == 'train':
                if not os.path.isfile(os.path.join(dataroot ,'tryon-stage1', im_name)):
                    temp=im_name.split('.')[0]+'.png'
                else:
                    temp=im_name
                tryon_image = Image.open(os.path.join(dataroot ,'tryon-stage1', temp))
                tryon_image = tryon_image.resize((self.width, self.height))
                tryon_image = self.transform(tryon_image)  # [-1,1]
            else:
                tryon_image = Image.open(os.path.join(dataroot, 'images', im_name))
                tryon_image = tryon_image.resize((self.width, self.height))
                tryon_image = self.transform(tryon_image)  # [-1,1]       

        if "garment" in self.outputlist:  # In-shop clothing image
            # garment image
            garment = Image.open(os.path.join(dataroot, 'images', c_name))
            mask = Image.open(os.path.join(dataroot, 'masks', c_name.replace(".jpg", ".png")))

            # Mask out the background
            garment = Image.composite(ImageOps.invert(mask.convert('L')), garment, ImageOps.invert(mask.convert('L')))
            garment = garment.resize((self.width, self.height))
            garment = self.transform(garment)  # [-1,1]


        if "image" in self.outputlist :
            # Person image
            image = Image.open(os.path.join(dataroot, 'images', im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]


        if  'agnostic_mask' in self.outputlist :
            # Label Map
            parse_name = im_name.replace('_0.jpg', '_4.png')
            im_parse = Image.open(os.path.join(dataroot, 'label_maps', parse_name))
            im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                                (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["hat"]).astype(np.float32) + \
                                (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                                (parse_array == label_map["scarf"]).astype(np.float32) + \
                                (parse_array == label_map["bag"]).astype(np.float32)

            parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

            if dataroot.split('/')[-1] == 'dresses':
                label_cat = 7
                parse_cloth = (parse_array == 7).astype(np.float32)
                parse_mask = (parse_array == 7).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)
                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

            elif dataroot.split('/')[-1] == 'upper_body':
                label_cat = 4
                parse_cloth = (parse_array == 4).astype(np.float32)
                parse_mask = (parse_array == 4).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                     (parse_array == label_map["pants"]).astype(np.float32)

                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
            elif dataroot.split('/')[-1] == 'lower_body':
                label_cat = 6
                parse_cloth = (parse_array == 6).astype(np.float32)
                parse_mask = (parse_array == 6).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                     (parse_array == 14).astype(np.float32) + \
                                     (parse_array == 15).astype(np.float32)
                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
            else:
                raise NotImplementedError

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
            parse_mask = parse_mask.cpu().numpy()

            if "im_head" in self.outputlist:
                # Masked cloth
                im_head = image * parse_head - (1 - parse_head)
            if "im_cloth" in self.outputlist:
                im_cloth = image * parse_cloth + (1 - parse_cloth)

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
            shape = self.transform2D(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('_0.jpg', '_2.json')
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 4))

            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
                point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []
            for pose_d in pose_data:
                ux = pose_d[0] / 384.0
                uy = pose_d[1] / 512.0

                # scale posemap points
                px = ux * self.width
                py = uy * self.height

                d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2D(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body' or dataroot.split('/')[
                -1] == 'lower_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                    shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                    elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                    elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                    wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                    wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)

                if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                    parse_mask += im_arms
                    parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    points = []
                    points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0))
                    points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0))
                    x_coords, y_coords = zip(*points)
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, c = lstsq(A, y_coords, rcond=None)[0]
                    for i in range(parse_array.shape[1]):
                        y = i * m + c
                        parse_head_2[int(y - 20 * (self.height / 512.0)):, i] = 0

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)

            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            agnostic_mask = 1 - parse_mask_total

            agnostic_mask = agnostic_mask.unsqueeze(0)


        if "densepose" in self.outputlist:
            uv = Image.open(os.path.join(dataroot, 'image-densepose', im_name))
            uv=uv.resize((self.width, self.height), Image.NEAREST)
            densepose=self.transform(uv)


        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result

    def __len__(self):
        return len(self.c_names)
