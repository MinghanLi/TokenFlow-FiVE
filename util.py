from pathlib import Path
from PIL import Image
import torch
import yaml
import math

import torchvision.transforms as T
from torchvision.io import read_video, write_video
import os
import random
import numpy as np
from torchvision.io import write_video
# from kornia.filters import joint_bilateral_blur
from kornia.geometry.transform import remap
from kornia.utils.grid import create_meshgrid
import cv2


def save_video_frames(video_path, img_size=(512,512)):
    video_dir = video_path.split('/')[0]
    video_path_resize = '/'.join([video_dir+'_resize'] + video_path.split('/')[1:])
    if os.path.exists(video_path_resize):
        print(f"Video existed! Skiping {video_path_resize}")
        return 
    Path(video_path_resize).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(video_path):
        for filename in os.listdir(video_path):
            if not (filename.endswith('.jpg') or filename.endswith('.png')):
                continue

            image = Image.open(os.path.join(video_path, filename))
            image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
            image_resized.save(f'{video_path_resize}/{filename}')
       
    else:
        video, _, _ = read_video(video_path, output_format="TCHW")
        # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
        if video_path.endswith('.mov'):
            video = T.functional.rotate(video, -90)
        video_name = Path(video_path).stem
        
        for i in range(len(video)):
            ind = str(i).zfill(5)
            image = T.ToPILImage()(video[i])
            image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
            image_resized.save(f'{video_path_resize}/{ind}.png')

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video_old(raw_frames, save_path, fps=10, img_size=(512,512)):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def save_video(raw_frames, save_path, fps=10, img_size=(512, 512)):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset
    }

    resized_frames = []
    for frame in raw_frames:
        # Convert to PIL Image for resizing
        pil_frame = T.ToPILImage()(frame)
        resized_frame = pil_frame.resize((img_size),  resample=Image.Resampling.LANCZOS)
        # Convert back to tensor
        resized_frame = T.ToTensor()(resized_frame)
        resized_frames.append(resized_frame)
    
    frames = torch.stack(resized_frames)
    frames = (frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


