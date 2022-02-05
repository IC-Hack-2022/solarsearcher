import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.unet import UNet


parser = argparse.ArgumentParser(description='Predict masks from input images')
parser.add_argument('--model', '-m', default='saved_models/CP_epoch30.pth',
                    metavar='FILE',
                    help="Specify the file in which the model is stored")
parser.add_argument('--input', '-i', help='filenames of input images')
parser.add_argument('--output', '-o', help='Filenames of ouput images')
parser.add_argument('--scale', '-s', type=float, default=0.2)
parser.add_argument('--mask-threshold', '-t', type=float, default=0.5)
parser.add_argument('--output', '-o', default='predictions/')
parser.add_argument('--valid-idx', type=int, default=2)

def predict(model, full_img, **kwargs):

    args = kwargs['args']

    model.eval()

    img = torch.from_numpy(preprocess(full_img, args.scale))
    img = img.unsqueeze(0)
    img = img.to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)
        h, w = probs.shape[1], probs.shape[2]

        # convert probabilities to class index and then to RGB
        mapping = {0: (0  , 255, 255),     #urban_land
                   1: (255, 255, 0  ),     #agriculture
                   2: (255, 0  , 255),     #rangeland
                   3: (0  , 255, 0  ),     #forest_land
                   4: (0  , 0  , 255),     #water
                   5: (255, 255, 255),     #barren_land
                   6: (0  , 0  , 0  )}     #unknown
        class_idx = torch.argmax(probs, dim=0)
        image = torch.zeros(h, w, 3, dtype=torch.uint8)

        for key in mapping:  
          idx = (class_idx == torch.tensor(key, dtype=torch.uint8))
          validx = (idx == 1)
          image[validx,:] = torch.tensor(mapping[key], dtype=torch.uint8)

        # image = image.permute(1,2,0)
        image = image.squeeze().cpu().numpy()

    return image, class_idx


def preprocess(pil_img, scale):

    W, H = pil_img.size
    new_W, new_H = int(scale*W), int(scale*H)
    pil_img = pil_img.resize((new_W, new_H))

    img = np.array(pil_img)
    img = img.transpose((2, 0, 1))
    if img.max() > 1:
        img = img / 255
    return img

    
def calculate_val_area(mask_indices, valid_idx=2):
    area = (mask_indices == torch.tensor(valid_idx, dtype=torch.uint8)).sum()
    return area


def main():

    args = parser.parse_args()

    model = UNet(n_channels=3, n_classes=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    model.load_state_dict(torch.load(args.model, device))

    for i, fn in enumerate(args.input):

        name=fn.split('/')[-1]
        name=name.split('.')[0]

        img = Image.open(fn)
        seg, mask_indices = predict(model=model,
                           full_img=img, 
                           args=args)

        im = Image.fromarray(seg)
        im.save(args.output+'pred_'+name+'.jpeg')

        return im 


if __name__ == "__main__":
    main()