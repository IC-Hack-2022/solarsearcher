import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.unet import UNet

class Segmenter:

    def __init__(self, saved_model):

        self.model = UNet(n_channels=3, n_classes=7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        self.model.load_state_dict(torch.load(saved_model, self.device))
        

    def predict(self, model, full_img, scale):

        model.eval()

        img = torch.from_numpy(self.preprocess(full_img, scale))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

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

            image = image.squeeze().cpu().numpy()

        return image, class_idx


    def preprocess(self, pil_img, scale):

        W, H = pil_img.size
        new_W, new_H = int(scale*W), int(scale*H)
        pil_img = pil_img.resize((new_W, new_H))

        img = np.array(pil_img)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255
        return img


    def calculate_max_val_area(self, mask_indices, valid_idx=2):
        area = (mask_indices == torch.tensor(valid_idx, dtype=torch.uint8)).sum()
        frac_area = area / np.prod(mask_indices)
        return frac_area


    def segment(self, location, input_path, output_path, scale):
        
        inpath = os.path.join(input_path, location)
        outpath = os.path.join(output_path, location)

        if not os.path.exists(outpath):
            os.makedirs(outpath)
            
            for img_path in os.listdir(inpath):

                img = Image.open(os.path.join(inpath, img_path))

                seg, mask_indices = self.predict(model=self.model, full_img=img, scale=scale)

                im = Image.fromarray(seg)
                im.save(os.path.join(outpath, img_path))


if __name__ == "__main__":
    pass