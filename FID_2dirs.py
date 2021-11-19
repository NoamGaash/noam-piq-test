import os
import piq
import torch
import torchvision.transforms as transforms
from PIL import Image

import argparse
import pathlib
import lpips
from piq import SSIMLoss, FID

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()






def getFilePaths(path):
    result = (list(pathlib.Path(path).glob("**/*.png")))
    return result


class DataProcess(torch.utils.data.Dataset):
    def __init__(self, s_src, img_w = 256, img_h = 256):
        super(DataProcess, self).__init__()
        # self.img_w = img_w
        # self.img_h = img_h

        self.img_transform = transforms.Compose([
            # transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),  # transform to [0, 1]
        ])

        self.f_srcs = getFilePaths(s_src)

    def __getitem__(self, index):
        src = Image.open(self.f_srcs[index])
        t_src = self.img_transform(src.convert('RGB'))

        return {
            'images': t_src,
        }

    def __len__(self):
        return len(self.f_srcs)

if __name__ == '__main__':   
    s_1 = opt.dir0
    s_2 = opt.dir1
    
    set_1 = DataProcess(s_1) #, 256, 256)
    set_2 = DataProcess(s_2) #, 256, 256)
    
    loader_1 = torch.utils.data.DataLoader(set_1, batch_size=1, shuffle=False)
    loader_2 = torch.utils.data.DataLoader(set_2, batch_size=1, shuffle=False)
    
    print(f'compare {opt.dir0} vs {opt.dir1}')

    fid_metric = piq.FID()
    feat_1 = fid_metric.compute_feats(loader_1)
    feat_2 = fid_metric.compute_feats(loader_2)
    fid = fid_metric.compute_metric(feat_1, feat_2)
    print(f'fid: {fid}')

    is_metric = piq.IS()
    feat_1 = is_metric.compute_feats(loader_1)
    feat_2 = is_metric.compute_feats(loader_2)
    IS = is_metric.compute_metric(feat_1, feat_2)
    print(f'IS: {IS}')