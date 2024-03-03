import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage import color
import torch
from torchvision import transforms

def load_pics_to_numpy(L_path, ab_path, datasize=8000):
    ab = np.load(ab_path)
    L = np.load(L_path)
    ab = ab[:min(datasize, ab.shape[0])]
    L = np.expand_dims(L[:min(datasize, L.shape[0])], axis=-1)
    return L, ab

# using lab color pictures to train DCGAN model
def load_lab_dataset(L_path, ab_path, img_size=128, transforms=None, datasize=8000):
    #loading color pics and grey pics to numpy array
    L, ab = load_pics_to_numpy(L_path, ab_path, datasize=datasize)
    return LABDataset(grey_imgs=L, ab_imgs=ab, img_size=img_size, transforms=transforms)

class LABDataset(Dataset):
    def __init__(self, grey_imgs, ab_imgs, img_size=None, transforms=None):
        self.grey_imgs = grey_imgs
        self.ab_imgs = ab_imgs
        self.img_size = img_size
        self.transforms = transforms
        self.grey_imgs, self.ab_imgs = self.transform_img()
    
    def transform_img(self):
        grey_imgs = []
        ab_imgs = []
        for idx in range(self.grey_imgs.shape[0]):
            grey_img = self.grey_imgs[idx].transpose(1, 2, 0)
            ab_img = self.ab_imgs[idx].transpose(1, 2, 0)
            if self.transforms:
                grey_img, ab_img = self.transforms()(grey_img), self.transforms()(ab_img)
            else:
                grey_img, ab_img = transforms.ToTensor()(grey_img), transforms.ToTensor()(ab_img)
            grey_imgs.append(grey_img.permute(2, 0, 1))
            ab_imgs.append(ab_img.permute(2, 0, 1))
        return grey_imgs, ab_imgs
            
    
    def __len__(self):
        return len(self.grey_imgs)
    
    def __getitem__(self, idx):
        return self.grey_imgs[idx], self.ab_imgs[idx]
        # grey_img = self.grey_imgs[idx]
        # ab_img = self.ab_imgs[idx]
        # if self.img_size:
        #     rgb_img = lab_to_rgb(L=grey_img, ab=ab_img)[0]
        #     rgb_img = resize(rgb_img, (self.img_size, self.img_size),  anti_aliasing=True)
        #     img = color.rgb2lab(rgb_img).astype(np.float32).transpose(-1, 0, 1)
        #     grey_img = img[:1, :, :]
        #     ab_img = img[1:, :, :]
        # if self.transforms:
        #     return  self.transforms()(grey_img), self.transforms()(ab_img)
        # else:
        #     return torch.from_numpy(grey_img), torch.from_numpy(ab_img)
    
# lab_dataset = load_lab_dataset(L_path="data/kaggle_image_colorization_dataset/l/gray_scale.npy", ab_path="data/kaggle_image_colorization_dataset/ab/ab/ab1.npy")
# print(lab_dataset[0][0].shape, lab_dataset[0][1].shape)