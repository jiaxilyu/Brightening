import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import cv2


def generate_img(L, ab=None):
    L = L  * 100
    if ab is None:
        ab_shape = (2, L.shape[1], L.shape[2])
        ab = np.zeros(ab_shape)
    else:
        ab = (ab - 0.5) * 128 * 2
    img = np.concatenate([L, ab], axis=0)
    return  color.lab2rgb(img.transpose(1, 2, 0))


    

