import numpy as np
import cv2


def linear_correction_bgr(img):
    img_corrected = np.zeros(img.shape)
    for color in range(3):
        color_max = np.max(img[:, :, color])
        color_min = np.min(img[:, :, color])
        img_corrected[:, :, color] = (img[:, :, color] - color_min) / (color_max - color_min) * 255
    img_corrected = np.clip(img_corrected, 0, 255)
    return img_corrected.astype(dtype=np.uint8, copy=False)


def gray_world_bgr(img):
    img_corrected = np.zeros(img.shape)
    all_color_mean = np.mean(img)
    for color in range(3):
        color_mean = np.mean(img[:, :, color])
        img_corrected[:, :, color] = img[:, :, color] * (all_color_mean / color_mean)
    img_corrected = np.clip(img_corrected, 0, 255)
    return img_corrected.astype(dtype=np.uint8, copy=False)


def gamma_corr(img, gamma=1.):
    img_corrected = np.clip(np.power(img / 255, 1. / gamma) * 255, 0, 255)
    return img_corrected.astype(dtype=np.uint8, copy=False)


def retinex(img, sigma=50):
    img_corrected = (np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))) * 255 + 127
    img_corrected = np.clip(img_corrected, 0, 255)
    return img_corrected.astype(dtype=np.uint8, copy=False)
