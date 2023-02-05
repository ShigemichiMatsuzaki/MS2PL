import numpy as np
import cv2

def rgb2lab(rgb_img: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB

    Parameters
    ----------
    rgb_img : `np.ndarray`
        RGB image

    Returns
    -------
    `np.ndarray`
        LAB image
    """
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
    return lab_img

def compute_statistics(img: np.ndarray, is_std: bool=False) -> list:
    """Compute mean and standard deviation of pixel values

    Parameters
    ----------
    img : `np.ndarray`
        Input image
    is_std : `bool`
        `True` to return standard deviation (np.sqrt(variance)).
        Else, return variance. Default: `False`

    Returns
    -------
    `list`
        0: mean, 1: variance or standard deviation
        
    """
    # Ensure that the image is not in uint8 (subtraction may underfloat)
    img_float = np.array(img, dtype=np.float64)
    mean = np.mean(img_float, axis=(0, 1))
    variance = ((img_float - mean) ** 2).mean(axis=(0, 1))

    if is_std:
        std = np.sqrt(variance)
        return mean, std
    else:
        return mean, variance

def translate_image(
    img_source: np.ndarray, 
    source_mean: np.ndarray, 
    source_std: np.ndarray, 
    target_mean: np.ndarray, 
    target_std: np.ndarray
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    img_source : `np.ndarray`
        Source image
    source_mean : `np.ndarray`
        Mean of pixel values of source images
    source_std : `np.ndarray`
        Standard deviation of pixel values of source images
    target_mean : `np.ndarray`
        Mean of pixel values of target images
    target_std : `np.ndarray`
        Standard deviation of pixel values of target images

    Returns
    -------
    `np.ndarray`
        Image translated from source to target
    """
    img_source_float = np.array(img_source, dtype=np.float64)

    img_target_float = (img_source_float - source_mean) / source_std * target_std + target_mean
    img_target = np.array(img_target_float, dtype=np.uint8)

    return img_target