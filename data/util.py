import torch
import numpy as np
import rasterio


def load_tiff(path, max_value, min_value=0, norm=(-1, 1), dtype=np.float32):
    with rasterio.open(path) as src:
        image = src.read(1).astype(dtype)
        image = (norm[1] - norm[0]) * (image - min_value) / (max_value - min_value) + norm[0]
        image = torch.from_numpy(image).unsqueeze(0)
    return image


def get_profile(path):
    with rasterio.open(path) as src:
        profile = dict(src.profile)
        profile["crs"] = profile["crs"].to_string()
        profile["transform"] = profile["transform"].to_gdal()
    return profile