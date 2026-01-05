import SimpleITK as sitk
from torch import nn
import torch.nn.functional as F
import re
from .utils_general import *
from monai.transforms import *
from monai.transforms import Lambda


class GetCentroidOfLargestComponentd(MapTransform):
    def __init__(self, keys, output_key='centroid', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[key]
            # remove channel dimension if present
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            indices = np.argwhere(mask == mask.max())
            center = np.round(indices.mean(axis=0)).astype(int)
            d[self.output_key] = tuple(center.tolist())  # ensure tuple, length=3
        return d


class CropAroundCentroidd(MapTransform):
    def __init__(self, keys, centroid_key='centroid', roi_size=(96, 96, 96), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.centroid_key = centroid_key
        self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)
        center = np.array(d[self.centroid_key])
        for key in self.keys:
            img = d[key]
            if img.ndim == 4:  # assume (C, Z, Y, X)
                c, z, y, x = img.shape
            else:
                raise ValueError(f"Unexpected shape for {key}: {img.shape}")
            size = np.array(self.roi_size)
            start = np.maximum(center - size // 2, 0)
            end = np.minimum(start + size, [z, y, x])
            # Adjust start if near border
            start = np.maximum(end - size, 0)
            slices = tuple([slice(None)] + [slice(s, e) for s, e in zip(start, end)])
            d[key] = img[slices]
        return d




transforms = Compose([
   
    # followed by MONAI transforms
])


def get_transforms(traintest='train'):

    return dose_transforms(traintest=traintest)



        
##Transforms for a pCT
def dose_transforms(traintest):

    image_key = "ct"
    dose_key = "dose"
    mask_keys = ["structure_masks", "possible_dose_mask"]
    all_keys = [image_key] + mask_keys + [dose_key]

    if traintest == 'train':
        return Compose([
            ToTensord(keys=all_keys),

            # Ensure correct channel-first format
            Lambdad(keys=all_keys, func=lambda x: x.permute(3, 0, 1, 2) if x.ndim == 4 and x.shape[-1] in [1, 10] else x),

            # Normalize HU range for CT
            ScaleIntensityRanged(
                keys=image_key, a_min=-1024, a_max=1500, b_min=-1.024, b_max=1.5, clip=True
            ),

            # Spatial transforms
            RandFlipd(keys=all_keys, spatial_axis=[0, 1, 2], prob=0.5),

            RandAffined(
                keys=all_keys,
                rotate_range=(0.1, 0.1, 0.1),
                shear_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode="nearest",
                padding_mode="zeros",
                prob=0.3,
            ),

            Rand3DElasticd(
                keys=all_keys,
                sigma_range=(4, 8),
                magnitude_range=(40, 60),
                spatial_size=None,
                prob=0.15,
                rotate_range=(0.05, 0.05, 0.05),
                translate_range=(10, 10, 10),
            ),

            RandZoomd(
                keys=all_keys,
                min_zoom=0.9,
                max_zoom=1.3,
                mode="nearest",
                padding_mode="minimum",
                prob=0.5,
            ),

            # Intensity augmentations 
            RandBiasFieldd(keys=image_key, prob=0.2),
            RandGaussianNoised(keys=image_key, prob=0.2, mean=0.0, std=0.01),
            RandGibbsNoised(keys=image_key, prob=0.1, alpha=(0.2, 0.5)),
            RandShiftIntensityd(keys=image_key, offsets=0.1, prob=0.3),
            RandScaleIntensityd(keys=image_key, factors=0.1, prob=0.3),
            RandAdjustContrastd(keys=image_key, prob=0.2),
            RandGaussianSmoothd(keys=image_key, prob=0.2),

            # Finalize
            EnsureTyped(keys=all_keys),
            ])

    elif traintest =='infer':
        
        
                return  Compose([
            
            ToTensord(keys=all_keys),
            Lambdad(keys=all_keys,func=lambda x: x.permute(3, 0, 1, 2) if x.ndim == 4 and x.shape[-1] in [1,10] else x),
            ScaleIntensityRanged(keys=image_key, a_min=-1024, a_max=1500, b_min=-1.024, b_max=1.5, clip=True),
            EnsureTyped(keys=all_keys),
        ])