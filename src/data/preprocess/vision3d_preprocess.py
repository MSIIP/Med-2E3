import monai.transforms as mtf
import numpy as np
import torch


class Vision3DPreprocess:
    def __init__(self, vision3d_processor, data_arguments=None, mode="train"):
        self.vision3d_processor = vision3d_processor

        if mode == "train":
            self.transform = mtf.Compose(
                [
                    mtf.CropForeground(),
                    mtf.Resize(spatial_size=[32, 256, 256], mode="trilinear"),
                    mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandFlip(prob=0.10, spatial_axis=0),
                    mtf.RandFlip(prob=0.10, spatial_axis=1),
                    mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        else:
            self.transform = mtf.Compose(
                [
                    mtf.CropForeground(),
                    mtf.Resize(spatial_size=[32, 256, 256], mode="trilinear"),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )

    def __call__(self, vision3d):
        # vision3d = self.vision3d_processor(vision3d, return_tensors="pt")
        # vision3d = vision3d["pixel_values"][0]
        # vision3d = torch.tensor(vision3d)
        if vision3d.ndim == 3:
            vision3d = vision3d[np.newaxis, ...]  # [1, N, H, W]
        vision3d = self.transform(vision3d)  # [1, 32, 256, 256]
        return vision3d
