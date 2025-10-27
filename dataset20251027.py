from typing import Sequence, Union
import json
import random
import numpy as np
import nibabel as nib
#import cv2
from abc import abstractmethod
import torch
from types import SimpleNamespace as NS
from pathlib import Path
from monai.data import CacheDataset
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    Lambda,
    Transform,
    MapTransform,
    LoadImaged,
    DeleteItemsd,
    SpatialPadd,
    MaskIntensityd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    RandFlipd,
    CenterSpatialCropd,
    RandZoomd,
    CropForegroundd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandSpatialCropSamples,
    Resized,
    Lambdad,
    SpatialCropd,
    RandomizableTransform,
    EnsureTyped,
    ClipIntensityPercentilesd,
    ToDeviced
)
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable
from monai.transforms.inverse import TraceableTransform

from typing import Hashable, Mapping, Any, Dict, List
from monai.transforms.traits import MultiSampleTrait
from monai.utils.misc import ensure_tuple_rep, set_determinism
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.utils import ImageMetaKey as Key

class CropSlicesd(MapTransform):
    def __init__(self, keys, num_slices: Union[Sequence[int], int]):
        super().__init__(keys)
        self.num_slices = ensure_tuple_rep(num_slices, 2)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.crop_volume(d[key])
        return d

    def crop_volume(self, volume):
        """
        Function to crop the same amount of slices from the top and bottom of a volume.
        """
        return volume[..., self.num_slices[0]:-self.num_slices[1]]

class AddMetaData(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            # 检查键是否存在于数据中，并添加元数据
            if key in data:
                data[key + "_meta"] = data[key].meta  # 根据你的元数据结构进行调整
        return data    
def get_5EchoT2w_dataset_2d_3map(phase="train", prob=0.5, device='cpu', debug=False):  
    data_file = Path(__file__).parent / "mdme_datasetZC_YB.json"
    
    with data_file.open() as f:
        datalist = json.load(f)
    if phase == "train":
        transforms = Compose([
            LoadImaged(keys=["pdm", "t1m", "t2m"], ensure_channel_first=True),
            CropForegroundd(keys=["t1m", "t2m", "pdm"], source_key="t1m", margin=[5, 5, 0], allow_smaller=False, lazy=False),
            ToDeviced(keys=["pdm", "t1m", "t2m"], device=device),
            ClipIntensityPercentilesd(keys=["pdm", "t1m", "t2m"], lower=0.0, upper=99.5),
            SpatialPadd(keys=["t1m", "t2m", "pdm"], spatial_size=(256, 256, -1)),
            RandFlipd(keys=["t1m", "t2m", "pdm"], prob=prob, spatial_axis=0, lazy=False),
            RandZoomd(keys=["t1m", "t2m", "pdm"], prob=prob, min_zoom=[0.9, 0.9, 1], max_zoom=[1.1, 1.1, 1], lazy=False),
            CropSlicesd(keys=["t1m", "t2m", "pdm"], num_slices=[10, 5]),
            RandSpatialCropSamplesd(keys=["t1m", "t2m", "pdm"], roi_size=[-1, -1, 1], num_samples=4, random_center=True, lazy=True),
            Lambdad(keys=["t1m", "t2m", "pdm"], func=lambda x: x.squeeze(-1)),
            CenterSpatialCropd(keys=["t1m", "t2m", "pdm"], roi_size=[256, 256])
        ])
        return CacheDataset(datalist['train'], transform=transforms, cache_rate=0 if debug else 1, num_workers=1)
    elif phase == "val":
        transforms = Compose([
            LoadImaged(keys=["pdm", "t1m", "t2m"], ensure_channel_first=True),
            CropForegroundd(keys=["t1m", "t2m", "pdm"], source_key="t1m", margin=[5, 5, 0], allow_smaller=False, lazy=False),
            ToDeviced(keys=["pdm", "t1m", "t2m"], device=device),
            ClipIntensityPercentilesd(keys=["pdm", "t1m", "t2m"], lower=0.0, upper=99.5),
            SpatialPadd(keys=["t1m", "t2m", "pdm"], spatial_size=(256, 256, -1)),
            RandFlipd(keys=["t1m", "t2m", "pdm"], prob=prob, spatial_axis=0, lazy=False),
            RandZoomd(keys=["t1m", "t2m", "pdm"], prob=prob, min_zoom=[0.9, 0.9, 1], max_zoom=[1.1, 1.1, 1], lazy=False),
            CropSlicesd(keys=["t1m", "t2m", "pdm"], num_slices=[10, 5]),
            CenterSpatialCropd(keys=["pdm","t1m", "t2m"], roi_size=[256, 256, -1])
        ])
        return CacheDataset(datalist['val'], transform=transforms, cache_rate=0 if debug else 1, num_workers=1)
    
    elif phase == "test":
        transforms = Compose([
            LoadImaged(keys=["pdm", "t1m", "t2m"], ensure_channel_first=True),
            AddMetaData(keys=["t1m", "t2m", "pdm"]),
            CropForegroundd(keys=["t1m", "t2m", "pdm"], source_key="t1m", margin=[5, 5, 0], allow_smaller=False, lazy=False),
            ToDeviced(keys=["pdm", "t1m", "t2m"], device=device),
            ClipIntensityPercentilesd(keys=["pdm", "t1m", "t2m"], lower=0.0, upper=99.5),
            SpatialPadd(keys=["t1m", "t2m", "pdm"], spatial_size=(256, 256, -1)),
            CenterSpatialCropd(keys=["pdm", "t1m", "t2m"], roi_size=[256, 256, -1])
        ])
        return CacheDataset(datalist['test'], transform=transforms, cache_rate=0 if debug else 1, num_workers=1)       

