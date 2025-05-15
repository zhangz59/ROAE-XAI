import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn.functional as F


def nii_loader(path, dtype=np.float32, mmap_mode='r'):
    """
    Load NIfTI file with memory mapping option for large files.

    Args:
        path: Path to NIfTI file
        dtype: Data type to cast to (default: float32)
        mmap_mode: Memory mapping mode (default: 'r' for read-only)
                   Set to None to load data into memory
    """
    img = nib.load(str(path))
    # Use memory mapping for large files
    data = img.get_fdata(dtype=dtype, caching='unchanged')
    return data


def read_table(path):
    """Read Excel table and return values"""
    return pd.read_excel(path, header=None).values


def white0(image, threshold=0):
    """
    Standardize voxels with value > threshold

    Args:
        image: Input image
        threshold: Threshold value

    Returns:
        Standardized image
    """
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    # Vectorized implementation to avoid unnecessary memory allocation
    image_h = image * mask

    # Calculate mean and std only for relevant voxels
    non_zero_voxels = np.sum(mask)
    if non_zero_voxels > 0:
        mean = np.sum(image_h) / non_zero_voxels

        # More memory efficient way to calculate std
        std_sum = np.sum((image_h - mean * mask) ** 2)
        std = np.sqrt(std_sum / non_zero_voxels)

        if std > 0:
            normalized = mask * (image - mean) / std
            # Use in-place operations to reduce memory usage
            image = normalized + image * (1 - mask)
            return image

    # Default case
    return np.zeros_like(image, dtype=np.float32)


class IMG_Folder(torch.utils.data.Dataset):
    """
    Dataset class for loading brain images with memory optimizations
    """

    def __init__(self, excel_path, data_path, loader=nii_loader, transforms=None, preload=False):
        """
        Args:
            excel_path: Path to Excel file with metadata
            data_path: Path to directory with NIfTI files
            loader: Function to load NIfTI files
            transforms: Transforms to apply to images
            preload: Whether to preload all data into memory (default: False)
        """
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms
        self.preload = preload

        # Create a mapping from subject ID to metadata for faster lookup
        self.metadata = {}
        for f in self.table_refer:
            sid = str(f[0])
            slabel = int(f[1])
            smale = f[2]
            self.metadata[sid] = (slabel, smale)

        # Optionally preload all data into memory
        if preload:
            self.cached_data = {}
            for sub_fn in self.sub_fns:
                if sub_fn in self.metadata:
                    sub_path = os.path.join(self.root, sub_fn)
                    self.cached_data[sub_fn] = self.loader(sub_path)

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self, index):
        sub_fn = self.sub_fns[index]

        # Get metadata for this subject
        if sub_fn not in self.metadata:
            # Find manually if not in mapping (fallback)
            for f in self.table_refer:
                sid = str(f[0])
                slabel = int(f[1])
                smale = f[2]
                if sid == sub_fn:
                    break
        else:
            slabel, smale = self.metadata[sub_fn]
            sid = sub_fn

        # Load image data
        if self.preload and sub_fn in self.cached_data:
            img = self.cached_data[sub_fn].copy()  # Make a copy to avoid modifying cached data
        else:
            sub_path = os.path.join(self.root, sub_fn)
            img = self.loader(sub_path)

        # Preprocessing
        img = white0(img)

        # Apply transforms if provided
        if self.transform is not None:
            img = self.transform(img)

        # Convert to contiguous float tensor
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        return (img, sid, slabel, smale)