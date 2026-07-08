from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def restore_mask_to_original(mask, orig_size: Tuple[int, int], pad_info):
    """
    mask: PIL.Image or torch.Tensor (C,H,W) or (H,W)
    orig_size: (w,h)
    pad_info: optional square/rect padding metadata
    """
    orig_w, orig_h = orig_size

    if isinstance(mask, Image.Image):
        if pad_info is None or pad_info.get("mode") in ("rect", "none"):
            return mask.resize((orig_w, orig_h), resample=Image.NEAREST)
        if pad_info.get("mode") == "square":
            px = pad_info["paste_x"]
            py = pad_info["paste_y"]
            nw = pad_info["new_w"]
            nh = pad_info["new_h"]
            crop = mask.crop((px, py, px + nw, py + nh))
            return crop.resize((orig_w, orig_h), resample=Image.NEAREST)
        return mask.resize((orig_w, orig_h), resample=Image.NEAREST)

    if isinstance(mask, torch.Tensor):
        t = mask
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim == 3:
            c, h, w = t.shape
        elif t.ndim == 4:
            t = t[0]
            c, h, w = t.shape
        else:
            raise TypeError(f"Unsupported tensor mask shape: {t.shape}")

        if pad_info is None or pad_info.get("mode") in ("rect", "none"):
            t = t.unsqueeze(0)
            resized = F.resize(
                t, (orig_h, orig_w), interpolation=InterpolationMode.NEAREST
            )
            return resized.squeeze(0)
        if pad_info.get("mode") == "square":
            px = pad_info["paste_x"]
            py = pad_info["paste_y"]
            nw = pad_info["new_w"]
            nh = pad_info["new_h"]
            cropped = t[:, py : py + nh, px : px + nw].unsqueeze(0)
            resized = F.resize(
                cropped, (orig_h, orig_w), interpolation=InterpolationMode.NEAREST
            )
            return resized.squeeze(0)

    raise TypeError(f"Unsupported mask type: {type(mask)}")


def to_multi_mask(
    img: Image.Image,
    class_define: np.ndarray,
    full_colormap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Torch implementation of RGB mask to class-index mask by nearest prompt color.
    """
    img_np = np.array(img.convert("RGB"), dtype=np.float32)
    class_define = np.asarray(class_define, dtype=np.float32)
    if class_define.ndim != 2 or class_define.shape[1] != 3:
        raise ValueError(
            f"class_define must have shape [N, 3], got {class_define.shape}"
        )

    mask_black = np.all(img_np == 0, axis=-1)
    h, w, _ = img_np.shape
    pixels = torch.from_numpy(img_np.reshape(-1, 3))
    class_tensor = torch.from_numpy(class_define)

    dis = torch.cdist(pixels.unsqueeze(0), class_tensor.unsqueeze(0)).squeeze(0)
    pred_class = torch.argmin(dis, dim=1)
    min_dis = dis[torch.arange(dis.shape[0]), pred_class]

    if full_colormap is not None:
        pred_class[min_dis > 45] = len(class_define)

    mask = pred_class.reshape(h, w).numpy().astype(np.int32)
    mask[mask_black] = len(class_define)
    return mask
