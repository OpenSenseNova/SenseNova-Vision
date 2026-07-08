import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def rgb2id(rgb) -> np.ndarray:
    if isinstance(rgb, Image.Image):
        rgb = np.asarray(rgb)
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        return rgb.astype(np.int32)
    return (
        rgb[:, :, 0].astype(np.int32)
        + 256 * rgb[:, :, 1].astype(np.int32)
        + 256 * 256 * rgb[:, :, 2].astype(np.int32)
    )


def _decode_uncompressed_counts(counts):
    values = []
    number = 0
    shift = 0
    for char in str(counts):
        c = ord(char) - 48
        number |= (c & 0x1F) << shift
        if c & 0x20:
            shift += 5
        else:
            if c & 0x10:
                number |= -1 << (shift + 5)
            values.append(number)
            number = 0
            shift = 0
    for idx in range(2, len(values)):
        values[idx] += values[idx - 2]
    return values


def decode_rle(segmentation) -> np.ndarray:
    if mask_utils is None:
        raise RuntimeError("pycocotools is required to decode RLE masks.")
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        counts = rle.get("counts")
        if isinstance(counts, str):
            try:
                return mask_utils.decode(rle).astype(bool)
            except Exception:
                rle["counts"] = _decode_uncompressed_counts(counts)
        return mask_utils.decode(rle).astype(bool)
    return np.asarray(segmentation).astype(bool)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return boundary & mask


def thicken_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def to_binary_mask(img: Image.Image, threshold: int = 127) -> np.ndarray:
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    return (arr > threshold).astype(np.uint8)


def decode_mask(segmentation, height, width):
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], str):
            segmentation = dict(segmentation)
            segmentation["counts"] = segmentation["counts"].encode("utf-8")
        if isinstance(segmentation["counts"], list):
            segmentation = mask_utils.frPyObjects(segmentation, *segmentation["size"])
            segmentation["counts"] = segmentation["counts"].decode("utf-8")
        mask = mask_utils.decode(segmentation).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask.squeeze())
    elif isinstance(segmentation[0], dict):
        for seg in segmentation:
            mask = mask_utils.decode(seg).astype(np.uint8)
            binary_mask = np.maximum(binary_mask, mask.squeeze())
    elif isinstance(segmentation[0], list):
        for seg in segmentation:
            rles = mask_utils.frPyObjects([seg], height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            binary_mask = np.maximum(binary_mask, mask.squeeze())
    else:
        raise ValueError(f"Invalid segmentation type: {type(segmentation)}")

    return binary_mask


def encode_mask(mask, encoding="ascii"):
    assert set(np.unique(mask)).issubset({0, 1})
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode(encoding)
    return rle


def compute_confusion_matrix(gt, pred, num_classes=2):
    return np.bincount(
        (num_classes * pred.reshape(-1) + gt.reshape(-1)),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def to_multi_mask(
    img: Image.Image, class_define: np.ndarray, full_colormap=None
) -> np.ndarray:
    """
    Convert an RGB mask to a class-index mask by nearest prompt color.

    Black pixels are always assigned to the background index len(class_define).
    When full_colormap is provided, pixels farther than the legacy threshold
    from every prompt color are also assigned to the background index.
    """
    img_np = np.array(img.convert("RGB"), dtype=np.float32)
    class_define = np.asarray(class_define, dtype=np.float32)
    if class_define.ndim != 2 or class_define.shape[1] != 3:
        raise ValueError(
            f"class_define must have shape [N, 3], got {class_define.shape}"
        )

    mask_black = np.all(img_np == 0, axis=-1)
    h, w, _ = img_np.shape
    pixels = img_np.reshape(-1, 3)

    pred_class = np.empty((pixels.shape[0],), dtype=np.int32)
    min_dist = np.empty((pixels.shape[0],), dtype=np.float32)
    chunk_size = 262144
    for start in range(0, pixels.shape[0], chunk_size):
        end = min(start + chunk_size, pixels.shape[0])
        diff = pixels[start:end, None, :] - class_define[None, :, :]
        dist = np.sum(diff * diff, axis=-1)
        pred_class[start:end] = np.argmin(dist, axis=1)
        min_dist[start:end] = np.sqrt(np.min(dist, axis=1))

    if full_colormap is not None:
        pred_class[min_dist > 45] = len(class_define)

    mask = pred_class.reshape(h, w)
    mask[mask_black] = len(class_define)
    return mask.astype(np.int32, copy=False)


def decode_rle_to_class_mask(rle_list, height, width):
    """
    segmentation=[RLE1, RLE2, ...]
    每个 RLE 对应 class_id = index
    """
    class_mask = np.zeros((height, width), dtype=np.uint8)

    for cls_id, rle in enumerate(rle_list):
        mask = decode_mask(rle, height, width)
        class_mask[mask > 0] = cls_id

    return class_mask


def gt_to_rgb_mask(coco_gt, imgid, color_map):
    imginfo = coco_gt.loadImgs([imgid])[0]
    height, width = imginfo["height"], imginfo["width"]

    gt_ids = coco_gt.getAnnIds(imgIds=[imgid])
    gt_anns = coco_gt.loadAnns(gt_ids)

    recolor_map = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, ann in enumerate(gt_anns):
        seg = ann["segmentation"]  # RLE
        mask = decode_mask(seg, height, width)  # 你已有的函数
        color = color_map[idx % len(color_map)]
        recolor_map[mask > 0] = color

    return recolor_map


def pred_to_rgb_mask(pred, coco_gt, color_map):
    imgid = pred["image_id"]
    imginfo = coco_gt.loadImgs([imgid])[0]
    height, width = imginfo["height"], imginfo["width"]

    recolor_map = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, ann in enumerate(pred["segmentation"]):
        seg = ann  # RLE
        mask = decode_mask(seg, height, width)  # 你已有的函数
        color = color_map[idx % len(color_map)]
        recolor_map[mask > 0] = color

    return recolor_map
