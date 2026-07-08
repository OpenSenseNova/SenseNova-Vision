"""COCO API wrapper used by segmentation evaluators."""

from pycocotools.coco import COCO as _COCO


class COCO(_COCO):
    """pycocotools COCO class with optional in-memory dataset input."""

    def __init__(self, annotation_file=None, dataset=None):
        if dataset is not None:
            self.dataset = dataset
            self.createIndex()
            return
        super().__init__(annotation_file)
