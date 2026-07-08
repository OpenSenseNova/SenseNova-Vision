# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .edit_dataset_jsonl import EditJSONLIterableDataset
from .recon3d.recon3d_dataset_jsonl import Recon3DJsonLIterableDataset
from .seg_dataset_jsonl import SegJSONLIterableDataset
from .t2i_dataset_jsonl import T2IJSONLIterableDataset
from .vlm_dataset import SftJSONLIterableDataset

DATASET_REGISTRY = {
    "seg_instance_sft": SegJSONLIterableDataset,
    "seg_gcg_sft": SegJSONLIterableDataset,
    "seg_sft": SegJSONLIterableDataset,
    "vlm_sft": SftJSONLIterableDataset,
    "t2i_general": T2IJSONLIterableDataset,
    "general_edit": EditJSONLIterableDataset,
    "cv_depth": EditJSONLIterableDataset,
    "cv_normal": EditJSONLIterableDataset,
    "cv_detection_bbox": SftJSONLIterableDataset,
    "cv_detection_point": SftJSONLIterableDataset,
    "cv_detection_point_dense": SftJSONLIterableDataset,
    "cv_detection_bbox_dense": SftJSONLIterableDataset,
    "cv_detection_bbox_referring": SftJSONLIterableDataset,
    "cv_detection_point_referring": SftJSONLIterableDataset,
    "cv_detection_ocr_text_box": SftJSONLIterableDataset,
    "cv_detection_ocr_text_poly": SftJSONLIterableDataset,
    "cv_detection_ocr_word_box": SftJSONLIterableDataset,
    "cv_detection_ocr_word_poly": SftJSONLIterableDataset,
    "cv_detection_visual": SftJSONLIterableDataset,
    "cv_detection_keypoint": SftJSONLIterableDataset,
    "cv_detection_layout": SftJSONLIterableDataset,
    "cv_detection_gui": SftJSONLIterableDataset,
    "recon3d_jsonl_sft": Recon3DJsonLIterableDataset,
    "vlm_sft_image": SftJSONLIterableDataset,
}

DATASET_INFO = {
    "seg_instance_sft": {
        "coconut_b": {
            "data_dir": "",
            "jsonl_path": "segmentation/coconut_b_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 241602,
        },
        "coconut_l": {
            "data_dir": "",
            "jsonl_path": "segmentation/coconut_l_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 103010,
        },
        "coconut_xl": {
            "data_dir": "",
            "jsonl_path": "segmentation/coconut_xl_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 242640,
        },
        "Cityscapes_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/cityscapes_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 2975,
        },
        "Hypersim_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/hypersim_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 46835,
        },
        "Entity_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/entityv2_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 31677,
        },
        "Trashcan_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/Trashcan_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 5936,
        },
        "Pidray_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/pidray_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 29454,
        },
        "ZeroWaste_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/ZeroWaste_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 2947,
        },
        "LVIS_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/LVIS_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 56740,
        },
        "IDD-1_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/IDD-1_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 6984,
        },
        "IDD-2_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/IDD-2_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 7034,
        },
        "IDDA_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/IDDAV3_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 6240,
        },
        "MapillaryVistas_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/MapillaryVistas_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 18000,
        },
        "nuImages_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/NuScenes_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 65547,
        },
        "51World_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/51world_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 11588,
        },
        "StreetHazards_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/StreetHazards_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 6156,
        },
        "KITTI_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/KITTI_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 5027,
        },
        "TAS500_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/TAS500_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 440,
        },
        "UDD5_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/UDD5_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 120,
        },
        "UDD6_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/UDD6_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 106,
        },
        "TTPLA_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/TTPLA_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 1242,
        },
        "LoveDA_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/LOVEDA_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 2522,
        },
        "VIPSeg_panoptic": {
            "data_dir": "",
            "jsonl_path": "segmentation/VIPSeg_total_unified_genseg_edit_regenerated.jsonl",
            "num_total_samples": 66767,
        },
    },
    "seg_gcg_sft": {
        "gcg_grandf": {
            "data_dir": "",
            "jsonl_path": "segmentation/grandf_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 1000,
        },
        "gcg_refcocog": {
            "data_dir": "",
            "jsonl_path": "segmentation/refcocog_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 19327,
        },
        "gcg_psg": {
            "data_dir": "",
            "jsonl_path": "segmentation/PSG_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 27786,
        },
        "gcg_flickr": {
            "data_dir": "",
            "jsonl_path": "segmentation/flickr_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 148157,
        },
        "Cityscapes_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/cityscapes_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 2094,
        },
        "Hypersim_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/Hypersim_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 30352,
        },
        "Entity_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/entityv2_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 6137,
        },
        "IDDA_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/IDDAv3_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 1084,
        },
        "51World_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/51World_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 3522,
        },
        "StreetHazards_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/StreetHazards_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 966,
        },
        "KITTI_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/KITTI_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 3490,
        },
        "TAS500_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/TAS500_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 290,
        },
        "UDD5_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/UDD5_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 43,
        },
        "UDD6_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/UDD6_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 58,
        },
        "LoveDA_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/LOVEDA_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 1550,
        },
        "VIPSeg_panoptic_gcg_recaption": {
            "data_dir": "",
            "jsonl_path": "segmentation/VIPSeg_total_unified_gcg_edit_regenerated.jsonl",
            "num_total_samples": 59524,
        },
    },
    "seg_sft": {
        "refcoco_train": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 202181,
        },
        "refcoc+_train": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 201534,
        },
        "refcocog_train": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 141998,
        },
        "refclef_train": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 220665,
        },
        "grefcoco_train": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 353347,
        },
        "rea_train": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 132600,
        },
        "coco_interactive_psalm": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3433232,
        },
        "DOORS": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 30105,
        },
        "NDISPark": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 112,
        },
        "MinneApple": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 497,
        },
        "EYTH": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 330,
        },
        "PST900": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 345,
        },
        "PSTRGB": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1426,
        },
        "SUIM": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 4510,
        },
        "MyFood": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1000,
        },
        "CO-SKEL": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 348,
        },
        "VIS2022": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 97318,
        },
        "MVTecD2S": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 4470,
        },
        "VizWiz-FewShot": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 6386,
        },
        "Trans10K": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 5094,
        },
        "CIHP": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 160993,
        },
        "ATR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 91075,
        },
        "LIP": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 146916,
        },
        "FAT-single": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 26897,
        },
        "FAT-mixed": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 127200,
        },
        "Fashionpedia": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 125976,
        },
        "PartImageNet": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 51626,
        },
        "PartImageNet-Whole": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 493713,
        },
        "WaterOVS": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 5690,
        },
        "RaidaR-rainy": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1510,
        },
        "RaidaR-sunny": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 939,
        },
        "FSS-1000": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 9956,
        },
        "DAVIS": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3968,
        },
        "OCID": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 6497,
        },
        "PIC": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 53960,
        },
        "LaPa": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 46998,
        },
        "DeepFashion2": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 308056,
        },
        "MattingHumanHalf": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 34426,
        },
    },
    "vlm_sft": {
        "llava_v1_5": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 624254,
        },
        "finevision_image": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 5893201,
        },
        "finevision_multi_image": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 131932,
        },
        "mammoth_image": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8956368,
        },
        "mammoth_text": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 921269,
        },
        "finevision_text": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 9426088,
        },
    },
    "t2i_general": {
        "BLIP3o-Pretrain-Long-Caption": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 6847620,
        },
        "BLIP3o-Pretrain-Short-Caption": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2386046,
        },
        "BLIP3o-Long-part2": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 12809032,
        },
    },
    "general_edit": {
        "Nano-consistent-150k": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 123268,
        },
        "multi_edit": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 106532,
        },
        "ShareGPT_4o_edit": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 46539,
        },
        "GPT_Image_Edit_OmniEdit": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1270385,
        },
        "GPT_Image_Edit_HQEdit": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 183182,
        },
        "GPT_Image_Edit_UltraEdit": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 100008,
        },
    },
    "cv_depth": {
        "hypersim_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 61432,
        },
        "vkitti_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 17008,
        },
        "IRS_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 24614,
        },
        "tartanair_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 42521,
        },
        "tartanair_addition_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 75514,
        },
        "IRS_addition_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 49236,
        },
        "scenenet_rgbd_depth": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2273316,
        },
        "taskonomy_depth": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/taskonomy_total_unified_depth_edit_regenerated.jsonl",
            "num_total_samples": 2000000,
        },
        "scannetpp_depth": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/scannetpp_total_unified_depth_edit_regenerated.jsonl",
            "num_total_samples": 812765,
        },
        "coco_depth": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/coco_total_unified_depth_edit_regenerated.jsonl",
            "num_total_samples": 77870,
        },
        "sa_1b_depth": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/sa_1b_total_unified_depth_edit_regenerated.jsonl",
            "num_total_samples": 4441957,
        },
        "object365_depth": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/object365_total_unified_depth_edit_regenerated.jsonl",
            "num_total_samples": 1317199,
        },
    },
    "cv_normal": {
        "hypersim_normal": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 72655,
        },
        "interiorverse_normal": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 26557,
        },
        "IRS_normal": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 34439,
        },
        "tartanair_normal": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 117855,
        },
        "IRS_addition_normal": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 68877,
        },
        "coco_normal": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/coco_total_unified_normal_edit_regenerated.jsonl",
            "num_total_samples": 78801,
        },
        "SA_1B_normal": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/sa_1b_total_unified_normal_edit_regenerated.jsonl",
            "num_total_samples": 4462151,
        },
        "scenenet_rgbd_normal": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2379750,
        },
        "scannetpp_normal": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/scannetpp_total_unified_normal_edit_regenerated.jsonl",
            "num_total_samples": 812765,
        },
        "taskonomy_normal": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/taskonomy_total_unified_normal_edit_regenerated.jsonl",
            "num_total_samples": 2000000,
        },
        "object365_normal": {
            "data_dir": "",
            "jsonl_path": "dense_geometric_prediction/object365_total_unified_normal_edit_regenerated.jsonl",
            "num_total_samples": 1317199,
        },
    },
    "cv_detection_bbox": {
        "grounding_SA1B": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/sa_1b_total_unified_bbox_und.jsonl",
            "num_total_samples": 3119384,
        },
        "APT_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 28471,
        },
        "DeepFashion_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 191961,
        },
        "EgoObjects_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 78895,
        },
        "HumanParts_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 12000,
        },
        "ImageNetPart_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 16523,
        },
        "Objects365_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1742289,
        },
        "PACO_LVIS_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 45790,
        },
        "V3Det_ovd_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 116182,
        },
    },
    "cv_detection_point": {
        "pixmo_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1704004,
        },
        "SA1B_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/sa_1b_total_unified_point_und.jsonl",
            "num_total_samples": 1949930,
        },
        "Objects365_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/object365_total_unified_point_und.jsonl",
            "num_total_samples": 1077215,
        },
        "APT_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/APTv2_total_unified_point_und.jsonl",
            "num_total_samples": 14870,
        },
        "DeepFashion_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/DeepFashion_total_unified_point_und.jsonl",
            "num_total_samples": 113232,
        },
        "EgoObjects_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/EgoObjects_total_unified_point_und.jsonl",
            "num_total_samples": 49963,
        },
        "HumanParts_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/HumanParts_total_unified_point_und.jsonl",
            "num_total_samples": 7427,
        },
        "ImageNetPart_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/ImageNetPart_total_unified_point_und.jsonl",
            "num_total_samples": 10365,
        },
        "PACO_LVIS_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/PACO_total_unified_point_und.jsonl",
            "num_total_samples": 27037,
        },
        "V3Det_ovd_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/V3Detovd_total_unified_point_und.jsonl",
            "num_total_samples": 61159,
        },
    },
    "cv_detection_point_dense": {
        "BDD100K_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/BDD100K_total_unified_point_und.jsonl",
            "num_total_samples": 68502,
        },
        "DOTAv2_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/DOTAv2_total_unified_point_und.jsonl",
            "num_total_samples": 1700,
        },
        "FAIR1M_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/FAIR1M_total_unified_point_und.jsonl",
            "num_total_samples": 16363,
        },
        "NuImages_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/NuImages_total_unified_point_und.jsonl",
            "num_total_samples": 55671,
        },
        "VisDrone_pointing": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/VisDrone_total_unified_point_und.jsonl",
            "num_total_samples": 6411,
        },
        "FSC147_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3659,
        },
    },
    "cv_detection_bbox_dense": {
        "Locount_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 34021,
        },
        "LVIS_Fruits_And_Vegetables_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 6721,
        },
        "Shoes_data_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 135,
        },
        "pixmo_detect": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/pixmo_total_unified_bbox_und_regenerated.jsonl",
            "num_total_samples": 121,
        },
        "blood_cell_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 255,
        },
        "sheep_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3602,
        },
        "pill_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 51,
        },
        "WiderPerson_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 9000,
        },
        "METU_ALET_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2088,
        },
        "homeobjects_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2285,
        },
        "football_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 858,
        },
        "FiftyOne_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8001,
        },
        "CrowdHuman_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3484,
        },
        "GroceryStore_detect": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/GroceryStore_total_unified_bbox_und.jsonl",
            "num_total_samples": 1844,
        },
        "Industrial_Site_Safety_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 327,
        },
        "TinyPerson_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 745,
        },
        "TinyPerson_dense_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 793,
        },
        "S2TLD_dense_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 5037,
        },
        "BDD100K_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 70000,
        },
        "DOTAv2_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1824,
        },
        "FAIR1M_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 16488,
        },
        "NuImages_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 60668,
        },
        "VisDrone_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 6465,
        },
        "CARPK_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1098,
        },
        "owdod_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8001,
        },
        "SKU110k_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 28264,
        },
        "FSC147_detect": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/FSC147_total_unified_bbox_und.jsonl",
            "num_total_samples": 1814,
        },
    },
    "cv_detection_bbox_referring": {
        "openimages_refbbox_merge": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/openimages_total_unified_refbbox_und_merge.jsonl",
            "num_total_samples": 2017823,
        },
        "object365_refbbox_merge": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/object365_total_unified_refbbox_und_merge.jsonl",
            "num_total_samples": 2026308,
        },
        "humanref_ref_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 37057,
        },
        "refcoco_ref_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 120624,
        },
        "refcoco_plus_ref_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 120191,
        },
        "refcocog_ref_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 80506,
        },
        "rexverse_onesentense_ref_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 415050,
        },
        "rexverse_referring_ref_detect": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 415050,
        },
    },
    "cv_detection_point_referring": {
        "grounding_humanref_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 27363,
        },
        "grounding_refcoco_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 99265,
        },
        "grounding_refcoco_plus_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 98875,
        },
        "grounding_refcocog_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 62567,
        },
        "grounding_rexverse_onesentense_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 401200,
        },
        "grounding_rexverse_referring_pointing": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 401200,
        },
        "openimages_refpoint_merge": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/openimages_total_unified_refpoint_und_merge.jsonl",
            "num_total_samples": 2017823,
        },
        "object365_refpoint_merge": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/object365_total_unified_refpoint_und_merge.jsonl",
            "num_total_samples": 1563542,
        },
    },
    "cv_detection_ocr_text_box": {
        "Hiertext_text_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8281,
        },
        "icdar2019_text_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2757,
        },
        "LSVT2019_text_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 29992,
        },
        "RCTW_text_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8020,
        },
        "ReCTS_text_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 19998,
        },
        "SROIE_text_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 626,
        },
        "wildreceipt_text_bbox": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1267,
        },
        "blip3_ocr_200m_text_bbox": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/blip3_total_unified_ocr_text_box_und.jsonl",
            "num_total_samples": 399885,
        },
        "mtwi_text_bbox": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 10045,
        },
    },
    "cv_detection_ocr_text_poly": {
        "Hiertext_text_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8281,
        },
        "icdar2019_text_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2757,
        },
        "LSVT2019_text_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 29992,
        },
        "ReCTS_text_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 19998,
        },
        "blip3_ocr_200m_text_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/blip3_total_unified_ocr_text_poly_und.jsonl",
            "num_total_samples": 399885,
        },
        "mtwi_text_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 10045,
        },
    },
    "cv_detection_ocr_word_box": {
        "Hiertext_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8281,
        },
        "icdar2013_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 229,
        },
        "icdar2015_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 979,
        },
        "icdar2019_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2846,
        },
        "LSVT2019_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 29992,
        },
        "RCTW_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8020,
        },
        "ReCTS_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 19998,
        },
        "TextOCR_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 21778,
        },
        "blip3_ocr_200m_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/blip3_total_unified_ocr_world_box_und.jsonl",
            "num_total_samples": 392141,
        },
        "SynthText_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 858750,
        },
        "mtwi_word_bbox_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 10045,
        },
    },
    "cv_detection_ocr_word_poly": {
        "Hiertext_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8281,
        },
        "icdar2015_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 979,
        },
        "icdar2019_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2846,
        },
        "LSVT2019_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 29992,
        },
        "ReCTS_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 19998,
        },
        "TextOCR_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 21778,
        },
        "blip3_ocr_200m_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/blip3_total_unified_ocr_world_poly_und.jsonl",
            "num_total_samples": 392141,
        },
        "SynthText_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 858750,
        },
        "mtwi_word_poly_OCR": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 10045,
        },
    },
    "cv_detection_visual": {
        "LVIS_Fruits_And_Vegetables_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 11138,
        },
        "Locount_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 34021,
        },
        "blood_cell_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 255,
        },
        "sheep_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3602,
        },
        "pill_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 51,
        },
        "WiderPerson_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 9000,
        },
        "METU_ALET_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2088,
        },
        "homeobjects_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2285,
        },
        "football_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 858,
        },
        "FiftyOne_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8001,
        },
        "CrowdHuman_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3484,
        },
        "GroceryStore_visual": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/GroceryStore_total_unified_visual_und.jsonl",
            "num_total_samples": 1180,
        },
        "Industrial_Site_Safety_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 327,
        },
        "TinyPerson_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 745,
        },
        "S2TLD_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 5037,
        },
        "CARPK_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1098,
        },
        "owdod_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 8001,
        },
        "SKU110k_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 28264,
        },
        "BDD100K_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 70000,
        },
        "DOTAv2_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1824,
        },
        "FAIR1M_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 16488,
        },
        "VisDrone_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 6465,
        },
        "fish_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 628,
        },
        "Objects365_visual": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 2268587,
        },
        "SA_1B_visual": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/sa_1b_total_unified_visual_und.jsonl",
            "num_total_samples": 3116383,
        },
        "FSC147_visual": {
            "data_dir": "",
            "jsonl_path": "structure_view_understanding/FSC147_total_unified_visual_und.jsonl",
            "num_total_samples": 1160,
        },
    },
    "cv_detection_keypoint": {
        "ap-10k_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 9741,
        },
        "APT36k_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 35402,
        },
        "coco2017_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 56599,
        },
        "crowdpose_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 10000,
        },
        "Human-Art_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 33249,
        },
        "macaquepose_v1_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1301,
        },
        "mpii_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 17408,
        },
        "ochuman_keypoint": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 3293,
        },
    },
    "cv_detection_layout": {
        "CDLA_Layout": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 5000,
        },
        "DocLayNet_core_Layout": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 69103,
        },
        "publaynet_Layout": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 335703,
        },
        "TableBank_Layout": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 260582,
        },
        "TabRecSet_Layout": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 32072,
        },
    },
    "cv_detection_gui": {
        "OS-Atlas-data_desktop_domain_GUI": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1136719,
        },
        "OS-Atlas-data_mobile_domain_GUI": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 1309167,
        },
        "OS-Atlas-data_rico_GUI": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 101426,
        },
        "OS-Atlas-data_web_domain_GUI": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 12962897,
        },
        "ShowUI-desktop_GUI": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 7496,
        },
        "ui_refexp_GUI": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 15624,
        },
    },
    "recon3d_jsonl_sft": {
        "recon3d_jsonl_dl3dv": {
            "jsonl_path": "multiview_visual_geometry/dl3dv_total_unified_recon_edit_regenerated.jsonl",
            "data_dir": {
                "image_dir": "",
                "depth_dir": "SenseNova-Vision-Corpus-50M/",
                "camera_dir": "SenseNova-Vision-Corpus-50M/",
                "depth_scale": 0.01,
            },
            "num_total_samples": 10128,
        },
        "recon3d_jsonl_scannetpp": {
            "jsonl_path": "multiview_visual_geometry/scannetpp_total_unified_recon_edit_regenerated.jsonl",
            "data_dir": {
                "image_dir": "",
                "depth_dir": "SenseNova-Vision-Corpus-50M/",
                "camera_dir": "SenseNova-Vision-Corpus-50M/",
                "depth_scale": 0.001,
            },
            "num_total_samples": 807,
        },
        "recon3d_jsonl_scannetv2": {
            "jsonl_path": "multiview_visual_geometry/scannetv2_total_unified_recon_edit_regenerated.jsonl",
            "data_dir": {
                "image_dir": "",
                "depth_dir": "SenseNova-Vision-Corpus-50M/",
                "camera_dir": "SenseNova-Vision-Corpus-50M/",
                "depth_scale": 0.001,
            },
            "num_total_samples": 1502,
        },
        "recon3d_jsonl_wildrgbd": {
            "jsonl_path": "multiview_visual_geometry/wildrgbd_total_unified_recon_edit_regenerated.jsonl",
            "data_dir": {
                "image_dir": "",
                "depth_dir": "SenseNova-Vision-Corpus-50M/",
                "camera_dir": "SenseNova-Vision-Corpus-50M/",
                "depth_scale": 0.001,
            },
            "num_total_samples": 23049,
        },
    },
    "vlm_sft_image": {
        "scannetv2_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 134692,
        },
        "hypersim_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 17937,
        },
        "tartanair_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 246981,
        },
        "IRS_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 10330,
        },
        "scannetpp_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 158390,
        },
        "objaversev1_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 126487,
        },
        "co3dv2_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 184658,
        },
        "scenenet_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 83734,
        },
        "OmniObject3D_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 28820,
        },
        "MvsSynth_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 326,
        },
        "Megasynth_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 33544,
        },
        "Demon_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 434,
        },
        "Dl3dv_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 150854,
        },
        "Wildrgbd_camera_pose_estimate_Spec": {
            "data_dir": "",
            "jsonl_path": "",
            "num_total_samples": 170573,
        },
    },
}
