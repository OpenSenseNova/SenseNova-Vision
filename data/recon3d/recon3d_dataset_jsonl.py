import json
import os
import traceback

import numpy as np
from PIL import Image, ImageFile, PngImagePlugin
from scipy.spatial.transform import Rotation

from ..data_utils import pil_img2rgb
from ..interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset
from .mixed_path import MixedPath
from .seq_sampler import SeqSamplerBase


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class Recon3DJsonLIterableDataset(InterleavedBaseIterableDataset):
    """Recon3D iterable dataset backed by scene-level JSONL files.

    Each JSONL line is a scene:
      id, image, depth, trajectory, conversations

    The dataset loads scene rows during init. A worker samples one scene row,
    samples a short frame sequence, and builds pointmaps from depth + camera
    txt files.
    """

    # ------------------------------------------------------------------
    # Initialization and dataset metadata
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        pointmap_vae_transform,
        pointmap_vit_transform,
        jsonl_path_list,
        data_dir_list,
        dataset_info_list=None,
        num_used_data=None,
        sample_weights=None,
        epoch_size=None,
        seq_num_frames_range=(2, 10),
        expand_ratio=16,
        seq_norm_mean="minmaxcenter_xyz",
        seq_norm_std="maxabs_scalar",
        use_single_pointmap_split=True,
        max_resample_attempts=3,
        depth_max=80.0,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        InterleavedBaseIterableDataset.__init__(
            self,
            dataset_name,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
        )
        self.transform = transform
        self.vit_transform = vit_transform
        self.pointmap_vae_transform = pointmap_vae_transform
        self.pointmap_vit_transform = pointmap_vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        self.jsonl_path_list = list(jsonl_path_list)
        self.data_dir_list = list(data_dir_list)
        self.dataset_info_list = self._normalize_dataset_info(
            dataset_info_list,
            jsonl_path_list,
            data_dir_list,
        )
        self.sample_weights = sample_weights
        self.epoch_size = epoch_size
        self.use_single_pointmap_split = use_single_pointmap_split
        self.max_resample_attempts = int(max_resample_attempts)
        self.depth_max = float(depth_max)
        if isinstance(expand_ratio, (list, tuple)):
            raise ValueError("Recon3DJsonLIterableDataset expects a unified scalar expand_ratio")
        self._seq_sampler_args = dict(
            seq_num_frames_range=list(seq_num_frames_range),
            expand_ratio=expand_ratio,
            seq_norm_mean=seq_norm_mean,
            seq_norm_std=seq_norm_std,
        )

        self._rows_per_dataset = self._load_all_rows(num_used_data)
        self._paths_per_dataset = [
            [(dataset_id, line_no) for line_no in range(len(rows))]
            for dataset_id, rows in enumerate(self._rows_per_dataset)
        ]
        self.sample_weights = sample_weights

        if sample_weights is not None:
            if len(sample_weights) != len(self._rows_per_dataset):
                raise ValueError(
                    f"sample_weights length {len(sample_weights)} does not match "
                    f"dataset count {len(self._rows_per_dataset)}"
            )
            self.data_paths = None
        else:
            self.data_paths = []
            for paths in self._paths_per_dataset:
                self.data_paths.extend(paths)
        self.set_epoch()

    @staticmethod
    def _normalize_dataset_info(dataset_info_list, jsonl_path_list, data_dir_list):
        info_list = []
        source_list = dataset_info_list
        if source_list is None:
            source_list = [{"data_dir": data_dir} for data_dir in data_dir_list]

        for item, jsonl_path, data_dir in zip(source_list, jsonl_path_list, data_dir_list):
            info = dict(item)
            info.setdefault("data_dir", data_dir)
            if isinstance(info["data_dir"], dict):
                roots = dict(info["data_dir"])
                info.update(roots)
                info["data_dir"] = roots["image_dir"]
            info["jsonl_path"] = jsonl_path
            info_list.append(info)
        return info_list

    # ------------------------------------------------------------------
    # JSONL loading and scene-row resampling
    # ------------------------------------------------------------------

    def _load_all_rows(self, num_used_data):
        if num_used_data is None:
            limits = [None] * len(self.jsonl_path_list)
        else:
            if len(num_used_data) != len(self.jsonl_path_list):
                raise ValueError(
                    f"num_used_data length {len(num_used_data)} does not match "
                    f"jsonl count {len(self.jsonl_path_list)}"
                )
            limits = [None if item is None else int(item) for item in num_used_data]

        rows_per_dataset = []
        for dataset_id, (jsonl_path, limit) in enumerate(zip(self.jsonl_path_list, limits)):
            rows = self._load_jsonl_rows(jsonl_path, limit)
            if not rows:
                raise ValueError(f"no valid JSONL lines found in {jsonl_path}")
            rows_per_dataset.append(rows)
            if self.local_rank == 0:
                print(
                    f"[recon3d_jsonl] loaded {len(rows):,} rows from {jsonl_path}"
                )
        return rows_per_dataset

    @staticmethod
    def _load_jsonl_rows(jsonl_path, limit):
        jsonl_path = str(jsonl_path)
        rows = []
        with MixedPath(jsonl_path).open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    def get_data_paths(self, *args, **kwargs):
        return self.data_paths

    # ------------------------------------------------------------------
    # Data-plan helpers
    # ------------------------------------------------------------------

    def _add_pointmap(self, data, pointmap, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = self.pointmap_vae_transform(pointmap)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.pointmap_vae_transform.stride ** 2
            data["image_tensor_list"].append(image_tensor)

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = self.pointmap_vae_transform(pointmap)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.pointmap_vae_transform.stride ** 2
            data["image_tensor_list"].append(image_tensor.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            )
            vit_image_tensor = self.pointmap_vit_transform(pointmap)
            height, width = vit_image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.pointmap_vit_transform.stride ** 2
            data["image_tensor_list"].append(vit_image_tensor)

        return data

    def _add_pointmap_seq(self, data, pointmap_seq, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, pointmap in enumerate(pointmap_seq):
                data["sequence_plan"].append(
                    {
                        "type": "vae_image",
                        "enable_cfg": 0,
                        "loss": 1,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "split_start": idx == 0,
                        "split_end": idx + 1 == len(pointmap_seq),
                        "frame_delta": 1,
                    }
                )
                image_tensor = self.pointmap_vae_transform(pointmap)
                height, width = image_tensor.shape[1:]
                data["num_tokens"] += width * height // self.pointmap_vae_transform.stride ** 2
                data["image_tensor_list"].append(image_tensor)

        if need_vae:
            for idx, pointmap in enumerate(pointmap_seq):
                data["sequence_plan"].append(
                    {
                        "type": "vae_image",
                        "enable_cfg": int(enable_cfg),
                        "loss": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                        "split_start": idx == 0,
                        "split_end": idx + 1 == len(pointmap_seq),
                        "frame_delta": 1,
                    }
                )
                image_tensor = self.pointmap_vae_transform(pointmap)
                height, width = image_tensor.shape[1:]
                data["num_tokens"] += width * height // self.pointmap_vae_transform.stride ** 2
                data["image_tensor_list"].append(image_tensor.clone())

        return data

    def _make_data_plan(self, conversations, raw_images, all_pointmaps):
        data_plan = self._init_data()

        for conversation in conversations:
            if conversation["from"] == "human":
                text_list = conversation["value"].split("<image>")
                for idx, text in enumerate(text_list):
                    if text.strip():
                        self._add_text(
                            data_plan,
                            text.strip(),
                            need_loss=False,
                            enable_cfg=True,
                        )
                    if idx < len(text_list) - 1 and idx < len(raw_images):
                        self._add_image(
                            data_plan,
                            raw_images[idx],
                            need_loss=False,
                            need_vae=True,
                            need_vit=True,
                            enable_cfg=True,
                        )
            elif conversation["from"] == "gpt":
                if "<pointmap_seq>" in conversation["value"]:
                    assert "<pointmap>" not in conversation["value"]
                    text_list = conversation["value"].split("<pointmap_seq>")
                    for idx, text in enumerate(text_list):
                        if text.strip():
                            self._add_text(
                                data_plan,
                                text.strip(),
                                need_loss=True,
                                enable_cfg=True,
                            )
                        if idx == 0:
                            self._add_pointmap_seq(
                                data_plan,
                                all_pointmaps,
                                need_loss=True,
                                need_vae=False,
                                enable_cfg=False,
                            )
                else:
                    text_list = conversation["value"].split("<pointmap>")
                    num_pointmaps_to_add = min(len(text_list) - 1, len(all_pointmaps))
                    for idx, text in enumerate(text_list):
                        if text.strip():
                            self._add_text(
                                data_plan,
                                text.strip(),
                                need_loss=True,
                                enable_cfg=True,
                            )
                        if idx < num_pointmaps_to_add - 1:
                            self._add_pointmap(
                                data_plan,
                                all_pointmaps[idx],
                                need_loss=True,
                                need_vae=True,
                                need_vit=True,
                                enable_cfg=True,
                            )
                        elif idx == num_pointmaps_to_add - 1:
                            self._add_pointmap(
                                data_plan,
                                all_pointmaps[idx],
                                need_loss=True,
                                need_vae=False,
                                need_vit=False,
                                enable_cfg=False,
                            )
        return data_plan

    # ------------------------------------------------------------------
    # Frame-sequence sampling
    # ------------------------------------------------------------------

    def _make_seq_sampler(self, worker_id, dataset_id=None):
        dataset_seed = 0 if dataset_id is None else (dataset_id + 1) * 9176
        seed = (
            2025
            + self.local_rank * 100003
            + worker_id * 1009
            + dataset_seed
        )
        return SeqSamplerBase(
            **self._seq_sampler_args,
            rng=np.random.default_rng(seed),
        )

    def _choose_frame_indices(self, seq_sampler, num_frames):
        frame_ids = np.arange(num_frames, dtype=np.int64)
        for _ in range(self.max_resample_attempts):
            frame_seq = seq_sampler.choose_frame_seq(frame_ids)
            if frame_seq is not None:
                return [int(item) for item in frame_seq]
        return None

    # ------------------------------------------------------------------
    # Scene-row reading and path resolution
    # ------------------------------------------------------------------

    def _read_scene_row(self, dataset_id, line_no):
        return self._rows_per_dataset[dataset_id][line_no]

    @staticmethod
    def _is_absolute_path(path):
        return os.path.isabs(path)

    @classmethod
    def _join_root(cls, root, path):
        if path is None:
            raise ValueError("path must not be None")
        if cls._is_absolute_path(path):
            return MixedPath(path)
        if root in (None, ""):
            raise ValueError(f"relative path {path!r} requires a non-empty root")
        return MixedPath(root) / path

    # ------------------------------------------------------------------
    # RGB/depth/camera loading and pointmap construction
    # ------------------------------------------------------------------

    def _load_scene_sequence(self, dataset_id, row, frame_indices):
        """Load RGB, depth and camera txt, then express pointmaps in frame-0 camera coords."""
        info = self.dataset_info_list[dataset_id]
        image_root = info["data_dir"]
        depth_root = info["depth_dir"]
        camera_root = info["camera_dir"]
        depth_scale = float(info.get("depth_scale", 1.0))
        image_paths = row["image"]
        depth_paths = row["depth"]
        camera_paths = row["trajectory"]

        camera_params = []
        for frame_idx in frame_indices:
            camera_path = self._join_root(camera_root, camera_paths[frame_idx])
            with camera_path.open("r", encoding="utf-8") as file:
                values = np.fromstring(file.read(), sep=" ", dtype=np.float32)

            fx, fy, cx, cy = values[:4]
            quat = values[4:8]
            c2w_t = values[8:11].astype(np.float32)
            intrinsic = np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            c2w_R = Rotation.from_quat(quat).as_matrix().astype(np.float32)
            camera_params.append((intrinsic, c2w_R, c2w_t))

        def load_depth(depth_path):
            path_str = str(depth_path)
            candidates = [depth_path]
            if path_str.endswith(".png.npy"):
                candidates.append(MixedPath(path_str[: -len(".npy")]))
            elif path_str.endswith(".npy"):
                candidates.append(MixedPath(path_str[: -len(".npy")] + ".png"))

            for candidate in candidates:
                if candidate.exists():
                    depth_path = candidate
                    break
            else:
                tried = ", ".join(str(candidate) for candidate in candidates)
                raise FileNotFoundError(f"depth file not found; tried: {tried}")

            suffix = str(depth_path).lower()
            if suffix.endswith(".npy"):
                with depth_path.open("rb") as file:
                    depth = np.load(file, allow_pickle=False)
            elif suffix.endswith(".png"):
                with depth_path.open("rb") as file:
                    depth = np.asarray(Image.open(file))
                if depth.dtype != np.uint16:
                    raise ValueError(
                        f"16-bit png depth must have uint16 dtype, got {depth.dtype} "
                        f"from {depth_path}"
                    )
                depth = depth.astype(np.float32) * depth_scale
            else:
                raise ValueError(f"unsupported depth file format: {depth_path}")

            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2:
                raise ValueError(
                    f"expected 2D depth, got shape {depth.shape} from {depth_path}"
                )
            return depth

        def depth_to_camera_points(depth, intrinsic):
            height, width = depth.shape
            xs = np.arange(width, dtype=np.float32)[None, :]
            ys = np.arange(height, dtype=np.float32)[:, None]
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            cam_points = np.empty((height, width, 3), dtype=np.float32)
            valid_mask = (depth > 0) & (depth < self.depth_max)
            depth = depth.astype(np.float32, copy=True)
            depth[~valid_mask] = self.depth_max
            cam_points[..., 0] = ((xs - cx) / fx) * depth
            cam_points[..., 1] = ((ys - cy) / fy) * depth
            cam_points[..., 2] = depth
            return cam_points, valid_mask

        _, cam0_c2w_R, cam0_c2w_t = camera_params[0]
        seq = []
        raw_images = []
        for frame_idx, (intrinsic, c2w_R, c2w_t) in zip(frame_indices, camera_params):
            image_path = self._join_root(image_root, image_paths[frame_idx])
            with image_path.open("rb") as file:
                raw_images.append(pil_img2rgb(Image.open(file)))

            depth_path = self._join_root(depth_root, depth_paths[frame_idx])
            depth = load_depth(depth_path)
            cam_points, valid_mask = depth_to_camera_points(depth, intrinsic)

            world_points = cam_points @ c2w_R.T + c2w_t
            new_world_points = (world_points - cam0_c2w_t) @ cam0_c2w_R
            new_c2w_R = cam0_c2w_R.T @ c2w_R
            new_c2w_t = cam0_c2w_R.T @ (c2w_t - cam0_c2w_t)
            if not np.isfinite(new_world_points).all():
                raise ValueError(f"non-finite pointmap in scene {row.get('id')}")

            seq.append(
                {
                    "frame_id": int(frame_idx),
                    "world_points": new_world_points.astype(np.float32),
                    "valid_mask": valid_mask,
                    "c2w_R": new_c2w_R.astype(np.float32),
                    "c2w_t": new_c2w_t.astype(np.float32),
                }
            )
        return raw_images, seq

    # ------------------------------------------------------------------
    # Prompt construction and iterable entrypoint
    # ------------------------------------------------------------------

    def _build_conversations(self, row, num_frames, rng):
        prompt_candidates = [
            item.get("value", "").strip()
            for item in row["conversations"]
            if item.get("from") == "human" and item.get("value", "").strip()
        ]
        if not prompt_candidates:
            raise ValueError(f"scene {row.get('id')} has no human prompt in conversations")
        prompt = prompt_candidates[int(rng.integers(0, len(prompt_candidates)))]
        return [
            {
                "from": "human",
                "value": ("<image>" * num_frames) + "\n" + prompt,
            },
            {
                "from": "gpt",
                "value": "<pointmap_seq>"
                if self.use_single_pointmap_split
                else ("<pointmap>" * num_frames),
            },
        ]

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        seq_samplers = {}
        prompt_rng = np.random.default_rng(
            4049 + self.local_rank * 100003 + worker_id * 1009
        )
        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (dataset_id, line_no) in enumerate(
                data_paths_per_worker_,
                start=row_start_id,
            ):
                try:
                    row = self._read_scene_row(dataset_id, line_no)
                    num_frames = len(row["image"])
                    if dataset_id not in seq_samplers:
                        seq_samplers[dataset_id] = self._make_seq_sampler(
                            worker_id,
                            dataset_id=dataset_id,
                        )
                    seq_sampler = seq_samplers[dataset_id]
                    frame_indices = self._choose_frame_indices(
                        seq_sampler,
                        num_frames,
                    )
                    if frame_indices is None:
                        print(
                            f"recon3d_jsonl skipping scene with too few neighbors: "
                            f"{row.get('id')}"
                        )
                        continue

                    raw_images, frame_seq = self._load_scene_sequence(
                        dataset_id,
                        row,
                        frame_indices,
                    )
                    frame_seq = seq_sampler.normalize_sequence(frame_seq)
                    conversations = self._build_conversations(
                        row,
                        len(frame_indices),
                        prompt_rng,
                    )
                except Exception:
                    traceback.print_exc()
                    continue

                data_plan = self._make_data_plan(
                    conversations,
                    raw_images,
                    [frame["world_points"] for frame in frame_seq],
                )
                data_plan["data_indexes"] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                    "source_dataset_id": dataset_id,
                    "source_line_no": line_no,
                    "scene_id": row["id"],
                    "frame_indices": frame_indices,
                    "sampled_frame_ids": frame_indices,
                }
                yield data_plan

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
