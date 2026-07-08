import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from huggingface_hub import snapshot_download
from PIL import Image

try:
    from ..data.transforms import ImageTransform
    from .inferencer import InterleaveInferencer
    from ..modeling.autoencoder import load_ae
    from ..modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from ..modeling.qwen2 import Qwen2Tokenizer
    from ..data.data_utils import add_special_tokens, pil_img2rgb
    from .utils_3d import resolve_pose_string

except ImportError:
    from data.data_utils import add_special_tokens, pil_img2rgb
    from data.transforms import ImageTransform
    from inference.inferencer import InterleaveInferencer
    from inference.utils_3d import resolve_pose_string
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer


BASE_PARAMS: Dict[str, Dict[str, Any]] = {
    "generate": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    ),
    "think_generate": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        think=True,
    ),
    "caption_generate": dict(
        max_think_token_n=8192,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        caption=True,
    ),
    "dense_perception": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
    "edit": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
    "think_edit": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        think=True,
    ),
    "understanding": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
    ),
    "think_understanding": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
        think=True,
    ),
    "dense_detection": dict(
        max_think_token_n=8192,
        do_sample=False,
        understanding_output=True,
    ),
    "dense_OCR": dict(
        max_think_token_n=20000,
        do_sample=False,
        understanding_output=True,
    ),
    "recon3d": dict(
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    ),
}


IMAGE_OUTPUT_MODES = {
    "generate",
    "think_generate",
    "caption_generate",
    "dense_perception",
    "edit",
    "think_edit",
}

TEXT_OUTPUT_MODES = {
    "understanding",
    "think_understanding",
    "dense_detection",
    "dense_OCR",
}

RECON3D_OUTPUT_MODES = {
    "recon3d",
}

RECON3D_PROMPT = (
    "Reconstruct a scene from multiple input images and output one dense 3D "
    "coordinate map per view, all aligned to the first camera's perspective."
)
CAMERA_POSE_PROMPT = (
    "With the first frame as the reference frame, output the relative pose of"
    " all subsequent frames (excluding the first frame) with respect to the"
    " first frame, following the input order and adhering to the strict format"
    " below:Rotation: Represented by a quaternion in the format"
    " <quat>[x,y,z,w], enclosed in <quat> tags;Translation: Represented by a"
    " unit vector (direction) in the format <offset>[x,y,z], enclosed in"
    " <offset> tags (the vector has no absolute physical meaning, only"
    " directional information);Scale: Represented by a numerical value in the"
    " format <scale>value</scale> tags, where the value denotes the magnitude"
    " of translation (corresponding to the length of the translation unit"
    " vector);Enclose the result of each frame in <frame> tags, with no extra"
    " characters, spaces, or line breaks outside the tags."
)

def _resolve_torch_dtype(dtype: str) -> Optional[torch.dtype]:
    dtype = dtype.lower()

    if dtype in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if dtype in ["fp16", "float16", "half"]:
        return torch.float16
    if dtype in ["fp32", "float32"]:
        return torch.float32
    if dtype in ["auto", "none"]:
        return None

    raise ValueError(
        f"Unsupported dtype: {dtype}. "
        f"Supported dtypes: bf16, fp16, fp32, auto, nf4, int8."
    )


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SenseNovaVisionModel:
    """Standalone inference wrapper for SenseNova-Vision / Bagel MoT models.

    Features:
    - Does not inherit from any external Model class.
    - Defaults to `sensenova/SenseNova-Vision-7B-MoT`.
    - Defaults to checkpoint file `ema.safetensors`.
    - Uses `dense_perception` instead of `depth_normal_edit`.
    - Does not automatically save output images.
    - Image-output modes return `PIL.Image.Image`.
    - The `understanding` mode returns `str`.

    Expected model directory structure:

        ae.safetensors
        ema.safetensors
        config.json
        generation_config.json
        llm_config.json
        vit_config.json
        tokenizer_config.json
        tokenizer.json
        merges.txt
        vocab.json
    """

    def __init__(
        self,
        model_path: str = "sensenova/SenseNova-Vision-7B-MoT",
        checkpoint_name: str = "ema.safetensors",
        dtype: str = "bf16",
        device: str = "auto",
        max_mem_per_gpu: str = "80GiB",
        offload_folder: str = "offload",
        seed: Optional[int] = 42,
        download_revision: str = "main",
        download_local_files_only: bool = False,
    ):
        self.model_path = self._resolve_model_path(
            model_path=model_path,
            revision=download_revision,
            local_files_only=download_local_files_only,
        )

        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = os.path.join(self.model_path, checkpoint_name)

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please make sure `{checkpoint_name}` exists in {self.model_path}."
            )

        self.dtype = dtype.lower()

        if self.dtype in ["nf4", "int8"]:
            self.torch_dtype = None
        else:
            self.torch_dtype = _resolve_torch_dtype(self.dtype)

        self.device = self._resolve_device(device)
        self.max_mem_per_gpu = max_mem_per_gpu
        self.offload_folder = offload_folder

        print(f"[SenseNova-Vision] model_path      = {self.model_path}")
        print(f"[SenseNova-Vision] checkpoint_path = {self.checkpoint_path}")
        print(f"[SenseNova-Vision] dtype           = {self.dtype}")
        print(f"[SenseNova-Vision] vae/device      = {self.device}")

        (
            model,
            vae_model,
            tokenizer,
            new_token_ids,
            vit_transform,
            vae_transform,
        ) = self._build_model()

        model = self._load_model_weights(model)

        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.vit_transform = vit_transform
        self.vae_transform = vae_transform
        self.recon3d_vae_transform = ImageTransform(512, 256, 16)
        self.recon3d_vit_transform = ImageTransform(448, 224, 14)
        self.camera_vit_transform = ImageTransform(560, 378, 14)

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
            device=self.device,
        )

        set_seed(seed)

        torch.cuda.empty_cache()

    @staticmethod
    def _resolve_model_path(
        model_path: str,
        revision: str = "main",
        local_files_only: bool = False,
    ) -> str:
        local_model_path = Path(model_path)

        if local_model_path.exists() and local_model_path.is_dir():
            print(f"[SenseNova-Vision] Using local model path: {local_model_path}")
            return str(local_model_path)

        print(
            f"[SenseNova-Vision] Local path `{model_path}` does not exist. "
            f"Downloading from Hugging Face..."
        )

        try:
            cache_path = snapshot_download(
                repo_id=model_path,
                repo_type="model",
                revision=revision,
                local_files_only=local_files_only,
            )
            print(f"[SenseNova-Vision] Model downloaded to: {cache_path}")
            return cache_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model `{model_path}`.\n"
                f"Reason: {str(e)}\n"
                f"If your server cannot access Hugging Face, download the repo manually "
                f"and pass the local path to `model_path`."
            ) from e

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _cuda_device_ids(self) -> list[int]:
        if not torch.cuda.is_available() or self.device == "cpu":
            return []

        if self.device.startswith("cuda:"):
            return [int(self.device.split(":", 1)[1])]

        return list(range(torch.cuda.device_count()))

    def _build_model(self):
        # Load LLM config.
        llm_config = Qwen2Config.from_json_file(
            os.path.join(self.model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # Load ViT config.
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(self.model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # Load VAE.
        vae_model, vae_config = load_ae(
            local_path=os.path.join(self.model_path, "ae.safetensors")
        )
        vae_model = vae_model.to(device=self.device).eval()

        # Load tokenizer and add special tokens.
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Prepare image transforms.
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Build Bagel config.
        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=2,
            max_latent_size=64,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)

            model = Bagel(
                language_model,
                vit_model,
                bagel_config,
            )

            # Different Bagel versions may expose slightly different signatures.
            try:
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config,
                    meta=True,
                )
            except TypeError:
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config
                )

        return (
            model,
            vae_model,
            tokenizer,
            new_token_ids,
            vit_transform,
            vae_transform,
        )

    def _infer_device_map(self, model):
        cuda_device_ids = self._cuda_device_ids()

        if cuda_device_ids:
            max_memory = {
                i: self.max_mem_per_gpu
                for i in cuda_device_ids
            }
        else:
            max_memory = {
                "cpu": "128GiB",
            }

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=[
                "Bagel",
                "Qwen2MoTDecoderLayer",
            ],
        )

        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]

        if len(cuda_device_ids) == 1:
            first_device = device_map.get(
                same_device_modules[0],
                f"cuda:{cuda_device_ids[0]}",
            )
            for k in same_device_modules:
                device_map[k] = device_map.get(k, first_device)

        elif len(cuda_device_ids) > 1:
            first_device = device_map.get(same_device_modules[0])
            if first_device is not None:
                for k in same_device_modules:
                    if k in device_map:
                        device_map[k] = first_device

        print("[SenseNova-Vision] device_map:")
        print(device_map)

        return device_map

    def _load_model_weights(self, model):
        device_map = self._infer_device_map(model)

        if self.dtype in [
            "bf16",
            "bfloat16",
            "fp16",
            "float16",
            "half",
            "fp32",
            "float32",
            "auto",
            "none",
        ]:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=self.checkpoint_path,
                device_map=device_map,
                offload_folder=self.offload_folder,
                offload_buffers=True,
                dtype=self.torch_dtype,
                force_hooks=True,
            ).eval()

        elif self.dtype == "nf4":
            from accelerate.utils import (
                BnbQuantizationConfig,
                load_and_quantize_model,
            )

            model = load_and_quantize_model(
                model,
                weights_location=self.checkpoint_path,
                bnb_quantization_config=BnbQuantizationConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map=device_map,
                offload_folder=self.offload_folder,
            ).eval()

        elif self.dtype == "int8":
            from accelerate.utils import (
                BnbQuantizationConfig,
                load_and_quantize_model,
            )

            model = load_and_quantize_model(
                model,
                weights_location=self.checkpoint_path,
                bnb_quantization_config=BnbQuantizationConfig(
                    load_in_8bit=True,
                    torch_dtype=torch.float32,
                ),
                device_map=device_map,
                offload_folder=self.offload_folder,
            ).eval()

        else:
            raise NotImplementedError(f"Unsupported dtype: {self.dtype}")

        print("[SenseNova-Vision] Model loaded.")
        return model

    @staticmethod
    def _load_recon3d_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return pil_img2rgb(image)

        try:
            from data.recon3d.mixed_path import MixedPath
        except ImportError:
            try:
                from ..data.recon3d.mixed_path import MixedPath
            except ImportError:
                MixedPath = Path

        with MixedPath(image).open("rb") as f:
            return pil_img2rgb(Image.open(f))

    @staticmethod
    def _load_postprocess_reconstruction():
        try:
            from inference.utils_3d import postprocess_reconstruction
        except ModuleNotFoundError as e:
            if e.name not in {"inference", "inference.utils_3d"}:
                raise RuntimeError(
                    "Recon3D GLB postprocessing requires optional dependencies. "
                    "Install them with `pip install open3d trimesh`, or call "
                    "`reconstruct_3d(..., postprocess_predictions=False, "
                    "glb_output=None)` to only save/use raw point maps."
                ) from e
            from .utils_3d import postprocess_reconstruction

        return postprocess_reconstruction

    def reconstruct_3d(
        self,
        images: list[Any],
        prompt: str = RECON3D_PROMPT,
        noise_seed: Optional[int] = 123456,
        postprocess_predictions: bool = True,
        max_images: int = 10,
        raw_output: Optional[str] = None,
        glb_output: Optional[str] = None,
        mask_edge: bool = True,
        mask_sky: bool = False,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run multi-view 3D reconstruction.

        Returns:
            {
                "pts3d": np.ndarray,  # [num_input_images, H, W, 3]
                "scene": trimesh.Scene or None,
                "text": str or None,
            }
        """

        if not images:
            raise ValueError("`images` must contain at least one image.")

        if max_images is not None and max_images > 0 and len(images) > max_images:
            print(f"[SenseNova-Vision][Recon3D] limit to {max_images} images")
            images = images[:max_images]

        actual_num_images = len(images)
        inference_images = [
            self._load_recon3d_image(image)
            for image in images
        ]

        if actual_num_images == 1:
            print("[SenseNova-Vision][Recon3D] duplicate the single input image")
            inference_images = inference_images * 2

        inference_params = dict(BASE_PARAMS["recon3d"])
        inference_params.update(kwargs)
        inference_params.update(
            output_multiple_vae=True,
            output_raw_tensor=True,
            return_preprocessed_input=True,
        )

        set_seed(noise_seed)
        output_list, preprocessed_input = self.inferencer.interleave_inference(
            [*inference_images, prompt],
            vae_transform=self.recon3d_vae_transform,
            vit_transform=self.recon3d_vit_transform,
            **inference_params,
        )

        pred_pts3d = []
        text_output = None
        for item in output_list:
            if isinstance(item, np.ndarray):
                pred_pts3d.append(item)
            elif isinstance(item, str):
                text_output = item

        if len(pred_pts3d) < actual_num_images:
            raise RuntimeError(
                "Recon3D prediction count is smaller than input count: "
                f"{len(pred_pts3d)} predictions for {actual_num_images} images."
            )

        pred_pts3d = np.array(
            pred_pts3d[:actual_num_images],
            dtype=np.float32,
        )

        if raw_output is not None:
            np.save(raw_output, pred_pts3d)

        scene_3d = None
        if postprocess_predictions or glb_output is not None:
            preprocessed_images = [
                item
                for item in preprocessed_input
                if isinstance(item, Image.Image)
            ]

            if len(preprocessed_images) < actual_num_images:
                raise RuntimeError(
                    "Recon3D preprocessed image count is smaller than input count: "
                    f"{len(preprocessed_images)} images for {actual_num_images} inputs."
                )

            postprocess_reconstruction = self._load_postprocess_reconstruction()
            scene_3d = postprocess_reconstruction(
                list(pred_pts3d),
                preprocessed_images[:actual_num_images],
                mask_edge=mask_edge,
                mask_sky=mask_sky,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
            )

        if glb_output is not None:
            if scene_3d is None:
                raise RuntimeError("`glb_output` requires postprocessed scene output.")
            scene_3d.export(file_obj=glb_output)

        return {
            "pts3d": pred_pts3d,
            "scene": scene_3d,
            "text": text_output,
        }

    def generate(
        self,
        question: str,
        images: Optional[list[str]] = None,
        mode: Optional[str] = None,
        noise_seed: Optional[int] = 42,
        return_intermediate_outputs: bool = False,
        **kwargs,
    ):
        """Run SenseNova-Vision inference.

        Args:
            question:
                Input prompt.
                Use `<image>` placeholders when image inputs are provided.
                Example:
                    "<image> Estimate the depth map of this image."

            images:
                List of image paths.
                The number of image paths must match the number of `<image>`
                placeholders in `question`.

            mode:
                Supported modes:
                    generate
                    think_generate
                    caption_generate
                    dense_perception
                    edit
                    think_edit
                    understanding
                    think_understanding
                    dense_detection
                    dense_OCR
                    recon3d

            noise_seed:
                Random seed used before inference. Set to None to leave the
                current random state unchanged.

            return_intermediate_outputs:
                For image modes that internally produce text first, such as
                `caption_generate` and `think_generate`, return both the final
                image and the intermediate text in a dict.

            **kwargs:
                Extra inference arguments that override the default parameters
                in BASE_PARAMS.

        Returns:
            - Image-output modes return PIL.Image.Image.
            - The understanding mode returns str.
            - The recon3d mode returns the same dict as `reconstruct_3d`.
            - If `return_intermediate_outputs=True`, this method returns:
                {
                    "image": PIL.Image.Image or None,
                    "text": str or None
                }
        """

        if mode is None:
            raise ValueError(
                f"`mode` must be specified. "
                f"Supported modes: {list(BASE_PARAMS.keys())}"
            )

        if mode not in BASE_PARAMS:
            raise ValueError(
                f"Invalid mode `{mode}`. "
                f"Supported modes: {list(BASE_PARAMS.keys())}"
            )

        if mode in RECON3D_OUTPUT_MODES:
            params = dict(BASE_PARAMS[mode])
            params.update(kwargs)
            return self.reconstruct_3d(
                images=images or [],
                prompt=question or RECON3D_PROMPT,
                noise_seed=noise_seed,
                **params,
            )

        images = images or []
        text_parts = question.split("<image>")

        if len(text_parts) != len(images) + 1:
            raise ValueError(
                f"The number of `<image>` tokens must match the number of images. "
                f"Got {len(text_parts) - 1} image tokens but {len(images)} images."
            )

        input_lists = []

        for i, part in enumerate(text_parts):
            text = part.strip()

            if text:
                input_lists.append(text)

            if i < len(images):
                img_path = images[i]

                try:
                    image = Image.open(img_path).convert("RGB")
                    input_lists.append(image)

                except Exception as e:
                    raise RuntimeError(
                        f"Cannot load image `{img_path}`: {e}"
                    ) from e

        params = dict(BASE_PARAMS[mode])
        params.update(kwargs)

        understanding_output_flag = params.pop(
            "understanding_output",
            False,
        )
        think_flag = params.pop(
            "think",
            False,
        )

        set_seed(noise_seed)
        res = self.inferencer.interleave_inference(
            input_lists=input_lists,
            think=think_flag,
            understanding_output=understanding_output_flag,
            **params,
        )

        ret = {
            "image": [],
            "text": [],
        }

        for item in res:
            if isinstance(item, Image.Image):
                ret["image"].append(item)
            elif isinstance(item, str):
                ret["text"].append(item)

        img_cnt = len(ret["image"])
        txt_cnt = len(ret["text"])

        if return_intermediate_outputs:
            ret["image"] = ret["image"][0] if img_cnt else None
            ret["text"] = ret["text"][0] if txt_cnt else None
            return ret

        if mode in IMAGE_OUTPUT_MODES and img_cnt:
            return ret["image"][0]

        if mode in TEXT_OUTPUT_MODES and txt_cnt:
            return ret["text"][0]

        if img_cnt + txt_cnt != 1:
            print(
                f"[SenseNova-Vision][Warning] mode={mode} returned "
                f"{img_cnt} images and {txt_cnt} texts."
            )

            ret["image"] = ret["image"][0] if img_cnt else None
            ret["text"] = ret["text"][0] if txt_cnt else None

            return ret

        # This fallback should not be reached if mode sets are complete.
        if img_cnt:
            return ret["image"][0]

        return ret["text"][0]


if __name__ == "__main__":
    
    model = SenseNovaVisionModel(
        model_path="sensenova/SenseNova-Vision-7B-MoT",
        dtype="bf16",
    )

    # Example 1: general understanding
    text = model.generate(
        question="<image> What are the main objects in this scene and their relationships?",
        images=["examples/images/1.jpg"],
        mode="understanding",
    )
    
    print(text)

    # Example 2: binary segmentation, returns a PIL.Image.Image
    image = model.generate(
        question="<image> Could you return the binary segmentation masks for the specified categories: <p>person furthest to the right</p>?",
        images=["examples/images/2.jpg"],
        mode="dense_perception",
    )
    image.save("seg.png")


    # Example 3: depth estimation, returns a PIL.Image.Image
    image = model.generate(
        question="<image> Estimate relative depth for each pixel in the image, with closer objects appearing brighter and distant objects appearing darker. Output is a grayscale image with pixel values ranging from 0-255.",
        images=["examples/images/3.jpg"],
        mode="dense_perception",
    )
    
    image.save("depth.png")

    # Example 4: normal estimation, returns a PIL.Image.Image
    image = model.generate(
        question="<image> Generate an RGB normal map where R, G, B channels represent X, Y, Z surface directions. The output should show continuous color variations with no discrete regions, unlike segmentation results.",
        images=["examples/images/2.jpg"],
        mode="dense_perception",
    )
    image.save("normal.png")

    # Example 5: gcg seg
    res = model.generate(
        question="<image> Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
        images=["examples/images/4.jpg"],
        mode="caption_generate",
        return_intermediate_outputs=True,
    )
    image = res["image"]
    text = res["text"]
    print(text)
    image.save("gcg_seg.png")

    # Example 6: object detection
    text = model.generate(
        question="<image> Please detect all instances of <p>bird</p>, <p>boat</p>, <p>person</p>, <p>cell phone</p>, <p>backpack</p>, <p>handbag</p> in the image. Output the results as a structured text list with each detection including category and bounding box coordinates in <bbox> format.",
        images=["examples/images/5.jpg"],
        mode="understanding",
    )
    print(text)

    # Example 7: multi-view 3D reconstruction.
    # This is disabled by default because it is heavier and writes npy/glb files.
    recon_res = model.reconstruct_3d(
        images=[
            "examples/recon3d/47204575_4847.103.png",
            "examples/recon3d/47204575_4852.001.png",
            "examples/recon3d/47204575_4871.692.png",
            "examples/recon3d/47204575_4873.692.png",
            "examples/recon3d/47204575_4875.791.png"

        ],
        raw_output="pred_raw.npy",
        glb_output="pred_scene.glb",
        noise_seed=123456,
        postprocess_predictions=True,
    )
    print("recon pts3d shape:", recon_res["pts3d"].shape)

    # Example 8: camera pose estimation
    print("example camera pose estimation", flush=True)
    raw_text = model.generate(
        question=("<image>" * 5) + CAMERA_POSE_PROMPT,
        images=[
            "examples/recon3d/47204575_4847.103.png",
            "examples/recon3d/47204575_4852.001.png",
            "examples/recon3d/47204575_4871.692.png",
            "examples/recon3d/47204575_4873.692.png",
            "examples/recon3d/47204575_4875.791.png"
        ],
        mode="understanding",
        vit_transform=model.camera_vit_transform,
    )
    print(resolve_pose_string(raw_text))
