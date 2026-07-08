import os
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.utils_3d import resolve_pose_string
from inference.sensenova_vision import SenseNovaVisionModel  # noqa: E402
from utils.visualize import (  # noqa: E402
    VisualizationConfig,
    draw_visual_prompt,
    visualize_binary_segmentation,
    visualize_concat_col,
    visualize_detection,
    visualize_gcg_segmentation,
    visualize_panoptic_segmentation,
)


OUTPUT_DIR = REPO_ROOT / "examples/output/example_visualize"
DEFAULT_MODEL_PATH = "sensenova/SenseNova-Vision-7B-MoT"

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

EXAMPLE_08_PANOPTIC_IMAGE = "examples/images/6.jpg"
EXAMPLE_08_PANOPTIC_QUESTION = (
    "<image>Can you show me how the image would be segmented into <p>person</p>, <p>bicycle</p>, <p>car</p>, <p>motorcycle</p>, <p>airplane</p>, <p>bus</p>, <p>train</p>, <p>truck</p>, <p>boat</p>, <p>traffic light</p>, <p>fire hydrant</p>, <p>stop sign</p>, <p>parking meter</p>, <p>bench</p>, <p>bird</p>, <p>cat</p>, <p>dog</p>, <p>horse</p>, <p>sheep</p>, <p>cow</p>, <p>elephant</p>, <p>bear</p>, <p>zebra</p>, <p>giraffe</p>, <p>backpack</p>, <p>umbrella</p>, <p>handbag</p>, <p>tie</p>, <p>suitcase</p>, <p>frisbee</p>, <p>skis</p>, <p>snowboard</p>, <p>sports ball</p>, <p>kite</p>, <p>baseball bat</p>, <p>baseball glove</p>, <p>skateboard</p>, <p>surfboard</p>, <p>tennis racket</p>, <p>bottle</p>, <p>wine glass</p>, <p>cup</p>, <p>fork</p>, <p>knife</p>, <p>spoon</p>, <p>bowl</p>, <p>banana</p>, <p>apple</p>, <p>sandwich</p>, <p>orange</p>, <p>broccoli</p>, <p>carrot</p>, <p>hot dog</p>, <p>pizza</p>, <p>donut</p>, <p>cake</p>, <p>chair</p>, <p>couch</p>, <p>potted plant</p>, <p>bed</p>, <p>dining table</p>, <p>toilet</p>, <p>tv</p>, <p>laptop</p>, <p>mouse</p>, <p>remote</p>, <p>keyboard</p>, <p>cell phone</p>, <p>microwave</p>, <p>oven</p>, <p>toaster</p>, <p>sink</p>, <p>refrigerator</p>, <p>book</p>, <p>clock</p>, <p>vase</p>, <p>scissors</p>, <p>teddy bear</p>, <p>hair drier</p>, <p>toothbrush</p>, <p>banner</p>, <p>blanket</p>, <p>bridge</p>, <p>cardboard</p>, <p>counter</p>, <p>curtain</p>, <p>door</p>, <p>floor-wood</p>, <p>flower</p>, <p>fruit</p>, <p>gravel</p>, <p>house</p>, <p>light</p>, <p>mirror</p>, <p>net</p>, <p>pillow</p>, <p>platform</p>, <p>playingfield</p>, <p>railroad</p>, <p>river</p>, <p>road</p>, <p>roof</p>, <p>sand</p>, <p>sea</p>, <p>shelf</p>, <p>snow</p>, <p>stairs</p>, <p>tent</p>, <p>towel</p>, <p>wall-brick</p>, <p>wall-stone</p>, <p>wall-tile</p>, <p>wall-wood</p>, <p>water</p>, <p>window-blind</p>, <p>window</p>, <p>tree</p>, <p>fence</p>, <p>ceiling</p>, <p>sky</p>, <p>cabinet</p>, <p>table</p>, <p>floor</p>, <p>pavement</p>, <p>mountain</p>, <p>grass</p>, <p>dirt</p>, <p>paper</p>, <p>food</p>, <p>building</p>, <p>rock</p>, <p>wall</p>, <p>rug</p> using panoptic segmentation?Please find all instances in the image and assign color to each instance in the EXACT format: <p>instance-no<color>(R,G,B)</color></p>, then respond with panoptic segmentation masks."
)
EXAMPLE_09_INTERSEG_IMAGE = "examples/images/7.jpg"
EXAMPLE_09_INTERSEG_PROMPT_IMAGE = "examples/images/7_prompt.png"
EXAMPLE_09_INTERSEG_QUESTION = (
    "<image>Can you provide segmentation masks for this image based on these "
    "regions: <box><image>? Please provide the segmentation masks."
)

def resolve_model_path():
    return (
        os.environ.get("MODEL_PATH")
        or os.environ.get("SENSENOVA_MODEL_PATH")
        or DEFAULT_MODEL_PATH
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = VisualizationConfig()

    model = SenseNovaVisionModel(
        model_path=resolve_model_path(),
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
    image_path = "examples/images/2.jpg"
    source = Image.open(REPO_ROOT / image_path).convert("RGB")
    image = model.generate(
        question="<image> Could you return the binary segmentation masks for the specified categories: <p>person furthest to the right</p>?",
        images=[image_path],
        mode="dense_perception",
    )
    image.save(OUTPUT_DIR / "example_02_binary_segmentation_raw.png")
    pred = visualize_binary_segmentation(
        source,
        image,
        label="person furthest to the right",
        config=config,
    )
    visualize_concat_col(source, pred, concat_col=2).save(
        OUTPUT_DIR / "example_02_binary_segmentation.png"
    )

    # Example 3: depth estimation, returns a PIL.Image.Image
    image = model.generate(
        question="<image> Estimate relative depth for each pixel in the image, with closer objects appearing brighter and distant objects appearing darker. Output is a grayscale image with pixel values ranging from 0-255.",
        images=["examples/images/3.jpg"],
        mode="dense_perception",
    )
    image.save(OUTPUT_DIR / "example_03_depth.png")

    # Example 4: normal estimation, returns a PIL.Image.Image
    image = model.generate(
        question="<image> Generate an RGB normal map where R, G, B channels represent X, Y, Z surface directions. The output should show continuous color variations with no discrete regions, unlike segmentation results.",
        images=["examples/images/2.jpg"],
        mode="dense_perception",
    )
    image.save(OUTPUT_DIR / "example_04_normal.png")

    # Example 5: gcg seg
    image_path = "examples/images/4.jpg"
    source = Image.open(REPO_ROOT / image_path).convert("RGB")
    res = model.generate(
        question="<image> Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
        images=[image_path],
        mode="caption_generate",
        return_intermediate_outputs=True,
    )
    image = res["image"]
    text = res["text"]
    print(text)
    image.save(OUTPUT_DIR / "example_05_gcg_segmentation_raw.png")
    pred = visualize_gcg_segmentation(source, image, text, config=config)
    visualize_concat_col(source, pred, concat_col=2).save(
        OUTPUT_DIR / "example_05_gcg_segmentation.png"
    )

    # Example 6: object detection
    image_path = "examples/images/5.jpg"
    source = Image.open(REPO_ROOT / image_path).convert("RGB")
    text = model.generate(
        question="<image> Please detect all instances of <p>bird</p>, <p>boat</p>, <p>person</p>, <p>cell phone</p>, <p>backpack</p>, <p>handbag</p> in the image. Output the results as a structured text list with each detection including category and bounding box coordinates in <bbox> format.",
        images=[image_path],
        mode="understanding",
    )
    print(text)
    pred = visualize_detection(
        source,
        text,
        task_name="common_object_detection",
        config=config,
    )
    visualize_concat_col(source, pred, concat_col=2).save(
        OUTPUT_DIR / "example_06_object_detection.png"
    )

    # Example 7: multi-view 3D reconstruction.
    recon_res = model.reconstruct_3d(
        images=[
            "examples/recon3d/47204575_4847.103.png",
            "examples/recon3d/47204575_4852.001.png",
            "examples/recon3d/47204575_4871.692.png",
            "examples/recon3d/47204575_4873.692.png",
            "examples/recon3d/47204575_4875.791.png",
        ],
        raw_output=str(OUTPUT_DIR / "example_07_pred_raw.npy"),
        glb_output=str(OUTPUT_DIR / "example_07_pred_scene.glb"),
        noise_seed=123456,
        postprocess_predictions=True,
    )
    print("recon pts3d shape:", recon_res["pts3d"].shape)

    # Extra Example 8: panoptic segmentation
    image_path = EXAMPLE_08_PANOPTIC_IMAGE
    source = Image.open(REPO_ROOT / image_path).convert("RGB")
    res = model.generate(
        question=EXAMPLE_08_PANOPTIC_QUESTION,
        images=[image_path],
        mode="caption_generate",
        return_intermediate_outputs=True,
    )
    image = res["image"]
    text = res["text"]
    print(text)
    image.save(OUTPUT_DIR / "example_08_panoptic_segmentation_raw.png")
    pred = visualize_panoptic_segmentation(
        source,
        image,
        text,
        question=EXAMPLE_08_PANOPTIC_QUESTION,
        config=config,
    )
    visualize_concat_col(source, pred, concat_col=2).save(
        OUTPUT_DIR / "example_08_panoptic_segmentation.png"
    )

    # Extra Example 9: interactive segmentation
    image_path = EXAMPLE_09_INTERSEG_IMAGE
    prompt_path = EXAMPLE_09_INTERSEG_PROMPT_IMAGE
    source = Image.open(REPO_ROOT / image_path).convert("RGB")
    prompt = Image.open(REPO_ROOT / prompt_path).convert("L")
    image = model.generate(
        question=EXAMPLE_09_INTERSEG_QUESTION,
        images=[image_path, prompt_path],
        mode="dense_perception",
    )
    image.save(OUTPUT_DIR / "example_09_interactive_segmentation_raw.png")
    prompt_panel = draw_visual_prompt(source, prompt, prompt_style="boundary")
    pred = visualize_binary_segmentation(
        source,
        image,
        label="box prompt",
        config=config,
    )
    visualize_concat_col(source, pred, concat_col=3, prompt=prompt_panel).save(
        OUTPUT_DIR / "example_09_interactive_segmentation.png"
    )

    print(f"visualizations saved to {OUTPUT_DIR}")

    # Example 10: vgd seg
    image_path = "examples/images/8.jpg"
    source = Image.open(REPO_ROOT / image_path).convert("RGB")
    res = model.generate(
        question="<image> Identify all objects belonging to the same classes as the visually provided <p>object1</p><bbox>[0.616, 0.049, 0.785, 0.224]</bbox>. Generate an instance segmentation visualization and each identified category <p>object1</p> is colored different. First, enumerate each visible <p>object1</p> instance mentioned in the request and assign each <p>object1</p> a different color. Reformat them in the EXACT format: <p>object1<color>(R,G,B)</color></p>. Then respond with interleaved instance segmentation masks using those instance labels and colors.",
        images=[image_path],
        mode="caption_generate",
        return_intermediate_outputs=True,
    )
    image = res["image"]
    text = res["text"]
    print(text)
    image.save(OUTPUT_DIR / "example_10_vgd_segmentation_raw.png")
    pred = visualize_gcg_segmentation(source, image, text, config=config)
    visualize_concat_col(source, pred, concat_col=2).save(
        OUTPUT_DIR / "example_10_vgd_segmentation.png"
    )

    # Example 11: camera pose estimation
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
