import yaml
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
import sys

sys.path.append("sam2")
sys.path.append("sam2/sam2")
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor

SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_tiny.pt"


def get_sam2_image_model():
    return build_sam2(
        SAM_CONFIG,
        SAM_CHECKPOINT,
        device="cuda",
        apply_postprocessing=False,
    )


def get_image_predictor(sam2_img_model=None):
    if not sam2_img_model:
        sam2_img_model = get_sam2_image_model()
    predictor = SAM2ImagePredictor(sam2_img_model)
    return predictor


def get_video_predictor():
    predictor = build_sam2_video_predictor(SAM_CONFIG, SAM_CHECKPOINT)
    return predictor


def get_dino_models():
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        "cuda",
    )
    return processor, grounding_model


def get_dino_boxes(image, prompt, processor, model, return_full=False):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        # box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    if return_full:
        return results
    else:
        return results[0]["boxes"]


def get_image_masks_from_boxes(image_predictor, boxes, image):
    image_predictor.set_image(image)
    masks, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)
    return masks


def fetch_segments(image, prompt, processor, grounding_model, image_predictor):
    boxes = get_dino_boxes(image, prompt, processor, grounding_model)
    image_masks = get_image_masks_from_boxes(image_predictor, boxes, image)
    return image_masks


def propagate_mask(video_model, images_folder, mask):
    masks_result = []
    state = video_model.init_state(images_folder)
    video_model.add_new_mask(
        state,
        frame_idx=0,
        obj_id=0,
        mask=mask,
    )
    for (
        out_frame_idx,
        _,
        out_mask_logits,
    ) in video_model.propagate_in_video(state):
        out_mask_logits = out_mask_logits[:, 0, :, :] > 0.0
        masks_result.append(out_mask_logits)
    return masks_result


def segment_LF(
    LF_folder, image_predictor, dino_processor, dino_model, video_model, prompt
):
    imgs_directory = "/".join(LF_folder[0].split("/")[:-1])
    first_view = Image.open(LF_folder[0])
    image_masks = fetch_segments(
        first_view, prompt, dino_processor, dino_model, image_predictor
    )
    masks = propagate_mask(video_model, imgs_directory, image_masks[0])
    return masks


def get_dataset_masks(dataset_path, prompt):
    def list_dir(folder):
        png_files = list(sorted(os.listdir(folder)))
        png_files = [f"{folder}/{item}" for item in png_files if item.endswith(".png")]
        return png_files

    image_predictor = get_image_predictor()
    processor, grounding_model = get_dino_models()
    video_model = get_video_predictor()
    LF_folders_path = dataset_path
    LF_folders = list(sorted(os.listdir(LF_folders_path)))
    LF_folders = [
        f"{LF_folders_path}/{item}" for item in LF_folders if item.startswith("LF_")
    ]
    n_folders = len(LF_folders)
    tmp_folder = dataset_path + "/tmp_sam"
    os.makedirs(tmp_folder, exist_ok=True)
    for folder_i, folder in enumerate(LF_folders):
        LF_files = list_dir(folder)
        n_files = len(LF_files)
        for file_i, filename in enumerate(LF_files):
            img_i = folder_i * len(LF_files) + file_i
            file_name = str(img_i).zfill(4) + ".jpg"
            shutil.copy(filename, tmp_folder + "/" + file_name)
    masks = segment_LF(
        [f"{tmp_folder}/{item}" for item in list(sorted(os.listdir(tmp_folder)))],
        image_predictor,
        processor,
        grounding_model,
        video_model,
        prompt,
    )
    mask_i = 0
    for n_folder in range(n_folders):
        os.makedirs(f"{LF_folders[n_folder]}/masks", exist_ok=True)
        for n_file in range(n_files):
            img_i = n_folder * n_files + n_file
            mask = masks[mask_i]
            mask = (mask * 255.0).cpu().numpy().astype(np.uint8)[0]
            mask = Image.fromarray(mask)
            mask.save(f"{LF_folders[n_folder]}/masks" + f"/{str(n_file).zfill(4)}.png")
            file_name = str(img_i).zfill(4) + ".jpg"
            mask_i += 1
    shutil.rmtree(tmp_folder)
    with open(f"{dataset_path}/gdino_prompt.txt", "w") as file:
        file.write(prompt)


if __name__ == "__main__":
    pass
