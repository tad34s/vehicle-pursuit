import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

# Configuration
INPUT_DIR = Path("dataset") / "images"
OUTPUT_DIR = Path("dataset") / "output_masks"
TEXT_PROMPT = "red car"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_models():
    """Load Grounding DINO and SAM models with processors"""
    print("Loading models...")

    # Grounding DINO
    dino_id = "IDEA-Research/grounding-dino-tiny"
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)

    # SAM
    sam_id = "facebook/sam-vit-base"
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(device)

    return dino_processor, dino_model, sam_processor, sam_model


def create_mask_visual(mask, image_np):
    """Create visualization of mask overlay on image"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)

    # Apply semi-transparent blue mask
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_img)

    plt.axis("off")
    return plt


def process_image(image_path, dino_processor, dino_model, sam_processor, sam_model):
    """Process single image through detection and segmentation pipeline"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    filename = image_path.stem

    # Grounding DINO detection
    inputs = dino_processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    # Process detections
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],  # (height, width)
    )

    # Extract results - note: returns list per image
    if results and len(results) > 0:
        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
    else:
        boxes, scores, labels = [], [], []

    # Filter relevant boxes
    filtered_boxes = []
    for i, label in enumerate(labels):
        # Check if the detected label matches the prompt
        if TEXT_PROMPT.lower() in label.lower() and scores[i] > BOX_THRESHOLD:
            filtered_boxes.append(boxes[i])

    # Handle no detections
    if not filtered_boxes:
        print(f"No '{TEXT_PROMPT}' detected in {filename}")
        empty_mask = np.zeros(image.size[::-1], dtype=np.uint8)
        return empty_mask, image_np, filename

    # Prepare boxes for SAM - convert to list of lists
    boxes_list = [box.cpu().tolist() for box in filtered_boxes]

    # SAM segmentation
    sam_inputs = sam_processor(
        image,
        input_boxes=[boxes_list],  # Wrap in list for batch dimension
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    # Process masks
    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
        binarize=True,
    )

    # Combine all masks - FIXED LOGIC
    # Initialize a blank mask
    combined_mask = torch.zeros(image.size[::-1], dtype=torch.bool)

    # Process each set of masks
    for mask_set in masks:
        # Check if mask_set has any masks
        if mask_set.numel() > 0:  # Proper way to check for empty tensors
            # We'll take the first mask for each box (best quality)
            mask_first = mask_set[0, 0]  # Shape: (H, W)
            combined_mask = combined_mask | mask_first

    return combined_mask.numpy().astype(np.uint8) * 255, image_np, filename


# Main execution
if __name__ == "__main__":
    dino_processor, dino_model, sam_processor, sam_model = load_models()

    # Get image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in image_extensions]

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        sys.exit()

    print(f"Found {len(image_paths)} images to process")

    for path in image_paths:
        print(f"\nProcessing {path.name}...")
        mask, image_np, filename = process_image(
            path, dino_processor, dino_model, sam_processor, sam_model
        )

        # Save mask
        mask_path = OUTPUT_DIR / f"mask_{filename}.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"Mask saved to {mask_path}")

        # Save visualization
        plt = create_mask_visual(mask, image_np)
        overlay_path = OUTPUT_DIR / f"overlay_{filename}.png"
        plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Overlay saved to {overlay_path}")
