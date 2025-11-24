"""
Segment all 'car' instances in numbered images using SAM3 text prompt and save masks.

Features:
- Reads images named like 1.jpg, 2.jpg, ... in --input_dir
- Uses open-vocabulary text prompt with SAM3 (e.g. "car") to obtain instance masks
- Combines (logical OR) all instance masks into one binary mask per image
- Saves composite mask as PNG with same stem in --output_dir (default: masks/)
- Optional: save per-instance masks (instance_{i}) with unique filenames
- Graceful handling when no objects are found (blank mask)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from transformers import Sam3Model, Sam3Processor
except ImportError as e:
    raise ImportError(
        "Could not import Sam3Processor / Sam3Model. Ensure you installed a transformers version that includes SAM3.\n"
        "Try: pip install --upgrade transformers or install from source.\nOriginal error: {}".format(
            e
        )
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Segment cars (or other concept) in numbered frames using SAM3 text prompt."
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing numbered JPG frames (1.jpg, 2.jpg, ...).",
    )
    ap.add_argument(
        "--output_dir", type=str, default="masks", help="Directory to write composite mask PNGs."
    )
    ap.add_argument(
        "--prompt", type=str, default="car", help="Text prompt describing the concept to segment."
    )
    ap.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Minimum instance confidence score to keep.",
    )
    ap.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold in post-processing.",
    )
    ap.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip writing composite mask if file already exists.",
    )
    ap.add_argument(
        "--save_instances", action="store_true", help="Also save individual instance masks."
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. cpu, cuda, cuda:1). Default auto.",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype. bfloat16/float16 for GPU memory savings.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process at most N frames (debug).")
    return ap.parse_args()


def get_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_model_and_processor(device: torch.device, dtype: torch.dtype):
    # Load pre-trained SAM3 (open-vocabulary concept segmentation)
    model = Sam3Model.from_pretrained("facebook/sam3")
    model = model.to(device=device, dtype=dtype)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    return model, processor


def list_numbered_jpgs(folder: Path):
    files = [p for p in folder.glob("*.jpg") if p.stem.isdigit()]
    files.sort(key=lambda p: int(p.stem))
    return files


def segment_image(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
):
    """
    Run SAM3 instance segmentation with a text prompt.
    Returns dict with keys: masks (Tensor[N,H,W]), scores (Tensor[N]), boxes (Tensor[N,4]).
    All resized to original image size by post_process_instance_segmentation.
    """
    # Prepare inputs (batch size 1)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get instance masks at original resolution
    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]  # Single image
    # processed: dict(masks, boxes, scores)
    return processed


def union_instance_masks(masks: torch.Tensor) -> np.ndarray:
    """
    masks: Tensor[N,H,W] binary (0/1) or bool.
    Returns uint8 array HxW with 255 for any positive pixel.
    """
    if masks.numel() == 0:
        return None
    # Ensure binary
    masks_bin = (masks > 0).to(torch.bool)  # N,H,W
    combined = torch.any(masks_bin, dim=0).cpu().numpy().astype(np.uint8) * 255
    return combined


def save_mask(array: np.ndarray, path: Path):
    Image.fromarray(array).save(path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    device = get_device(args.device)
    dtype = get_torch_dtype(args.dtype)

    print(f"Device: {device}  |  DType: {dtype}")
    print("Loading SAM3 model & processor ...")
    model, processor = load_model_and_processor(device, dtype)

    frames = list_numbered_jpgs(input_dir)
    if not frames:
        print(f"No numbered .jpg files found in {input_dir}")
        sys.exit(0)

    if args.limit is not None:
        frames = frames[: args.limit]
        print(f"Limiting to {len(frames)} frames.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_instances:
        (output_dir / "instances").mkdir(parents=True, exist_ok=True)

    for img_path in frames:
        composite_path = output_dir / f"{img_path.stem}.png"
        if composite_path.exists() and args.no_overwrite:
            print(f"Skipping existing composite: {composite_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

        result = segment_image(
            model=model,
            processor=processor,
            image=image,
            prompt=args.prompt,
            device=device,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
        )

        masks = result.get("masks", None)
        scores = result.get("scores", torch.tensor([]))
        if masks is None or masks.shape[0] == 0:
            # Create blank mask
            blank = np.zeros((image.height, image.width), dtype=np.uint8)
            save_mask(blank, composite_path)
            print(f"{img_path.name}: no '{args.prompt}' instances found -> blank mask saved.")
            continue

        composite = union_instance_masks(masks)
        if composite is None:
            composite = np.zeros((image.height, image.width), dtype=np.uint8)

        save_mask(composite, composite_path)

        msg = f"{img_path.name}: {masks.shape[0]} '{args.prompt}' instance(s) -> {composite_path.name}"
        if args.save_instances:
            for i in range(masks.shape[0]):
                inst_mask = (masks[i] > 0).cpu().numpy().astype(np.uint8) * 255
                inst_path = output_dir / "instances" / f"{img_path.stem}_instance_{i}.png"
                save_mask(inst_mask, inst_path)
            msg += f" (+ {masks.shape[0]} instance mask files)"
        print(msg)

    print("Done.")


if __name__ == "__main__":
    main()
