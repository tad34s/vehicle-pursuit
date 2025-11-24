#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Merge two datasets with structure:
# <source>/
#   imgs/
#     0.jpg, 1.jpg, ...
#   masks/
#     0.jpg, 1.jpg, ...
#
# Usage:
#   ./merge_datasets.sh office hall [output_dir]
# Example:
#   ./merge_datasets.sh office hall combined
#
# Result:
#   combined/
#     imgs/
#       0.jpg, 1.jpg, ...
#     masks/
#       0.jpg, 1.jpg, ...
#
# Notes:
# - Images and masks are matched by the numeric basename.
# - Second dataset is renumbered to continue after the current max index.
# - If a mask is missing for an image, the pair is skipped with a warning.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <source_dataset1> <source_dataset2> [output_dir]" >&2
  exit 1
fi

SRC1="$1"
SRC2="$2"
OUT_ROOT="${3:-combined}"

IMGS_SUBDIR="imgs"
MASKS_SUBDIR="masks"

ACCEPTED_EXT_RE='(jpg|jpeg|png|bmp|tif|tiff)'

# Validate sources
for SRC in "$SRC1" "$SRC2"; do
  if [[ ! -d "$SRC" ]]; then
    echo "Error: Source directory not found: $SRC" >&2
    exit 1
  fi
  if [[ ! -d "$SRC/$IMGS_SUBDIR" ]]; then
    echo "Error: Missing imgs directory: $SRC/$IMGS_SUBDIR" >&2
    exit 1
  fi
  if [[ ! -d "$SRC/$MASKS_SUBDIR" ]]; then
    echo "Error: Missing masks directory: $SRC/$MASKS_SUBDIR" >&2
    exit 1
  fi
done

# Prepare output dirs
OUT_IMGS="$OUT_ROOT/$IMGS_SUBDIR"
OUT_MASKS="$OUT_ROOT/$MASKS_SUBDIR"
mkdir -p "$OUT_IMGS" "$OUT_MASKS"

get_max_index_in_dir() {
  local dir="$1"
  local max=-1
  # Find files with numeric basenames in the directory
  while IFS= read -r f; do
    local bn="${f##*/}"
    local num="${bn%%.*}"
    if [[ "$num" =~ ^[0-9]+$ ]]; then
      if (( num > max )); then
        max="$num"
      fi
    fi
  done < <(find "$dir" -maxdepth 1 -type f -regextype posix-extended -regex ".*/[0-9]+\..+" -print)
  echo "$max"
}

find_mask_for_num() {
  # Find a mask file for a given number, preferring same extension as image.
  local masks_dir="$1"
  local num="$2"
  local preferred_ext="$3"

  local cand="$masks_dir/$num.$preferred_ext"
  if [[ -f "$cand" ]]; then
    echo "$cand"
    return 0
  fi

  # Fall back to any supported extension
  local f
  for f in "$masks_dir/$num".*; do
    [[ -e "$f" ]] || continue
    local ext="${f##*.}"
    if [[ "$ext" =~ ^$ACCEPTED_EXT_RE$ ]]; then
      echo "$f"
      return 0
    fi
  done

  return 1
}

copy_dataset_with_offset() {
  local src="$1"
  local offset="$2"

  # Collect images with numeric basenames and supported extensions
  # We use find to be robust and handle only numeric filenames.
  while IFS= read -r img; do
    local bn="${img##*/}"
    local num="${bn%%.*}"
    local ext="${bn##*.}"

    # Validate numeric
    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
      echo "Skipping non-numeric image filename: $img" >&2
      continue
    fi

    # Find corresponding mask
    local mask
    if ! mask="$(find_mask_for_num "$src/$MASKS_SUBDIR" "$num" "$ext")"; then
      echo "Warning: No mask found for image $img (number $num). Skipping pair." >&2
      continue
    fi

    local dest_idx=$((num + offset))
    local dest_img="$OUT_IMGS/${dest_idx}.${ext}"
    local mask_ext="${mask##*.}"
    local dest_mask="$OUT_MASKS/${dest_idx}.${mask_ext}"

    # Copy files
    cp -f -- "$img" "$dest_img"
    cp -f -- "$mask" "$dest_mask"
    echo "Copied: $img -> $dest_img"
    echo "        $mask -> $dest_mask"
  done < <(find "$src/$IMGS_SUBDIR" -type f -regextype posix-extended -regex ".*/[0-9]+\.$ACCEPTED_EXT_RE" | sort -V)
}

# Start with current max in output (supports resuming/appending)
current_max="$(get_max_index_in_dir "$OUT_IMGS")"
if [[ "$current_max" == "-1" ]]; then
  offset1=0
else
  offset1=$((current_max + 1))
fi

echo "Merging '$SRC1' into '$OUT_ROOT' with offset $offset1 ..."
copy_dataset_with_offset "$SRC1" "$offset1"

# Update offset for second dataset based on new max
current_max="$(get_max_index_in_dir "$OUT_IMGS")"
offset2=$((current_max + 1))
echo "Merging '$SRC2' into '$OUT_ROOT' with offset $offset2 ..."
copy_dataset_with_offset "$SRC2" "$offset2"

echo "Done. Combined dataset at: $OUT_ROOT"
echo "Images: $OUT_IMGS"
echo "Masks:  $OUT_MASKS"
