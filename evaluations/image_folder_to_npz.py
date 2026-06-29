import argparse
import os
import tempfile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from guided_diffusion.image_datasets import center_crop_arr


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}


def list_images(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTENSIONS:
                image_paths.append(os.path.join(root, name))
    return sorted(image_paths)


def labels_from_parent_dirs(image_paths, data_dir):
    class_names = []
    for path in image_paths:
        rel_path = os.path.relpath(path, data_dir)
        parts = rel_path.split(os.sep)
        if len(parts) < 2:
            raise ValueError(
                "cannot infer labels: expected images under class subdirectories"
            )
        class_names.append(parts[0])
    class_to_idx = {name: idx for idx, name in enumerate(sorted(set(class_names)))}
    labels = np.array([class_to_idx[name] for name in class_names], dtype=np.int64)
    return labels, class_to_idx


def main():
    parser = argparse.ArgumentParser(
        description="Create an evaluator-compatible .npz from an image folder."
    )
    parser.add_argument("data_dir", help="image folder, e.g. ImageNet val/")
    parser.add_argument("output_npz", help="output .npz path")
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="do not save arr_1 labels inferred from parent directories",
    )
    args = parser.parse_args()

    image_paths = list_images(args.data_dir)
    if not image_paths:
        raise ValueError(f"no images found under {args.data_dir}")

    labels = None
    if not args.no_labels:
        labels, class_to_idx = labels_from_parent_dirs(image_paths, args.data_dir)
        print(f"found {len(class_to_idx)} classes")

    print(f"found {len(image_paths)} images")
    output_dir = os.path.dirname(os.path.abspath(args.output_npz)) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_npy = tempfile.mkstemp(
        prefix=".image_folder_to_npz_", suffix=".npy", dir=output_dir
    )
    os.close(fd)

    try:
        arr = np.lib.format.open_memmap(
            tmp_npy,
            mode="w+",
            dtype=np.uint8,
            shape=(len(image_paths), args.image_size, args.image_size, 3),
        )

        for idx, path in enumerate(tqdm(image_paths)):
            with Image.open(path) as pil_image:
                pil_image.load()
                pil_image = pil_image.convert("RGB")
                arr[idx] = center_crop_arr(pil_image, args.image_size).astype(np.uint8)
        arr.flush()

        if labels is None:
            np.savez(args.output_npz, arr)
        else:
            np.savez(args.output_npz, arr, labels)
        print(f"saved {args.output_npz}")
    finally:
        if os.path.exists(tmp_npy):
            os.remove(tmp_npy)


if __name__ == "__main__":
    main()
