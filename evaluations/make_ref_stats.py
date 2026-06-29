import argparse

import numpy as np
import tensorflow.compat.v1 as tf

from evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID/sFID statistics from a .npz with arr_0 images."
    )
    parser.add_argument("input_npz", help="input .npz containing arr_0 images")
    parser.add_argument("output_npz", help="output .npz containing only statistics")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    obj = np.load(args.input_npz)
    if "arr_0" not in obj.files:
        raise ValueError("input .npz must contain arr_0")

    arr = obj["arr_0"]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"arr_0 must have shape [N, H, W, 3], got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"arr_0 must be uint8 in [0, 255], got {arr.dtype}")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size)

    print("warming up TensorFlow...")
    evaluator.warmup()

    print("computing activations...")
    pool_acts, spatial_acts = evaluator.read_activations(args.input_npz)

    print("computing statistics...")
    stats = evaluator.compute_statistics(pool_acts)
    stats_spatial = evaluator.compute_statistics(spatial_acts)

    np.savez(
        args.output_npz,
        mu=stats.mu,
        sigma=stats.sigma,
        mu_s=stats_spatial.mu,
        sigma_s=stats_spatial.sigma,
    )
    print(f"saved {args.output_npz}")


if __name__ == "__main__":
    main()
