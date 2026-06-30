import argparse

import numpy as np
from scipy import linalg


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"mean vectors have different lengths: {mu1.shape}, {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"covariances have different dimensions: {sigma1.shape}, {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def require_keys(obj, path, keys):
    missing = [key for key in keys if key not in obj.files]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID/sFID from two .npz files containing precomputed statistics."
    )
    parser.add_argument("ref_npz", help="reference stats .npz")
    parser.add_argument("sample_npz", help="sample stats .npz")
    args = parser.parse_args()

    ref = np.load(args.ref_npz)
    sample = np.load(args.sample_npz)
    require_keys(ref, args.ref_npz, ["mu", "sigma"])
    require_keys(sample, args.sample_npz, ["mu", "sigma"])

    fid = frechet_distance(ref["mu"], ref["sigma"], sample["mu"], sample["sigma"])
    print(f"FID: {fid}")

    has_ref_sfid = "mu_s" in ref.files and "sigma_s" in ref.files
    has_sample_sfid = "mu_s" in sample.files and "sigma_s" in sample.files
    if has_ref_sfid and has_sample_sfid:
        sfid = frechet_distance(
            ref["mu_s"], ref["sigma_s"], sample["mu_s"], sample["sigma_s"]
        )
        print(f"sFID: {sfid}")
    elif has_ref_sfid != has_sample_sfid:
        print("sFID: skipped because only one file contains mu_s/sigma_s")


if __name__ == "__main__":
    main()
