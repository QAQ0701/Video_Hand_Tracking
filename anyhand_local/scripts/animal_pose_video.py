"""
animal_pose_video.py  -  DeepLabCut SuperAnimal (quadruped) pose on a video.

STEP 2 of 3 in the combined-render pipeline.

Runs SuperAnimal inference, then extracts DLC's HDF5 predictions into a plain
.npz that merge_render.py can read (no DeepLabCut needed at merge time).

    conda activate dlc
    python scripts\\animal_pose_video.py path\\to\\video.mp4

Outputs (next to the video):
    <video>_animal_kp.npz   - frames:(N,) int, kp:(N,K,2), conf:(N,K),
                              bodyparts:(K,) str, meta
    plus DLC's own *_labeled.mp4 / .h5 files

CPU notes:
  * --adapt (video_adapt fine-tuning) is OFF by default; it can take HOURS
    on CPU. Only enable it if you can leave it overnight.
  * If DLC inference was already run, use --skip-inference to just re-extract
    the .npz from the existing .h5 (fast).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def find_h5(video_path: Path, prefer_adapt: bool) -> Path:
    """Locate DLC's prediction HDF5 for this video.

    SuperAnimal runs can emit two .h5 files (before/after video adaptation),
    so pick deliberately rather than grabbing the first match.
    """
    stem = video_path.stem
    folder = video_path.parent
    cands = [Path(p) for p in glob.glob(str(folder / f"{stem}*.h5"))]
    if not cands:
        raise SystemExit(
            f"No DLC .h5 found for {video_path.name} in {folder}.\n"
            "Run without --skip-inference first."
        )

    adapt = [c for c in cands if "adapt" in c.name.lower()]
    if prefer_adapt and adapt:
        chosen = sorted(adapt)[-1]
    else:
        plain = [c for c in cands if "adapt" not in c.name.lower()]
        chosen = sorted(plain or cands)[-1]

    if len(cands) > 1:
        print(f"  Found {len(cands)} .h5 candidates; using: {chosen.name}")
        for c in cands:
            if c != chosen:
                print(f"    (ignoring {c.name})")
    return chosen


def extract_npz(h5_path: Path, video_path: Path, pcutoff: float) -> Path:
    """Convert DLC's MultiIndex DataFrame into a simple npz."""
    df = pd.read_hdf(h5_path)

    # DLC columns are a MultiIndex: (scorer, [individual,] bodypart, coord).
    # Reduce to (bodypart, coord).
    while df.columns.nlevels > 2:
        df = df.droplevel(0, axis=1)

    bodyparts = list(dict.fromkeys(df.columns.get_level_values(0)))
    n_frames = len(df)
    K = len(bodyparts)

    kp = np.full((n_frames, K, 2), np.nan, dtype=np.float32)
    conf = np.zeros((n_frames, K), dtype=np.float32)

    for k, bp in enumerate(bodyparts):
        sub = df[bp]
        kp[:, k, 0] = sub["x"].to_numpy(dtype=np.float32)
        kp[:, k, 1] = sub["y"].to_numpy(dtype=np.float32)
        if "likelihood" in sub.columns:
            conf[:, k] = sub["likelihood"].to_numpy(dtype=np.float32)
        else:
            conf[:, k] = 1.0

    # Blank out low-confidence points so the renderer can just skip NaNs
    kp[conf < pcutoff] = np.nan

    frames = np.arange(n_frames, dtype=np.int32)
    out = video_path.with_name(video_path.stem + "_animal_kp.npz")
    np.savez_compressed(
        out,
        frames=frames,
        kp=kp,
        conf=conf,
        bodyparts=np.array(bodyparts, dtype=object),
        pcutoff=np.float32(pcutoff),
        source_h5=str(h5_path.name),
    )
    kept = int((~np.isnan(kp[:, :, 0])).sum())
    print(f"Saved animal keypoints: {out}")
    print(f"  {n_frames} frames, {K} bodyparts, {kept} points above pcutoff={pcutoff}")
    print(f"  bodyparts: {', '.join(map(str, bodyparts))}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--adapt", action="store_true",
                    help="enable video_adapt fine-tuning (VERY slow on CPU)")
    ap.add_argument("--pcutoff", type=float, default=0.15,
                    help="confidence cutoff (default 0.15)")
    ap.add_argument("--skip-inference", action="store_true",
                    help="reuse existing .h5 and just re-export the .npz")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    if not args.skip_inference:
        import deeplabcut  # imported here so --skip-inference works without DLC
        print(f"Running SuperAnimal quadruped on: {video_path}")
        print(f"  video_adapt={args.adapt}  pcutoff={args.pcutoff}")
        deeplabcut.video_inference_superanimal(
            [str(video_path)],
            "superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_mobilenet_v3_large_fpn",
            videotype=video_path.suffix,
            video_adapt=args.adapt,
            scale_list=[],
            pcutoff=args.pcutoff,
        )
    else:
        print("Skipping inference, reusing existing .h5")

    h5 = find_h5(video_path, prefer_adapt=args.adapt)
    print(f"Reading predictions from: {h5.name}")
    extract_npz(h5, video_path, args.pcutoff)

    print("\nNext: run merge_render.py to draw hand + animal onto one video.")


if __name__ == "__main__":
    main()
