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


def _flatten_columns(df: pd.DataFrame, individual: str | None = None):
    """Reduce DLC's MultiIndex columns to just (bodypart, coord).

    DLC columns are (scorer, [individuals,] bodyparts, coords). SuperAnimal
    detection runs often carry an `individuals` level with MANY entries (one
    per detected instance), most of which are empty. Dropping levels blindly
    leaves duplicate bodypart columns, so handle each named level explicitly.

    Returns (flattened_df, chosen_individual_or_None).
    """
    names = list(df.columns.names)

    # 1) Drop the scorer level (always outermost when present).
    if "scorer" in names:
        df = df.droplevel("scorer", axis=1)
        names = list(df.columns.names)

    # 2) Collapse the individuals level by picking ONE individual.
    chosen = None
    if "individuals" in names:
        inds = list(dict.fromkeys(df.columns.get_level_values("individuals")))
        if individual is not None:
            if individual not in inds:
                raise SystemExit(
                    f"--individual '{individual}' not found. Available: {', '.join(map(str, inds))}"
                )
            chosen = individual
        elif len(inds) == 1:
            chosen = inds[0]
        else:
            # Score each individual by mean likelihood; take the best.
            best, best_score = None, -1.0
            for ind in inds:
                sub = df.xs(ind, axis=1, level="individuals")
                if "likelihood" in sub.columns.get_level_values(-1):
                    lk = sub.xs("likelihood", axis=1, level=-1)
                    s = float(np.nanmean(lk.to_numpy(dtype=float)))
                else:
                    s = float(np.isfinite(sub.to_numpy(dtype=float)).mean())
                if np.isnan(s):
                    s = -1.0
                if s > best_score:
                    best, best_score = ind, s
            chosen = best
            print(f"  {len(inds)} individuals in file; using '{chosen}' "
                  f"(highest mean likelihood {best_score:.3f}).")
            print(f"  Others: {', '.join(str(i) for i in inds if i != chosen)}")
            print("  Override with --individual NAME, or --merge-individuals.")
        df = df.xs(chosen, axis=1, level="individuals")

    # 3) Anything still left outside (bodyparts, coords) gets dropped.
    while df.columns.nlevels > 2:
        df = df.droplevel(0, axis=1)

    return df, chosen


def _merge_individuals(df: pd.DataFrame):
    """Collapse all individuals into one by taking the highest-likelihood
    detection per bodypart per frame."""
    names = list(df.columns.names)
    if "scorer" in names:
        df = df.droplevel("scorer", axis=1)
    inds = list(dict.fromkeys(df.columns.get_level_values("individuals")))
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    n = len(df)
    df = df.sort_index(axis=1)  # avoids lexsort PerformanceWarning

    kp = np.full((n, len(bodyparts), 2), np.nan, dtype=np.float32)
    conf = np.zeros((n, len(bodyparts)), dtype=np.float32)

    for k, bp in enumerate(bodyparts):
        best_lk = np.full(n, -1.0)
        for ind in inds:
            try:
                sub = df[(ind, bp)]
            except KeyError:
                continue
            lk = (sub["likelihood"].to_numpy(dtype=float)
                  if "likelihood" in sub.columns else np.ones(n))
            lk = np.nan_to_num(lk, nan=-1.0)
            better = lk > best_lk
            if better.any():
                kp[better, k, 0] = sub["x"].to_numpy(dtype=np.float32)[better]
                kp[better, k, 1] = sub["y"].to_numpy(dtype=np.float32)[better]
                conf[better, k] = lk[better].astype(np.float32)
                best_lk[better] = lk[better]

    print(f"  Merged {len(inds)} individuals -> best detection per bodypart.")
    return bodyparts, kp, conf


def extract_npz(h5_path: Path, video_path: Path, pcutoff: float,
                individual: str | None = None, merge: bool = False) -> Path:
    """Convert DLC's MultiIndex DataFrame into a simple npz."""
    df = pd.read_hdf(h5_path)
    print(f"  HDF5 columns: {df.columns.nlevels} levels {df.columns.names}, "
          f"{len(df)} frames")

    if merge and "individuals" in (df.columns.names or []):
        bodyparts, kp, conf = _merge_individuals(df)
        return _write_npz(video_path, bodyparts, kp, conf, pcutoff, h5_path)

    df, _chosen = _flatten_columns(df, individual)
    bodyparts = list(dict.fromkeys(df.columns.get_level_values(0)))
    n_frames = len(df)
    K = len(bodyparts)

    kp = np.full((n_frames, K, 2), np.nan, dtype=np.float32)
    conf = np.zeros((n_frames, K), dtype=np.float32)

    for k, bp in enumerate(bodyparts):
        sub = df[bp]
        x = sub["x"].to_numpy(dtype=np.float32)
        y = sub["y"].to_numpy(dtype=np.float32)
        # Guard: if a bodypart still maps to >1 column, the MultiIndex wasn't
        # fully flattened. Fail loudly with a useful message instead of a
        # cryptic broadcast error.
        if x.ndim > 1:
            raise SystemExit(
                f"Bodypart '{bp}' resolved to {x.shape[1]} columns, expected 1.\n"
                f"The HDF5 has an unexpected column layout: {df.columns.names}.\n"
                "Try --merge-individuals, or send me the output of:\n"
                "  python -c \"import pandas as pd;"
                "df=pd.read_hdf(r'<your.h5>');print(df.columns.names);"
                "print(df.columns[:8].tolist())\""
            )
        kp[:, k, 0] = x
        kp[:, k, 1] = y
        if "likelihood" in sub.columns:
            conf[:, k] = sub["likelihood"].to_numpy(dtype=np.float32)
        else:
            conf[:, k] = 1.0

    return _write_npz(video_path, bodyparts, kp, conf, pcutoff, h5_path)


def _write_npz(video_path: Path, bodyparts, kp, conf, pcutoff, h5_path) -> Path:
    # Blank out low-confidence points so the renderer can just skip NaNs
    kp = kp.copy()
    kp[conf < pcutoff] = np.nan

    n_frames, K = conf.shape
    frames = np.arange(n_frames, dtype=np.int32)
    out = video_path.with_name(video_path.stem + "_animal_kp.npz")
    np.savez_compressed(
        out,
        frames=frames,
        kp=kp,
        conf=conf,
        bodyparts=np.array(list(map(str, bodyparts)), dtype=object),
        pcutoff=np.float32(pcutoff),
        source_h5=str(h5_path.name),
    )
    kept = int((~np.isnan(kp[:, :, 0])).sum())
    print(f"Saved animal keypoints: {out}")
    print(f"  {n_frames} frames, {K} bodyparts, {kept} points above pcutoff={pcutoff}")
    if kept == 0:
        print("  WARNING: no points survived pcutoff. Try --pcutoff 0.05 "
              "--skip-inference to re-export without re-running the model.")
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
    ap.add_argument("--individual", default=None,
                    help="pick a specific individual by name (default: the one "
                         "with the highest mean likelihood)")
    ap.add_argument("--merge-individuals", action="store_true",
                    help="instead of picking one individual, take the best "
                         "detection per bodypart per frame across all of them")
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
    extract_npz(h5, video_path, args.pcutoff,
                individual=args.individual, merge=args.merge_individuals)

    print("\nNext: run merge_render.py to draw hand + animal onto one video.")


if __name__ == "__main__":
    main()
