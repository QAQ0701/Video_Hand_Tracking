"""
hand_pose_video.py  -  AnyHand / WiLoR hand pose on a video (local, CPU).

STEP 1 of 3 in the combined-render pipeline.

Instead of rendering its own video, this now exports ALL 21 hand joints per
processed frame to an .npz keyed by absolute frame index, so merge_render.py
can draw hand + animal onto the same frames later.

    conda activate anyhand
    cd AnyHand
    python ..\\scripts\\hand_pose_video.py path\\to\\video.mp4

Outputs (next to the video):
    <video>_hand_kp.npz      - frames:(N,) int, kp:(N,H,21,2), score:(N,H),
                               is_right:(N,H), n_hands:(N,), meta
    <video>_hand_traj.png    - wrist trajectory scatter (same as before)

Use --write-video only if you want a hand-only preview; the combined render
comes from merge_render.py.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.rgb_predictor import AnyHandPredictor

# MANO 21-joint skeleton, grouped by finger. cv2 uses BGR.
FINGER_BONES = [
    ((58,  61, 214), [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ((51, 126, 242), [(0, 5), (5, 6), (6, 7), (7, 8)]),
    ((75, 162,  44), [(0, 9), (9, 10), (10, 11), (11, 12)]),
    ((185, 128, 31), [(0, 13), (13, 14), (14, 15), (15, 16)]),
    ((189, 103, 148), [(0, 17), (17, 18), (18, 19), (19, 20)]),
]

MAX_HANDS = 2  # pad per-frame arrays to this many hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps-target", type=float, default=10.0,
                    help="process ~this many frames per second (default 10)")
    ap.add_argument("--score", type=float, default=0.75,
                    help="min detection score to KEEP a hand (default 0.75)")
    ap.add_argument("--hand", choices=["right", "left", "both"], default="right",
                    help="which hand(s) to keep (default right)")
    ap.add_argument("--write-video", action="store_true",
                    help="also save a hand-only preview video")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    print("Building predictor (WiLoR backend)...")
    predictor = AnyHandPredictor(backend="wilor")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(fps / args.fps_target))
    print(f"fps={fps:.1f} size={width}x{height} -> processing every {step} frame(s)")

    writer = None
    out_path = None
    if args.write_video:
        out_path = video.with_name(video.stem + "_hand_only.mp4")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps_target, (width, height))

    frames_idx = []      # absolute frame index of each processed frame
    kp_all = []          # (MAX_HANDS, 21, 2) per processed frame
    score_all = []
    right_all = []
    nhands_all = []

    f_idx = 0
    processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % step == 0:
            hands = predictor.predict(frame)
            vis = frame.copy() if writer is not None else None

            kp_frame = np.full((MAX_HANDS, 21, 2), np.nan, dtype=np.float32)
            sc_frame = np.zeros(MAX_HANDS, dtype=np.float32)
            rt_frame = np.zeros(MAX_HANDS, dtype=bool)
            kept = 0

            for hand in hands:
                kp2d = AnyHandPredictor.project_3d_to_2d(
                    hand.keypoints_3d,
                    hand.cam_t,
                    hand.focal_length,
                    img_size=(frame.shape[1], frame.shape[0]),
                )

                want = (
                    args.hand == "both"
                    or (args.hand == "right" and hand.is_right)
                    or (args.hand == "left" and not hand.is_right)
                )
                if hand.score > args.score and want and kept < MAX_HANDS:
                    kp_frame[kept] = kp2d.astype(np.float32)
                    sc_frame[kept] = float(hand.score)
                    rt_frame[kept] = bool(hand.is_right)
                    kept += 1

                if vis is not None:
                    k = kp2d.astype(int)
                    for color, bones in FINGER_BONES:
                        for i, j in bones:
                            cv2.line(vis, tuple(k[i]), tuple(k[j]),
                                     color=color, thickness=2, lineType=cv2.LINE_AA)

            frames_idx.append(f_idx)
            kp_all.append(kp_frame)
            score_all.append(sc_frame)
            right_all.append(rt_frame)
            nhands_all.append(kept)

            if writer is not None:
                writer.write(vis)

            processed += 1
            if processed % 10 == 0:
                total_kept = int(np.sum(nhands_all))
                print(f"  {processed} frames processed, {total_kept} hand detections kept")

        f_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved hand-only preview: {out_path}")

    frames_idx = np.array(frames_idx, dtype=np.int32)
    kp_all = np.stack(kp_all) if kp_all else np.zeros((0, MAX_HANDS, 21, 2), np.float32)
    score_all = np.stack(score_all) if score_all else np.zeros((0, MAX_HANDS), np.float32)
    right_all = np.stack(right_all) if right_all else np.zeros((0, MAX_HANDS), bool)
    nhands_all = np.array(nhands_all, dtype=np.int32)

    npz_path = video.with_name(video.stem + "_hand_kp.npz")
    np.savez_compressed(
        npz_path,
        frames=frames_idx,
        kp=kp_all,
        score=score_all,
        is_right=right_all,
        n_hands=nhands_all,
        fps=np.float32(fps),
        step=np.int32(step),
        width=np.int32(width),
        height=np.int32(height),
    )
    total = int(nhands_all.sum())
    print(f"Saved hand keypoints: {npz_path}")
    print(f"  {len(frames_idx)} processed frames, {total} hand detections kept")

    if total == 0:
        print("No qualifying detections. Try --score 0.5 or --hand both.")
        return

    # Wrist trajectory plot (joint 0 of the first kept hand each frame)
    wrist = kp_all[:, 0, 0, :]
    ok = ~np.isnan(wrist[:, 0])
    fig, ax = plt.subplots()
    sc = ax.scatter(wrist[ok, 0], -wrist[ok, 1],
                    c=frames_idx[ok], cmap="viridis")
    ax.set_title(f"Wrist trajectory ({args.hand} hand)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("-y (px)")
    fig.colorbar(sc, ax=ax, label="frame index")
    png_path = video.with_name(video.stem + "_hand_traj.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Saved trajectory plot: {png_path}")


if __name__ == "__main__":
    main()
