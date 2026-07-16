"""
hand_pose_video.py  -  AnyHand / WiLoR hand pose on a video (local, CPU).

Ported from BG_AnyHand_video.ipynb. Run from INSIDE the AnyHand folder so
that `scripts.rgb_predictor` and the cached checkpoints resolve:

    conda activate anyhand
    cd AnyHand
    python ..\\scripts\\hand_pose_video.py path\\to\\video.mp4

Outputs:
    <video>_hand_traj.npy   - (N,3) array of (x, y, frame_order) wrist positions
    <video>_hand_traj.png   - scatter plot of the wrist trajectory
    <video>_annotated.mp4   - (optional, with --write-video) skeleton overlay

The predictor is built ONCE and reused; no model is re-downloaded as long as
prepare_wilor.sh has already cached the checkpoints in this folder.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # harmless on CPU/Windows

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

from scripts.rgb_predictor import AnyHandPredictor

# MANO 21-joint skeleton, grouped by finger. cv2 uses BGR.
FINGER_BONES = [
    ((58,  61, 214), [(0, 1), (1, 2), (2, 3), (3, 4)]),        # thumb  - red
    ((51, 126, 242), [(0, 5), (5, 6), (6, 7), (7, 8)]),        # index  - orange
    ((75, 162,  44), [(0, 9), (9, 10), (10, 11), (11, 12)]),   # middle - green
    ((185, 128, 31), [(0, 13), (13, 14), (14, 15), (15, 16)]), # ring   - blue
    ((189, 103, 148), [(0, 17), (17, 18), (18, 19), (19, 20)]),# pinky  - purple
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--fps-target", type=float, default=10.0,
                    help="process ~this many frames per second (default 10)")
    ap.add_argument("--score", type=float, default=0.75,
                    help="min detection score to record a wrist point")
    ap.add_argument("--hand", choices=["right", "left", "both"], default="right",
                    help="which hand's wrist to track")
    ap.add_argument("--write-video", action="store_true",
                    help="also save an annotated .mp4 with the skeleton drawn")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    print("Building predictor (WiLoR backend)... this loads the cached checkpoint.")
    predictor = AnyHandPredictor(backend="wilor")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / args.fps_target))
    print(f"Video fps={fps:.1f}, processing every {step} frame(s).")

    writer = None
    if args.write_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = video.with_name(video.stem + "_annotated.mp4")
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps_target, (w, h))

    hand_pos = []
    f_idx = 0
    processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % step == 0:
            hands = predictor.predict(frame)
            vis = frame.copy()
            for hand in hands:
                kp2d = AnyHandPredictor.project_3d_to_2d(
                    hand.keypoints_3d,
                    hand.cam_t,
                    hand.focal_length,
                    img_size=(frame.shape[1], frame.shape[0]),
                ).astype(int)

                want = (
                    args.hand == "both"
                    or (args.hand == "right" and hand.is_right)
                    or (args.hand == "left" and not hand.is_right)
                )
                if hand.score > args.score and want:
                    hand_pos.append(kp2d[0])  # joint 0 = wrist

                for color, bones in FINGER_BONES:
                    for i, j in bones:
                        cv2.line(vis, tuple(kp2d[i]), tuple(kp2d[j]),
                                 color=color, thickness=2, lineType=cv2.LINE_AA)

            if writer is not None:
                writer.write(vis)

            processed += 1
            if processed % 10 == 0:
                print(f"  processed {processed} frames, {len(hand_pos)} wrist points so far")

        f_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video: {out_path}")

    if not hand_pos:
        print("No qualifying hand detections. Try lowering --score or --hand both.")
        return

    traj = np.array([(x, y, i) for i, (x, y) in enumerate(hand_pos)], dtype=float)
    npy_path = video.with_name(video.stem + "_hand_traj.npy")
    np.save(npy_path, traj)
    print(f"Saved trajectory array: {npy_path}  (shape {traj.shape})")

    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(traj[:, 0], -traj[:, 1], c=traj[:, 2], cmap="viridis")
    ax.set_title(f"Wrist trajectory ({args.hand} hand)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("-y (px)")
    fig.colorbar(sc, ax=ax, label="frame order")
    png_path = video.with_name(video.stem + "_hand_traj.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Saved trajectory plot: {png_path}")


if __name__ == "__main__":
    main()
