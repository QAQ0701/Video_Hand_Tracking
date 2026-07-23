"""
merge_render.py  -  Draw hand skeleton + animal keypoints into ONE video.

STEP 3 of 3 in the combined-render pipeline.

Reads the .npz files produced by hand_pose_video.py and animal_pose_video.py
and composites both overlays onto the ORIGINAL video frames.

    python scripts\\merge_render.py path\\to\\video.mp4

Only needs numpy + opencv, so it runs in EITHER conda env (anyhand or dlc).

Key detail - frame alignment:
  The hand pipeline subsamples (~10 fps) while DLC labels EVERY frame. This
  script uses the original video's frame index as the shared clock:
    * animal keypoints index directly by frame number
    * hand keypoints are held from the most recent processed frame
      (--hand-hold controls how many frames a detection stays on screen)

Outputs:  <video>_combined.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# --- Hand skeleton (MANO 21 joints), BGR ---------------------------------
FINGER_BONES = [
    ((58,  61, 214), [(0, 1), (1, 2), (2, 3), (3, 4)]),        # thumb  red
    ((51, 126, 242), [(0, 5), (5, 6), (6, 7), (7, 8)]),        # index  orange
    ((75, 162,  44), [(0, 9), (9, 10), (10, 11), (11, 12)]),   # middle green
    ((185, 128, 31), [(0, 13), (13, 14), (14, 15), (15, 16)]), # ring   blue
    ((189, 103, 148), [(0, 17), (17, 18), (18, 19), (19, 20)]),# pinky  purple
]

ANIMAL_COLOR = (0, 255, 255)   # yellow dots
ANIMAL_TEXT = (255, 255, 255)


def load_hand(npz_path: Path):
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    return {
        "frames": d["frames"],
        "kp": d["kp"],
        "score": d["score"],
        "is_right": d["is_right"],
        "n_hands": d["n_hands"],
    }


def load_animal(npz_path: Path):
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    return {
        "kp": d["kp"],
        "conf": d["conf"],
        "bodyparts": [str(b) for b in d["bodyparts"]],
    }


def draw_hand(img, kp_frame, n_hands, thickness, radius):
    for h in range(int(n_hands)):
        k = kp_frame[h]
        if np.isnan(k).any():
            continue
        k = k.astype(int)
        for color, bones in FINGER_BONES:
            for i, j in bones:
                cv2.line(img, tuple(k[i]), tuple(k[j]), color=color,
                         thickness=thickness, lineType=cv2.LINE_AA)
        for color, bones in FINGER_BONES:
            for i, j in bones:
                cv2.circle(img, tuple(k[j]), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(k[0]), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_animal(img, kp_frame, bodyparts, radius, label):
    for k_i, (x, y) in enumerate(kp_frame):
        if np.isnan(x) or np.isnan(y):
            continue
        p = (int(x), int(y))
        cv2.circle(img, p, radius, ANIMAL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, (0, 0, 0), 1, cv2.LINE_AA)
        if label and k_i < len(bodyparts):
            cv2.putText(img, bodyparts[k_i], (p[0] + radius + 2, p[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ANIMAL_TEXT, 1,
                        cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="the ORIGINAL video (not a labeled one)")
    ap.add_argument("--hand-npz", default=None)
    ap.add_argument("--animal-npz", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hand-hold", type=int, default=None,
                    help="frames a hand detection persists (default: the "
                         "hand pipeline's subsample step, so it looks continuous)")
    ap.add_argument("--hand-thickness", type=int, default=2)
    ap.add_argument("--hand-radius", type=int, default=3)
    ap.add_argument("--animal-radius", type=int, default=4)
    ap.add_argument("--label-parts", action="store_true",
                    help="write bodypart names next to animal dots")
    ap.add_argument("--legend", action="store_true", default=True)
    ap.add_argument("--no-legend", dest="legend", action="store_false")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    hand_npz = Path(args.hand_npz) if args.hand_npz else video.with_name(video.stem + "_hand_kp.npz")
    animal_npz = Path(args.animal_npz) if args.animal_npz else video.with_name(video.stem + "_animal_kp.npz")

    hand = load_hand(hand_npz)
    animal = load_animal(animal_npz)

    if hand is None and animal is None:
        raise SystemExit(
            f"Neither keypoint file found:\n  {hand_npz}\n  {animal_npz}\n"
            "Run hand_pose_video.py and animal_pose_video.py first."
        )
    print(f"Hand keypoints:   {'OK ' + hand_npz.name if hand else 'MISSING (skipping)'}")
    print(f"Animal keypoints: {'OK ' + animal_npz.name if animal else 'MISSING (skipping)'}")

    # Map absolute frame index -> row in the hand arrays
    hand_lookup = {}
    hold = args.hand_hold
    if hand is not None:
        hf = hand["frames"]
        if hold is None:
            hold = int(np.median(np.diff(hf))) if len(hf) > 1 else 1
            hold = max(1, hold)
        for row, f in enumerate(hf):
            for d in range(hold):
                hand_lookup.setdefault(int(f) + d, row)
        print(f"Hand detections held for {hold} frame(s) each.")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.out) if args.out else video.with_name(video.stem + "_combined.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("Could not open the video writer (codec issue?).")

    print(f"Rendering {total} frames at {fps:.1f} fps -> {out_path.name}")

    n_animal = len(animal["kp"]) if animal else 0
    f_idx = 0
    drew_hand = drew_animal = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        vis = frame

        if animal is not None and f_idx < n_animal:
            before = np.count_nonzero(~np.isnan(animal["kp"][f_idx][:, 0]))
            if before:
                draw_animal(vis, animal["kp"][f_idx], animal["bodyparts"],
                            args.animal_radius, args.label_parts)
                drew_animal += 1

        if hand is not None and f_idx in hand_lookup:
            row = hand_lookup[f_idx]
            if hand["n_hands"][row] > 0:
                draw_hand(vis, hand["kp"][row], hand["n_hands"][row],
                          args.hand_thickness, args.hand_radius)
                drew_hand += 1

        if args.legend:
            cv2.rectangle(vis, (8, 8), (188, 56), (0, 0, 0), -1)
            cv2.putText(vis, "hand (MANO)", (34, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (14, 22), (28, 22), (51, 126, 242), 2, cv2.LINE_AA)
            cv2.putText(vis, "animal (DLC)", (34, 46), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, (21, 42), 4, ANIMAL_COLOR, -1, cv2.LINE_AA)

        writer.write(vis)
        f_idx += 1
        if f_idx % 100 == 0:
            print(f"  {f_idx}/{total} frames")

    cap.release()
    writer.release()

    print(f"\nDone -> {out_path}")
    print(f"  frames with hand overlay:   {drew_hand}")
    print(f"  frames with animal overlay: {drew_animal}")
    if animal is not None and n_animal < f_idx:
        print(f"  NOTE: animal predictions covered only {n_animal} of {f_idx} frames.")


if __name__ == "__main__":
    main()
