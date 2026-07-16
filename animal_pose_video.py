"""
animal_pose_video.py  -  DeepLabCut SuperAnimal (quadruped) pose on a video.

Ported from BG_AnyHand_video.ipynb (Animal Pose section). Local CPU run:

    conda activate dlc
    python scripts\\animal_pose_video.py path\\to\\video.mp4

Notes for CPU:
  * video_adapt=True (on-the-fly fine-tuning) is VERY slow on CPU.
    It is OFF by default here. Add --adapt only if you can wait (can be hours).
  * The first run downloads the SuperAnimal weights unless setup_animal.bat
    already pre-cached them. Subsequent runs reuse the cache.

Output: DeepLabCut writes a *_labeled.mp4 (or *_labeled_after_adapt.mp4 with
--adapt) plus prediction files next to the input video.
"""
import argparse
from pathlib import Path

import deeplabcut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--adapt", action="store_true",
                    help="enable video_adapt fine-tuning (SLOW on CPU)")
    ap.add_argument("--pcutoff", type=float, default=0.15,
                    help="confidence cutoff for drawing keypoints")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    videotype = video_path.suffix  # e.g. ".mp4"

    print(f"Running SuperAnimal quadruped on: {video_path}")
    print(f"  video_adapt={args.adapt}  pcutoff={args.pcutoff}")

    deeplabcut.video_inference_superanimal(
        [str(video_path)],
        "superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_mobilenet_v3_large_fpn",
        videotype=videotype,
        video_adapt=args.adapt,
        scale_list=[],
        pcutoff=args.pcutoff,
    )

    print("\nDone. Look for a *_labeled*.mp4 next to your input video.")


if __name__ == "__main__":
    main()
