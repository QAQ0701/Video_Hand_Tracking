# AnyHand + DeepLabCut — Local Windows Setup (CPU)

Persistent, offline-capable local version of `BG_AnyHand_video.ipynb`. Models
download **once** and are reused every run — no more rebuilding on Colab.

## Why two environments?

WiLoR needs `torch < 2.6` (old checkpoint loading); DeepLabCut wants a current
PyTorch. Those conflict, so they live in separate conda envs and just share
video files on disk.

| Env | Purpose | Python | PyTorch |
|-----|---------|--------|---------|
| `anyhand` | hand pose (WiLoR) | 3.10 | `<2.6` CPU |
| `dlc` | animal pose (DeepLabCut SuperAnimal) | 3.10 | current CPU |

## Prerequisites

- **Miniforge/Anaconda** (you already use miniforge — good).
- **Git for Windows** — needed because `prepare_wilor.sh` is a bash script.
  Install from https://git-scm.com/download/win so `bash` is on PATH.
- ~10 GB free disk for model weights.

## One-time setup

Open the **Anaconda Prompt** (miniforge) and run each once:

```bat
setup_hand.bat
setup_animal.bat
```

`setup_hand.bat` clones AnyHand, installs WiLoR, and downloads the AnyHand
checkpoint, the YOLO hand detector, and the MANO hand model — all cached inside
the `AnyHand` folder. `setup_animal.bat` installs DeepLabCut and pre-caches the
SuperAnimal quadruped weights.

> By running `setup_hand.bat` you agree to the MANO license:
> https://mano.is.tue.mpg.de/license.html . Keep `MANO_RIGHT.pkl` private.

## Running

**Hand pose** (run from inside the `AnyHand` folder so imports resolve):

```bat
conda activate anyhand
cd AnyHand
python ..\scripts\hand_pose_video.py my_video.mp4
python ..\scripts\hand_pose_video.py my_video.mp4 --write-video   REM also save skeleton overlay
python ..\scripts\hand_pose_video.py my_video.mp4 --hand both --score 0.6
```

Produces `my_video_hand_traj.npy` (the wrist trajectory) and
`my_video_hand_traj.png` (the scatter plot), matching the notebook.

**Animal pose:**

```bat
conda activate dlc
python scripts\animal_pose_video.py my_video.mp4
python scripts\animal_pose_video.py my_video.mp4 --adapt   REM fine-tune (SLOW on CPU)
```

Produces a `*_labeled.mp4` next to the input.

## CPU reality check

No NVIDIA GPU means inference is slow:

- **WiLoR**: seconds per processed frame. A 30 s clip at 10 fps ≈ 300 frames ≈
  10–40 min depending on your CPU.
- **DeepLabCut without `--adapt`**: minutes for a short clip — usable.
- **DeepLabCut with `--adapt`**: can be **hours** on CPU. Avoid unless overnight.

If iteration speed becomes the bottleneck, the intended path is a rented GPU
(Colab Pro, Lambda, RunPod — a T4/A10 is ~$0.40–0.80/hr). Develop locally with
these scripts, run heavy batches on cloud GPU.

## What gets cached where

- AnyHand checkpoint + detector + MANO → inside the `AnyHand/` folder (from
  `prepare_wilor.sh` and the MANO step). Back up this folder to keep them.
- SuperAnimal weights → Hugging Face / torch hub cache under your user profile
  (`%USERPROFILE%\.cache`) plus the local `full_dog/` folder.

Back up the `AnyHand/` folder and `%USERPROFILE%\.cache\huggingface` to move the
whole thing to another machine without re-downloading.
