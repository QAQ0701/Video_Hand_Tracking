# AnyHand + DeepLabCut — Local Windows Setup (CPU)

Persistent, offline-capable local version of `BG_AnyHand_video.ipynb`, producing
**one video with both the hand skeleton and the animal keypoints drawn on it**.

## How the combined render works

The two models can't run in one process (WiLoR needs `torch<2.6`, DeepLabCut
needs current PyTorch), so the pipeline decouples *detection* from *drawing*:

```
                 ┌─ hand_pose_video.py   (env: anyhand) ─┐
your_video.mp4 ──┤                                        ├──> merge_render.py ──> your_video_combined.mp4
                 └─ animal_pose_video.py (env: dlc)     ──┘
                        each writes a .npz of keypoints
```

Each detector saves keypoints to a `.npz`; `merge_render.py` reads both and
composites them onto the original frames. Nothing runs simultaneously, and you
can re-render (colors, sizes, labels) instantly without re-running inference.

### Frame alignment — the important detail

The hand pipeline subsamples to ~10 fps; DeepLabCut labels **every** frame.
`merge_render.py` uses the original video's frame index as the shared clock:

- animal keypoints index directly by frame number
- hand keypoints are **held** from the most recent processed frame, so the
  skeleton doesn't strobe. The hold defaults to the hand pipeline's subsample
  step; override with `--hand-hold N`.

## Why two environments?

| Env | Purpose | Python | PyTorch |
|-----|---------|--------|---------|
| `anyhand` | hand pose (WiLoR) | 3.10 | `<2.6` CPU |
| `dlc` | animal pose (DeepLabCut SuperAnimal) | 3.10 | current CPU |

`merge_render.py` needs only numpy + opencv, so it runs in **either** env.

## Prerequisites

- **Miniforge/Anaconda**
- **Git for Windows** — `prepare_wilor.sh` is a bash script, so `bash` must be
  on PATH. https://git-scm.com/download/win
- ~10 GB free disk for model weights.

## One-time setup

From the **Anaconda Prompt**, run each once:

```bat
setup_hand.bat
setup_animal.bat
```

These download and cache the AnyHand checkpoint, YOLO hand detector, MANO model,
and SuperAnimal quadruped weights. After this, nothing re-downloads.

> By running `setup_hand.bat` you agree to the MANO license:
> https://mano.is.tue.mpg.de/license.html . Keep `MANO_RIGHT.pkl` private.

## Running (three steps)

**1. Hand keypoints** — run from inside the `AnyHand` folder so imports resolve:

```bat
conda activate anyhand
cd AnyHand
python ..\scripts\hand_pose_video.py ..\my_video.mp4
```
→ `my_video_hand_kp.npz` + `my_video_hand_traj.png`

Options: `--hand both|left|right`, `--score 0.6`, `--fps-target 10`,
`--write-video` (hand-only preview).

**2. Animal keypoints:**

```bat
conda activate dlc
python scripts\animal_pose_video.py my_video.mp4
```
→ `my_video_animal_kp.npz` (+ DLC's own `.h5` / labeled mp4)

Options: `--pcutoff 0.3`, `--adapt` (slow!), `--skip-inference` (re-export the
`.npz` from an existing `.h5` without re-running the model).

**3. Merge into one video:**

```bat
python scripts\merge_render.py my_video.mp4
```
→ **`my_video_combined.mp4`** — hand skeleton + animal dots on the same frames.

Options:
| Flag | Effect |
|------|--------|
| `--hand-hold N` | frames each hand detection stays visible |
| `--hand-thickness` / `--hand-radius` | hand skeleton line/joint size |
| `--animal-radius` | animal dot size |
| `--label-parts` | write bodypart names next to the dots |
| `--no-legend` | hide the top-left legend |
| `--out PATH` | custom output path |

If one `.npz` is missing, it renders the other and tells you — so you can check
the hand overlay before the slow DLC run finishes.

### Colors

Hand bones use the notebook's per-finger scheme (thumb red, index orange, middle
green, ring blue, pinky purple); the wrist is a white dot. Animal keypoints are
yellow dots with black outlines. To make them match a different scheme, edit
`FINGER_BONES` / `ANIMAL_COLOR` at the top of `merge_render.py` — re-rendering is
seconds, no re-inference needed.

## CPU reality check

No NVIDIA GPU means detection is slow (drawing is fast):

- **WiLoR**: seconds per processed frame. 30 s clip at 10 fps ≈ 300 frames ≈
  10–40 min.
- **DeepLabCut without `--adapt`**: minutes for a short clip — usable.
- **DeepLabCut with `--adapt`**: can be **hours** on CPU. Overnight only.
- **merge_render.py**: seconds. Iterate on visuals freely.

Start with a 5–10 second test clip before committing to a long video.

If iteration speed becomes a wall, the intended path is a rented GPU (Colab Pro,
Lambda, RunPod; a T4/A10 is ~$0.40–0.80/hr). The `.npz` files are portable — run
detection on cloud GPU, download the `.npz`s, merge locally.

## What gets cached where

- AnyHand checkpoint + detector + MANO → the `AnyHand/` folder
- SuperAnimal weights → `%USERPROFILE%\.cache\huggingface` + local `full_dog/`

Back up those two locations to move machines without re-downloading.

## Troubleshooting

**"No DLC .h5 found"** — inference didn't finish, or the video is elsewhere. The
`.h5` lands next to the input video.

**Two `.h5` files** — SuperAnimal emits one before and one after adaptation. The
script picks the non-adapt one by default and the adapt one with `--adapt`; it
prints which it chose and which it ignored.

**Hand skeleton strobes/flickers** — raise `--hand-hold`.

**Video writer fails to open** — the `mp4v` codec is missing. Install ffmpeg
(`conda install -c conda-forge ffmpeg`) or change the fourcc in
`merge_render.py`.

**Wrong checkpoint paths** — run `cat scripts/prepare_wilor.sh` inside AnyHand to
confirm where it writes, in case the repo layout changed.
