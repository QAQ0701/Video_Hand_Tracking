@echo off
REM ============================================================
REM  setup_hand.bat  -  AnyHand / WiLoR hand-pose environment
REM  Run this ONCE from the Anaconda Prompt (miniforge).
REM  CPU-only Windows setup. No NVIDIA GPU required.
REM ============================================================

echo.
echo === Creating conda env "anyhand" (Python 3.10) ===
call conda create -y -n anyhand python=3.10 git
if errorlevel 1 goto :err

call conda activate anyhand
if errorlevel 1 goto :err

echo.
echo === Installing CPU PyTorch (pinned < 2.6 for WiLoR ckpt loading) ===
REM WiLoR loads checkpoints with weights_only semantics that need torch < 2.6.
pip install "torch<2.6" "torchvision<0.21" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :err

echo.
echo === Cloning AnyHand (with submodules) ===
if not exist AnyHand (
    git clone --recurse-submodules https://github.com/chen-si-cs/AnyHand.git
) else (
    echo AnyHand folder already exists, skipping clone.
)
if errorlevel 1 goto :err

cd AnyHand

echo.
echo === Installing WiLoR + downloading checkpoints + hand detector ===
REM prepare_wilor.sh needs bash; on Windows use "Git Bash" or WSL.
REM If you have Git for Windows installed, this works:
bash scripts/prepare_wilor.sh
if errorlevel 1 (
    echo.
    echo [WARN] prepare_wilor.sh failed - you probably don't have bash on PATH.
    echo Run it manually from Git Bash:  bash scripts/prepare_wilor.sh
    echo Continuing anyway...
)

echo.
echo === Extra Python deps for inference/plotting ===
pip install opencv-python matplotlib huggingface_hub numpy

echo.
echo === MANO hand model ===
echo By continuing you agree to the MANO license: https://mano.is.tue.mpg.de/license.html
python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/JonathanLehner/Colab-collection/releases/download/MANO/mano_v1_2.zip','mano_v1_2.zip')"
python -c "import zipfile; zipfile.ZipFile('mano_v1_2.zip').extractall('.')"
if not exist mano_data mkdir mano_data
move /Y mano_v1_2\models\MANO_RIGHT.pkl mano_data\
move /Y scripts\mano_assets\mano_mean_params.npz mano_data\
rmdir /S /Q mano_v1_2
del mano_v1_2.zip

echo.
echo ============================================================
echo  DONE. Models are cached inside the AnyHand folder.
echo  To use later:  conda activate anyhand  ^&^&  cd AnyHand
echo  Run:  python ..\scripts\hand_pose_video.py your_video.mp4
echo ============================================================
goto :eof

:err
echo.
echo [ERROR] Setup failed. See message above.
exit /b 1
