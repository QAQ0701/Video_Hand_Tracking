@echo off
REM ============================================================
REM  setup_animal.bat  -  DeepLabCut SuperAnimal environment
REM  Run this ONCE from the Anaconda Prompt (miniforge).
REM  CPU-only Windows setup. Uses the PyTorch backend.
REM ============================================================

echo.
echo === Creating conda env "dlc" (Python 3.10) ===
call conda create -y -n dlc python=3.10
if errorlevel 1 goto :err

call conda activate dlc
if errorlevel 1 goto :err

echo.
echo === ffmpeg (needed for reading/writing labeled videos) ===
call conda install -y -c conda-forge ffmpeg
if errorlevel 1 goto :err

echo.
echo === CPU PyTorch (current) ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :err

echo.
echo === DeepLabCut (PyTorch backend) + model zoo helper ===
pip install "deeplabcut" dlclibrary
if errorlevel 1 goto :err

echo.
echo === Pre-caching SuperAnimal quadruped weights ===
REM Downloads the model now so later runs are offline/instant.
REM DLC caches SuperAnimal weights under the huggingface / torch hub cache.
python -c "from dlclibrary import download_huggingface_model; from pathlib import Path; d=Path('full_dog'); d.mkdir(exist_ok=True); download_huggingface_model('full_dog', d); print('cached full_dog ->', d.resolve())"

echo.
echo ============================================================
echo  DONE. SuperAnimal weights cached.
echo  To use later:  conda activate dlc
echo  Run:  python scripts\animal_pose_video.py your_video.mp4
echo ============================================================
goto :eof

:err
echo.
echo [ERROR] Setup failed. See message above.
exit /b 1
