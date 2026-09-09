@echo off
REM General headless analyzer for ANY .nd2 or .tif/.tiff dataset.
REM All arguments are forwarded to app\analyze.py.
REM
REM Examples:
REM   run_analyze.bat experiment.nd2 --model models\organoid_finetuned_orgaug_v2_BEST --out results\experiment
REM   run_analyze.bat exp.ome.tif   --model models\organoid_finetuned_orgaug_v2_BEST --out results\exp --voxel-xy 0.65 --voxel-z 5 --time-step 4
setlocal
set TF_USE_LEGACY_KERAS=1
cd /d "%~dp0"
".venv\Scripts\python.exe" "app\analyze.py" %*
endlocal
