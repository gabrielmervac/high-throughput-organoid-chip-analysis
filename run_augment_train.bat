@echo off
REM Stage data -> augment the TRAINING split (ENHANCED recipe: geometry + magnify zoom
REM + size-weighting + images-only photometric) -> fine-tune on it (val/testing stay RAW).
REM
REM The augmented model is saved under a DISTINCT name so it does NOT overwrite the
REM current models\organoid_finetuned_BEST -- keep that as your A/B baseline.
REM
REM   run_augment_train.bat                  -> stage + augment(2000, enhanced) + train
REM   run_augment_train.bat -E 150 -P 15     -> extra args are forwarded to train_finetune
REM
REM Tuning / A-B:
REM   * more samples:   app\augment_data.py --count 3000
REM   * stock baseline: app\augment_data.py --recipe organoid
REM   then: run_train.bat --train-dir dataset\training_augmented
setlocal
set TF_USE_LEGACY_KERAS=1
cd /d "%~dp0"
".venv\Scripts\python.exe" "app\stage_dataset.py"
".venv\Scripts\python.exe" "app\augment_data.py" --count 2000
".venv\Scripts\python.exe" "app\train_finetune.py" --train-dir "dataset\training_augmented" --name organoid_finetuned_aug %*
endlocal
