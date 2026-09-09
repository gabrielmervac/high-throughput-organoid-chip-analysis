@echo off
REM Fine-tune the OrganoID model on the staged ChipData set.
REM   run_train.bat                 -> stage data (if needed) + train + evaluate
REM   run_train.bat --eval-only     -> just evaluate the current fine-tuned model
setlocal
set TF_USE_LEGACY_KERAS=1
cd /d "%~dp0"
".venv\Scripts\python.exe" "app\stage_dataset.py"
".venv\Scripts\python.exe" "app\train_finetune.py" %*
endlocal
