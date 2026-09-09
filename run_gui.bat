@echo off
REM Launch the Organoid Annotator GUI (TensorFlow OrganoID backend).
setlocal
set TF_USE_LEGACY_KERAS=1
cd /d "%~dp0"
".venv\Scripts\python.exe" "app\gui\organoid_annotator.py" %*
endlocal
