@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM Batch-analyze the Paper Data ND2 files (headless, resumable, metrics-only).
REM
REM Reproduces the GUI "Analyze positions" pipeline with the fixed settings:
REM   model organoid_finetuned_orgaug_v2_BEST, illumination top-hat radius 200,
REM   flat-field none, per-Z+stitch, multiscale ON, defaults otherwise.
REM
REM Reads the ND2 files lazily over the network (no full download), keeps only
REM organoids fully inside each position's ROI (Mask\<stem>_Mask\NNN.tif), writes
REM D:\Yiyu\Paper_Data_Analysis\<stem>\<stem>_metrics.xlsx incrementally, and
REM RESUMES automatically (already-finished positions are skipped).
REM
REM Usage:
REM   run_batch_analysis.bat                 (all 5 files, all positions)
REM   run_batch_analysis.bat --files 260401  (one file)
REM   run_batch_analysis.bat --files 260601 260609 260617   (a subset)
REM ─────────────────────────────────────────────────────────────────────────────
setlocal
set TF_USE_LEGACY_KERAS=1
cd /d "%~dp0app\gui"
"%~dp0.venv\Scripts\python.exe" batch_analyze.py %*
endlocal
pause
