@echo off
python -V -V
call venv\Scripts\activate.bat
pyinstaller MagiaTimeline.spec --noconfirm
xcopy dist\MagiaTimeline\_internal\move_to_root\* dist\MagiaTimeline /E /H /K /Y
rmdir /S /Q dist\MagiaTimeline\_internal\move_to_root
pause
