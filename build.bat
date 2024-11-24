@echo off
python -V -V
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install pyinstaller
pyinstaller MagiaTimeline.spec --noconfirm
xcopy dist\MagiaTimeline\_internal\move_to_root\* dist\MagiaTimeline /E /H /K /Y
rmdir /S /Q dist\MagiaTimeline\_internal\move_to_root
pause
