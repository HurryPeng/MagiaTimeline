@echo off
python -V -V
python -c "import platform; print(platform.architecture()[0])" | findstr "64" > nul
if %errorlevel% neq 0 (
    echo You are using a 32-bit version of Python. MagiaTimeline does not support it. Please install a 64-bit version.
    pause
    goto end
)
echo Creating virtual environment and installing dependencies...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple/
python -m pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
if %errorlevel% neq 0 (
    echo Installation failed. Please check the error message above.
    pause
    goto end
)
echo Installation complete. 
pause
:end
