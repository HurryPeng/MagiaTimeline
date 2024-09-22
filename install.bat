@echo off
echo Creating virtual environment and installing dependencies...
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
echo Installation complete. 
pause
