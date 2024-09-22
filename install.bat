@echo off
echo Creating virtual environment and installing dependencies...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo Installation complete. Press any key to exit.
pause
