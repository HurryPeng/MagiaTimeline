#!/bin/bash
echo "Creating virtual environment and installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
echo "Installation complete."
