#!/bin/bash
python3 -c "import platform; print(platform.architecture()[0])" | grep "64" > /dev/null
if [ $? -ne 0 ]; then
    echo "You are using a 32-bit version of Python. MagiaTimeline does not support it. Please install a 64-bit version."
    exit 1
fi
echo "Creating virtual environment and installing dependencies..."
python3 -m venv venv || exit 1
source venv/bin/activate || exit 1
python3 -m pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple/ || exit 1
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ || exit 1
echo "Installation complete."
