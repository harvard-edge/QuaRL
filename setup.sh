sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install swig ffmpeg
pip install -r requirements.txt
cd stable-baselines
pip install -e .[docs,tests]

