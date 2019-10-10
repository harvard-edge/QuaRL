brew install cmake openmpi
pip install -r requirements.txt
cd stable-baselines
pip install -e .[docs,tests]

