brew install cmake openmpi
pip install -r requirements.txt
cd stable-baselines
pip install -e .[docs,tests]
mkdir original_agents
cp -r rl-baselines-zoo/trained_agents original_agents/

