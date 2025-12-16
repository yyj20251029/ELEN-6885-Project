# ELEN-6885-Project
- Deployed CityFlow in my experiments, and CityFlow does not support the latest Python verson. (Python 3.10 required.)   
- Train and tested on macOS (Apple Silicon), and used Visual Studio Code.
- Experimental results can be found @ "cityflow_single_intersection/logs/{model_name}_{flow_name}"  

## Setup the environments:
- change directory to file "cityflow_single_intersection"
- run the following code in terminal:
```bash
# Create and activate conda environment:  
conda create -n cityflow310 python=3.10 -y    
conda activate cityflow310  

# Install Packages
pip install torch torchvision torchaudio  
pip install tqdm

# Install build tools  
xcode-select --install  

# Install CMake (version should < 4)  
conda install -n cityflow310 -c conda-forge "cmake<4" -y  
cmake --version  

#   
pip install -r requirements.txt  
pip install git+https://github.com/cityflow-project/CityFlow.git  
python -c "import cityflow; print('cityflow OK')"  
```

## How to test @ terminal
- result will be saved in "cityflow_single_intersection/logs/"
- baseline test codes
``` bash
python baseline.py flow_low0.json
python baseline.py flow_low1.json
python baseline.py flow_low2.json
python baseline.py flow_low3.json
python baseline.py flow_low4.json
python baseline.py flow_medium0.json
python baseline.py flow_medium1.json
python baseline.py flow_medium2.json
python baseline.py flow_medium3.json
python baseline.py flow_medium4.json
python baseline.py flow_high0.json
python baseline.py flow_high1.json
python baseline.py flow_high2.json
python baseline.py flow_high3.json
python baseline.py flow_high4.json
```
- DQN test codes
```bash
python DQN.py test flow_low0.json
python DQN.py test flow_low1.json
python DQN.py test flow_low2.json
python DQN.py test flow_low3.json
python DQN.py test flow_low4.json
python DQN.py test flow_medium0.json
python DQN.py test flow_medium1.json
python DQN.py test flow_medium2.json
python DQN.py test flow_medium3.json
python DQN.py test flow_medium4.json
python DQN.py test flow_high0.json
python DQN.py test flow_high1.json
python DQN.py test flow_high2.json
python DQN.py test flow_high3.json
python DQN.py test flow_high4.json
```
- MP_DQN test codes
```bash
python MP_DQN.py test flow_low0.json
python MP_DQN.py test flow_low1.json
python MP_DQN.py test flow_low2.json
python MP_DQN.py test flow_low3.json
python MP_DQN.py test flow_low4.json
python MP_DQN.py test flow_medium0.json
python MP_DQN.py test flow_medium1.json
python MP_DQN.py test flow_medium2.json
python MP_DQN.py test flow_medium3.json
python MP_DQN.py test flow_medium4.json
python MP_DQN.py test flow_high0.json
python MP_DQN.py test flow_high1.json
python MP_DQN.py test flow_high2.json
python MP_DQN.py test flow_high3.json
python MP_DQN.py test flow_high4.json
```









## CityFlow Single Intersection (2-phase, no turns)

This project builds a minimal CityFlow simulation environment:
- Single isolated intersection (N/S/E/W)
- 1 incoming lane + 1 outgoing lane for each direction
- Straight-only routes (no left/right turns)
- Decision every second (1s)
- Episode length: 3600 steps
- min green = 3s, yellow = 1s (inserted by env state machine)
- Traffic generation: per-second Bernoulli(p) per approach (max 1 vehicle / step / direction)
  - Low p=0.061
  - Medium p=0.100
  - High p=0.139

## File overview
- cityflow_cfg/roadnet.json: road network + signal plan template
- cityflow_cfg/config.json: CityFlow config (points to roadnet + flow)
- scripts/gen_flow.py: generate flow JSON (low/medium/high)
- envs/cityflow_single_intersection_env.py: reset()/step() with min-green + yellow state machine
- run_sanity_check.py: run a fixed or random policy and print queues/throughput
