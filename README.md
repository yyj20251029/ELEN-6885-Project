# ELEN-6885-Project
- Tested on macOS (Apple Silicon).  

I used CityFlow in my experiment, and CityFlow does not support the latest Python verson. (Python 3.10 required.)   
Tested on macOS (Apple Silicon), and used Visual Studio Code.  

run the following code in terminal:
### Create and activate conda environment  
conda create -n cityflow310 python=3.10 -y    
conda activate cityflow310  
pip install torch torchvision torchaudio  
pip install tqdm

### Install build tools  
xcode-select --install  
### Install CMake (version should < 4)  
conda install -n cityflow310 -c conda-forge "cmake<4" -y  
cmake --version  
  
pip install -r requirements.txt  
pip install git+https://github.com/cityflow-project/CityFlow.git  
python -c "import cityflow; print('cityflow OK')"  
