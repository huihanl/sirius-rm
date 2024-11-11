# Setup for policy training

## Setup codebases

### Changes to .bashrc:

for mujoco installation:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${home_dir}/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${home_dir}/.mujoco/mujoco200/bin
```

for running determinstic training:
```
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Installing robomimic

#### If using conda:
```
git clone git@github.com:huihanl/sirius-rm.git
cd sirius-rm
conda env create -f lflf.yml
pip install -e .
```

#### If using pip:
```
pip install -r requirements.txt
pip install -e .
```

### Installing robosuite

```
git clone git@github.com:huihanl/robosuite-hitl.git
cd robosuite-hitl
pip install -r requirements.txt
pip install -e .
```
