# patocin

Duckietown agent developed by Researches at Centro de Informatica UFPE


## Installation

First download the repository with its submodules

```
git clone https://github.com/goncamateus/patocin.git --recursive
```

### Installation with conda (recommended)

```
cd gym-duckietown
conda env create -f environment.yaml
```

### Installation with pip

```
cd gym-duckietown 
pip install -e .
cd ..
pip install -e .
```

### Testing

Test manual control

```
./scripts/manual_control.py --env-name Duckietown-udem1-v0
```

## Troubleshooting

### ImportError: Library "GLU" not found

```
sudo apt-get install freeglut3-dev
```

### Unknown encoder 'libx264' when using gym.wrappers.Monitor

```
conda install -c conda-forge ffmpeg

```
