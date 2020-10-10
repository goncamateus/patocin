# patocin

Duckietown agent developed by Researches at Centro de Informatica UFPE


## Create Conda Env

```
conda create -n patocin python=3.6
conda activate patocin
```

## Install some dependencies to run	

```
pip install -e . # Install setup.py dependencies
```


## Install Duckietown Gym

```
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
pip3 install -e .
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