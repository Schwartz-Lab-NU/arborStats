# arborstats

A command-line tool that automatically downloads the mesh of an EyeWire II neuron as `.obj` with [CaveClient/CloudVolume](https://github.com/seung-lab/cloud-volume), skeletonizes it as an `.swc` with [skeliner](https://github.com/berenslab/skeliner), flattens it with [pywarper](https://github.com/berenslab/pywarper) and computes arbor Stats using all all these information.

> __NOTE__
> 
> `flatone` relies on SuiteSparse, which does **NOT** run on native Windows. Use it on Unix-like enviroment or Windows Subsystem for Linux (WSL 2) instead.
>
> Instructions to install prerequisites for flatone
> ```bash
> # prerequisites
> ## mac
> brew update
> brew install suite-sparse
> 
> ## debian/ubuntu/WSL
> sudo apt-get update
> sudo apt-get install build-essential # if not already installed
> sudo apt-get install libsuitesparse-dev
>```


## Installation Instructions for ArborStats

> System prerequisites for `flatone` (e.g., SuiteSparse) still apply; follow the steps mentioned above.

```bash
# clone this repo
git clone git@github.com:Schwartz-Lab-NU/arborStats.git
cd arborStats

# install conda environment with Python=3.13
conda create -n arborstats python=3.13

# activate conda environment
conda activate arborstats

# installing dependencies using the following command
pip install -e .

# check if the installation worked
arborstats -h
```
