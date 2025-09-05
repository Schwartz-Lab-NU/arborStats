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


## Install

> System prerequisites for `flatone` (e.g., SuiteSparse) still apply; follow flatone’s README.

```bash
pip install arborstats
