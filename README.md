<div align="left">
  <img src="docs/src/logo1.png" height="60px"/>
</div>

# FermiCG
A Julia package for course-grained electronic structure calculations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall-vt.github.io/FermiCG/)
[![Build Status](https://github.com/nmayhall-vt/FermiCG/workflows/CI/badge.svg)](https://github.com/nmayhall-vt/FermiCG/actions)
[![Coverage](https://codecov.io/gh/nmayhall-vt/FermiCG/branch/master/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/FermiCG)

## Details
`FermiCG` ("Fermionic Course-Graining") is a code for computing high-accuracy electronic states for molecular systems in a tensor product state (TPS) basis. Unlike in the traditional Slater determinant basis, a TPS basis can be chosen such that each basis vector has a considerable amount of electron correlation already included. As a result, the exact wavefunction in this basis can be considerably more compact. This increased compactness comes at the cost of a significant increase in complexity for determining matrix elements. So far, we have implemented multiple approach for discovering highly accurate wavefunctions in this TPS basis. This package includes:
1. `CMF` - Meaning "Cluster Mean-Field", this is simply a variational optimization of both orbital and cluster state parameters, minimizing the energy of a single TPS. This was originally proposed by Scuseria and coworkers [link](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.085101).
1. `CMF-PT2` - Second order PT2 correction on top of `CMF` using a barycentric Moller-Plesset-type partitioning.
1. `CMF-CEPA` - A CEPA-type formalism on top of CMF. First published [here](https://arxiv.org/abs/2206.02333).
1. `TPSCI` - this is a generalization of the CIPSI method to a TPS basis. Essentially, one starts with a small number of TPS functions, solves the Schrodinger equation in this small subspace, then uses perturbation theory to determine which TPS's to add to improve the energy. This is done iteratively until the results stop changing. First published [here](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00141).
1. `BST` - "Block-Sparse-Tucker"

### Download
Downlond FermiCG and change into main directory

```
git clone https://github.com/nmayhall-vt/FermiCG.git
cd FermiCG/
```

### Installation with Virtual Environment
Create python virtual environment which will hold the PYSCF executable

```julia
cd src/python
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
export TPSCI_PYTHON=$(which python)
cd ../../
julia --project=./ -tauto
julia> using Pkg; Pkg.build("PyCall")
```
where `-tauto` let's Julia pick the max number of threads. Use `-t N` to select `N` manually. Removing defaults to 1 thread.

Run tests

```
julia> Pkg.test()
```

### Installation with Conda
Create conda virtual environment which will hold the PYSCF executable and set path for python version

```julia
conda create -n my_env python=3.7 
conda activate my_env
conda install numpy
pip install pyscf
export PYTHON_PATH=$(which python)
export PYTHONPATH="/usr/bin/python" #this is also in my ~/.bash_profile
```

Start a Julia REPL and add Conda to install PYSCF using pip

```julia
julia --project=./ -tauto 
julia> using Pkg; Pkg.add("Conda")
julia> import Conda
julia> Conda.add("pip")
julia> Conda.pip_interop(true)
julia> Conda.pip("install", "pyscf")
julia> using Pkg; Pkg.build("PyCall")
```
where `-tauto` let's Julia pick the max number of threads. Use `-t N` to select `N` manually. Removing defaults to 1 thread. 

Run tests
        
```
julia> Pkg.test()
```

### Installation with Conda on Apple M1 Mac
Create conda virtual environment (specific to M1 chips) which will hold the PYSCF executable and set path for python version

```julia
CONDA_SUBDIR=osx-64 conda create -n myenv_x86 python=3.7
conda activate my_env_x86
conda config --env --set subdir osx-64
conda install numpy
pip install pyscf
export PYTHON_PATH=$(which python)
export PYTHONPATH="/usr/bin/python" #this is also in my ~/.bash_profile
```

Start a Julia REPL and add Conda to install PYSCF using pip

```julia
julia --project=./ -tauto 
julia> using Pkg; Pkg.add("Conda")
julia> import Conda
julia> Conda.add("pip")
julia> Conda.pip_interop(true)
julia> Conda.pip("install", "pyscf")
julia> using Pkg; Pkg.build("PyCall")
```
where `-tauto` let's Julia pick the max number of threads. Use `-t N` to select `N` manually. Removing defaults to 1 thread. 

Run tests
	
```
julia> Pkg.test()
```



