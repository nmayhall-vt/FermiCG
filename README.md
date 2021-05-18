<div align="left">
  <img src="docs/src/logo1.png" height="60px"/>
</div>

# FermiCG
A Julia package for course-grained electronic structure calculations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall-vt.github.io/FermiCG/stable)
[![Build Status](https://github.com/nmayhall-vt/FermiCG/workflows/CI/badge.svg)](https://github.com/nmayhall-vt/FermiCG/actions)
[![Coverage](https://codecov.io/gh/nmayhall-vt/FermiCG/branch/master/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/FermiCG)


### Installation
1. Download

	```julia
	git clone https://github.com/nmayhall-vt/FermiCG.git
	cd FermiCG/
	```


2. Create python virtual environment which will hold the PYSCF executable

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
2. Run tests
	```
	julia> Pkg.test()
	```
