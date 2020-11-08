# FermiCG
A Julia package for course-grained electronic structure calculations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall.github.io/FermiCG.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall.github.io/FermiCG.jl/dev)
[![Build Status](https://github.com/nmayhall/FermiCG.jl/workflows/CI/badge.svg)](https://github.com/nmayhall/FermiCG.jl/actions)
[![Build Status](https://travis-ci.com/nmayhall/FermiCG.jl.svg?branch=master)](https://travis-ci.com/nmayhall/FermiCG.jl)
[![Coverage](https://codecov.io/gh/nmayhall/FermiCG.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nmayhall/FermiCG.jl)


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
	cd ../
	julia --project=./
	julia> using Pkg; Pkg.build("PyCall")
	```
### Notes
- Use ITensor for dense algorithm
