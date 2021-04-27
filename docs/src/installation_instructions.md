# FermiCG
A Julia package for course-grained electronic structure calculations


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
cd ../../
julia --project=./
julia> using Pkg; Pkg.build("PyCall")
```

