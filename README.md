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
1. `CMF-PT2` - Second order PT2 correction on top of `CMF` using a barycentric Moller-Plesset-type partitioning.
1. `CMF-CEPA` - A CEPA-type formalism on top of CMF. First published [here](https://arxiv.org/abs/2206.02333).
1. `TPSCI` - this is a generalization of the CIPSI method to a TPS basis. Essentially, one starts with a small number of TPS functions, solves the Schrodinger equation in this small subspace, then uses perturbation theory to determine which TPS's to add to improve the energy. This is done iteratively until the results stop changing. First published [here](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00141).
1. `SPD` - "SubspaceProduct Decomposition"

## Download 
Download FermiCG and move into main directory

```
git clone https://github.com/nmayhall-vt/FermiCG.git
cd FermiCG/
julia --project=./ -tauto
```

Now test:
```julia
julia> using Pkg; Pkg.test()
```

## Install directly from GitHub 
Because FermiCG's dependencies are not (yet) in the registry, we will need to add them ourselves.
First, start the REPL,
```
julia --project="PROJECT_NAME" 
```
And then move to the package manager prommpt by typing "`]`", then add the following packages:

```
add https://github.com/nmayhall-vt/QCBase.jl
add https://github.com/nmayhall-vt/BlockDavidson.jl
add https://github.com/nmayhall-vt/InCoreIntegrals.jl
add https://github.com/nmayhall-vt/RDM.jl
add https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl
add https://github.com/nmayhall-vt/FermiCG
```