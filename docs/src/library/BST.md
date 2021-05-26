# BST
The Block-Sparse Tucker (BST) method, approximates FCI as a linear combination of individually compressed (via HOSVD)
blocks of the Hilbert space.

## Background
Similar to TPSCI, the approach here starts with a CMF wavefunction, and systematically reintroduces the discarded tensor product
states to variationally approach FCI. Unlike with TPSCI, however, we don't assume that the final wavefunction is written as 
a purely sparse form (where only a few TPS's are needed), but rather we assume that collections of TPS's where certain numbers of clusters are "excited" can be efficiently compressed via HOSVD (although the basic idea would extend to other tensor decompositions, like CP or MPS). 

## Performance considerations 

## Index
```@index
Pages   = ["BST.md"]
```

## Documentation 
```@autodocs
Modules = [FermiCG]
Pages   = ["tucker_inner.jl","tucker_outer.jl","bst.jl"]
Order   = [:type, :function]
Depth	= 2
```

## HOSVD
```@autodocs
Modules = [FermiCG]
Pages   = ["hosvd.jl"]
Order   = [:type, :function]
Depth	= 2
```
