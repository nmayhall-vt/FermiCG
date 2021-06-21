# Wavefunction types

Each of the methods implemented in this package use a unique data representation for the wavefunction.
These all rely on some kind of hash table lookup, and so the following indexing schemes are defined to make this easier and
faster (i.e., implemented using only stack allocated data).

List of Indexing types for representing states
- `ClusterConfig`
- `TuckerConfig`
- `FockConfig`

Each of the different wavefunction types map different Index types to data (wavefunction coefficients)
- TPSCI: `ClusteredState`: Maps `FockConfig` → `ClusterConfig` →  Vector of coefficients (one for each state)
- BST: `CompressedTuckerState`: Maps `FockConfig` → `TuckerConfig` →  `Tucker` instance representing compressed set of coefficients for that subspace

## Index
```@index
Pages = ["States.md"]
```

## Types 
```@autodocs
Modules = [FermiCG]
Pages   = ["States.jl", 
	"FockSparse_ElementSparse.jl",
	"FockSparse_BlockSparse.jl",	
	"FockSparse_BlockSparseTucker.jl", 
	"Indexing.jl"]
Order   = [:type]
Depth	= 2
```

## Methods
```@autodocs
Modules = [FermiCG]
Pages   = ["States.jl", 
	"FockSparse_ElementSparse.jl",
	"FockSparse_BlockSparse.jl",	
	"FockSparse_BlockSparseTucker.jl", 
	"Indexing.jl"]
Order   = [:function]
Depth	= 2
```

