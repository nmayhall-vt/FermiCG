# TPSCI 

Tensor Product Selected CI (TPSCI) is a method for approximating FCI on large active spaces using a basis of tensor products of many-body cluster states.
The algorithm consists of the following steps:
1. **CMF:** Optimize both orbitals and cluster ground states to obtain the variationally best single tensor product state wavefunction.
2. **Compute basis:** Compute up to `M` excited states in each Fock sector desired (defaults to all) for each cluster.  
   These are excited states of the CMF Hamiltonian, which is an effective 1-cluster Hamiltonian containing the 1RDM contributions from all other clusters.
3. **Form operators:** Compute matrix representations of all the 1, 2, and 3 creation/annihilation operator strings in the CMF cluster basis.
   ```math 
   \Gamma_{p^\dagger q\bar{r}}^{I,J} = \left<I\right.| \hat{p}^\dagger \hat{q}\hat{\bar{r}}\left. | J\right>
   ```
   where `I` and `J` are cluster states on the same cluster, with well defined particle number and spin-projection.


# Index
```@index
Pages = ["TPSCI.md"]
```
# Documentation 
```@autodocs
Modules = [FermiCG]
Pages   = ["tpsci_inner.jl","tpsci_outer.jl"]
Order   = [:type, :function]
Depth	= 2
```

