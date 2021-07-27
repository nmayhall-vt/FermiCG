# TPSCI 
## Background

Tensor Product Selected CI (TPSCI) approximates FCI on large active spaces using a sparse basis of tensor products of many-body cluster states.
The main idea is fold much of the electron correlation up into the basis vectors themselves, by diagonalizing local Hamiltonians (Hamiltonians acting on disjoint sets of orbitals, "clusters"), 
and using the tensor product space of these cluster states as our basis. 
For entangled clusters, the convergence of the global energy with the number of local cluster states is slow, making direct truncation of the cluster basis ineffective. 
However, instead of seeking a simple trunctation based on local information, 
we seek a sparse representation, such that only a small number of global states are needed to obtain an accurate approximation of the ground state. 

### Algorithm
The algorithm consists of the following steps:
1. **CMF:** Optimize both orbitals and cluster ground states to obtain the variationally best single tensor product state wavefunction.
2. **Compute basis:** Compute up to `M` excited states in each Fock sector desired (defaults to all) for each cluster.  
   These are excited states of the CMF Hamiltonian, which is an effective 1-cluster Hamiltonian containing the 1RDM contributions from all other clusters.
3. **Form operators:** Compute matrix representations of all the 1, 2, and 3 creation/annihilation operator strings in the CMF cluster basis. E.g.:
   ```math 
   \Gamma_{p^\dagger q\bar{r}}^{I,J} = \left<I\right| \hat{p}^\dagger \hat{q}\hat{\bar{r}}\left| J\right>
   ```
   where `I` and `J` are cluster states on the same cluster, with well defined particle number and spin-projection. 
4. **Initialize iterations:** Set iteration counter to zero ($n=0$). 
   Initialize TPSCI state with CMF wavefunction, in the current $\mathcal{P}$-space basis $\lbrace \left|P_i^0\right>\rbrace$, with the orthogonal complement defining the $\mathcal{Q}$-space, $\lbrace \left|Q_i^0\right>\rbrace$.
5. **Iterate Selected CI:** 
   1. Diagonalize $\hat{H}$ in the current $\mathcal{P}$-space, $\lbrace \left|P_i^n\right>\rbrace$
      ```math
      \hat{P}^n\hat{H}\left|\psi^{(0)}_n\right> = E_n\left|\psi^{(0)}_n\right>
      ```
   2. Form PT1 wavefunction by applying the Hamiltonian to the current variational state 
      ```math
      \left|\psi^{(1)}_n\right> = \hat{R}\left|\psi^{(0)}_n\right> = \sum_i c_i^{(1)}\left|Q_i^n\right>
      ```
      where, $\hat{R}$ is the relevant resolvant. 
   3. Select from  $\left|\psi^{(1)}_n\right>$ the coefficients with magnitude larger than `thresh_cipsi` and add to $\mathcal{P}$ space:
      ```math
      \lbrace\left|Q^n_i\right>\rbrace \xrightarrow{|c_i^{(1)}| > \epsilon}\lbrace\left|P^{n+1}_j\right>\rbrace 
      ```

### Tips on clustering 
## Performance considerations 
- Robust integral screening
- `thresh_asci` 
- HOSVD boot-strapping

## Index
```@index
Pages = ["TPSCI.md"]
```
## Documentation 
```@autodocs
Modules = [FermiCG]
Pages   = ["tpsci_inner.jl","tpsci_outer.jl","tpsci_matvec_thread.jl"]
Order   = [:type, :function]
Depth	= 2
```

