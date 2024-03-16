using ClusterMeanField
using RDM
using QCBase
using InCoreIntegrals
using PyCall
using ActiveSpaceSolvers
using LinearAlgebra
using Printf
using NPZ
using JLD2

h0 = npzread("./integrals_h0_12.npy")
h1 = npzread("./integrals_h1_12.npy")
h2 = npzread("./integrals_h2_12.npy")

ints = InCoreInts(h0, h1, h2)



na = 6
nb = 6

clusters = [[1,2,3,4,5,6],[7,8,9,10,11,12]]
init_fspace = [(3,3),(3,3)]


rdm1 = RDM1(n_orb(ints))

# define clusters
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
ansatze = [FCIAnsatz(6, 3, 3),FCIAnsatz(6, 3, 3)]
@time e_cmf, U, d1 = ClusterMeanField.cmf_oo_newton(ints, clusters, init_fspace,ansatze,rdm1, maxiter_oo = 400,
                           tol_oo=1e-8, 
                           tol_d1=1e-9, 
                           tol_ci=1e-11,
                           verbose=4, 
                           zero_intra_rots = true,
                           sequential=true)

ints = orbital_rotation(ints, U)

@save "data_cmf_TD_12.jld2" clusters init_fspace ints d1 e_cmf U 
