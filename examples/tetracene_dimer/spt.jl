using QCBase
using Printf
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2

@load  "data_cmf_TD_12.jld2"
M = 100
#@load "cmf_op_TD_with_ops.jld2"
display(clusters)
display(init_fspace)
ints = deepcopy(ints_cmf)
ref_fspace = FockConfig(init_fspace)
ecore = ints.h0

cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [5,5], ref_fspace, max_roots=M, verbose=1);

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
@save "cmf_op_TD_with_ops.jld2" clusters init_fspace ints cluster_bases cluster_ops clustered_ham
nroots = 8
ci_vector = BSTstate(clusters,FermiCG.FockConfig(init_fspace), cluster_bases, R=nroots);
#ci_vector = FermiCG.TPSCIstate(clusters, FermiCG.FockConfig(init_fspace), R=nroots);
#ci_vector = FermiCG.add_spin_focksectors(ci_vector)

# Add the lowest energy single exciton and biexciton to basis
#ci_vector=FermiCG.bst_single_excitonic_basis(FermiCG.FockConfig(init_fspace),ci_vector,R=nroots)
#ci_vector=FermiCG.bst_biexcitonic_basis(FermiCG.FockConfig(init_fspace),ci_vector,R=nroots)
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((1:1,1:1))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((2:2,1:1))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((1:1,2:2))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((1:1,3:3))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((3:3,1:1))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((2:2,2:2))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))


# Spin-flip states
fspace_0 = FermiCG.FockConfig(init_fspace)

## ba
tmp_fspace = FermiCG.replace(fspace_0, (1,2), ([6,4],[4,6]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.TuckerConfig((1:1,1:1))]=FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))

## ab
tmp_fspace = FermiCG.replace(fspace_0, (1,2), ([4,6],[6,4]))
FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][FermiCG.TuckerConfig((1:1,1:1))]=FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
FermiCG.eye!(ci_vector)
display(ci_vector)
e_ci, v = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham);
@save "data_ci.jld2" v e_ci
FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-5)
bst_energy, v = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,max_iter    =2000, thresh_var=1e-2, thresh_foi=1e-4,thresh_pt=1e-3,thresh_spin=1e-3,do_pt=false);
@save "data_bst_1e2.jld2" clusters init_fspace ints cluster_bases v bst_energy