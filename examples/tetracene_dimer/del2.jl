
using QCBase
using Printf
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2

if false
@load  "data_cmf_TD_12.jld2"
M = 100
#@load "cmf_op_TD_with_ops.jld2"
display(clusters)
display(init_fspace)
ref_fspace = FockConfig(init_fspace)
ecore = ints.h0

cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [5,5], ref_fspace, max_roots=M, verbose=1);

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
@save "cmf_op_TD_with_ops.jld2" clusters init_fspace ints cluster_bases cluster_ops clustered_ham
nroots = 8

vcmf = BSTstate(clusters,FermiCG.FockConfig(init_fspace), cluster_bases, R=1);
end

σ = FermiCG.build_compressed_1st_order_state(vcmf, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-3)
σ = FermiCG.compress(σ, thresh=1e-3)
v2 = BSTstate(σ,R=3)
FermiCG.eye!(v2)
e_ci, v2 = FermiCG.ci_solve(v2, cluster_ops, clustered_ham);

σ = FermiCG.build_compressed_1st_order_state(v2, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-3)
σ = FermiCG.compress(σ, thresh=1e-4)
FermiCG.zero!(σ)
FermiCG.nonorth_add!(v2, σ)
e_ci, v2 = FermiCG.ci_solve(v2, cluster_ops, clustered_ham);

σ = FermiCG.build_compressed_1st_order_state(v2, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-3)
σ = FermiCG.compress(σ, thresh=1e-2)
FermiCG.zero!(σ)
FermiCG.nonorth_add!(v2, σ)
e_ci, v2 = FermiCG.ci_solve(v2, cluster_ops, clustered_ham);

