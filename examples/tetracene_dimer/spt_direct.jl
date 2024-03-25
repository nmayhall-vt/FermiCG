using QCBase
using Printf
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2


@load  "data_cmf_TD_12.jld2"

M = 50

display(clusters)
display(init_fspace)

ref_fspace = FockConfig(init_fspace)
ecore = ints.h0

cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3], ref_fspace, max_roots=M, verbose=1);

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
# 
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
# @save "cmf_op_TD.jld2" clusters init_fspace ints cluster_bases cluster_ops clustered_ham
nroots = 10
ci_vector = BSTstate(clusters,FermiCG.FockConfig(init_fspace), cluster_bases, R=nroots);


# # Add the lowest energy single exciton to basis

ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.TuckerConfig((1:1,1:1))] =
    FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
FermiCG.add_single_excitons!(ci_vector,FermiCG.FockConfig(init_fspace),nroots)
FermiCG.add_double_excitons!(ci_vector,FermiCG.FockConfig(init_fspace),nroots)

#electron transfer states
fspace_0 = FermiCG.FockConfig(init_fspace)
# FermiCG.add_1electron_transfers!(ci_vector, fspace_0, 1)
FermiCG.add_spin_flip_states!(ci_vector, fspace_0,1)
# # Spin-flip states
# ## ba
# tmp_fspace = FermiCG.replace(fspace_0, (1,2), ([4,2],[2,4]))
# FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
# ci_vector[tmp_fspace][FermiCG.TuckerConfig((1:1,1:1))]=FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
# ## ab
# tmp_fspace = FermiCG.replace(fspace_0, (1,2), ([2,4],[4,2]))
# FermiCG.add_fockconfig!(ci_vector, tmp_fspace)
# ci_vector[tmp_fspace][FermiCG.TuckerConfig((1:1,1:1))]=FermiCG.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
display(ci_vector.data)
FermiCG.eye!(ci_vector)
display(ci_vector)
# σ = FermiCG.build_compressed_1st_order_state(ci_vector, cluster_ops, clustered_ham, 
#                                     nbody=4,
#                                     thresh=1e-3)
# σ = FermiCG.compress(σ, thresh=1e-5)
# v2 = BSTstate(σ,R=10)
# FermiCG.eye!(v2)
e_ci, v2 = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham);
e_var, v_var = FermiCG.block_sparse_tucker(v2, cluster_ops, clustered_ham,
                                               max_iter    = 200,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-3,
                                               thresh_foi  = 1e-5,
                                               thresh_pt   = 1e-4,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
e_var, v_var = FermiCG.block_sparse_tucker(v_var, cluster_ops, clustered_ham,
                                               max_iter    = 200,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 5e-4,
                                               thresh_foi  = 1e-6,
                                               thresh_pt   = 1e-4,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
