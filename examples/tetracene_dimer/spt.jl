using QCBase
using Printf
using FermiCG
using NPZ
using InCoreIntegrals
using RDM
using JLD2

@load  "data_cmf_TD_12.jld2"
M = 50
#@load "cmf_op_TD_with_ops.jld2"
display(clusters)
display(init_fspace)
ref_fspace = FockConfig(init_fspace)
ecore = ints.h0
cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3], ref_fspace, max_roots=M, verbose=1);
 
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
  
v = FermiCG.BSstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=10)
FermiCG.add_single_excitons_upto_L!(v,4)
FermiCG.add_double_excitons_upto_L!(v,4)
# FermiCG.add_1electron_transfers!(v)
FermiCG.add_spin_flip_states!(v,init_fspace)
FermiCG.eye!(v)

display(v)
# e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham, solver="davidson");
e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

v_bst = FermiCG.BSTstate(v_ci, thresh=1e-5)

display(v_bst)
FermiCG.randomize!(v_bst)
FermiCG.orthonormalize!(v_bst)
# FermiCG.eye!(v_bst)
display(v_bst)
σ = FermiCG.build_compressed_1st_order_state(v_bst, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-3)
σ = FermiCG.compress(σ, thresh=1e-5)
v2 = BSTstate(σ,R=10)
FermiCG.eye!(v2)
e_ci, v2 = FermiCG.ci_solve(v2, cluster_ops, clustered_ham);
e_var, v_var = FermiCG.block_sparse_tucker(v2, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-4,
                                               thresh_pt   = 1e-3,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
# e_ci2, v_ci2 = FermiCG.ci_solve(v_bst, cluster_ops, clustered_ham, solver="davidson");
# e_ci2, v_ci2 = FermiCG.ci_solve(v_bst, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);
