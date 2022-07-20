using FermiCG
using Printf
using Test
using JLD2 

@load "_testdata_cmf_h12.jld2"

v = FermiCG.BSTstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=1)


e_var, v_var = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,
                                           max_iter    = 20,
                                           max_iter_pt = 200, 
                                           nbody       = 4,
                                           H0          = "Hcmf",
                                           thresh_var  = 1e-2,
                                           thresh_foi  = 1e-3,
                                           thresh_pt   = 1e-3,
                                           ci_conv     = 1e-5,
                                           do_pt       = true,
                                           resolve_ss  = false,
                                           tol_tucker  = 1e-4,
                                           solver      = "davidson")


