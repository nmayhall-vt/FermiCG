using FermiCG
using Printf
using Test
using JLD2 

@load "_testdata_cmf_h12.jld2"

v = FermiCG.BSTstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=1)


lambda_grid = [.0, .1, .2]
lvec, evec, dvec, dims, times = FermiCG.compute_ac(v, cluster_ops, clustered_ham, lambda_grid, 
                                                   thresh_var = 1e-1,
                                                   thresh_foi = 1e-5,
                                                   thresh_pt = 1e-4
                                                  )
dfit, efit = FermiCG.quadratic_fits(lvec, dvec, evec)



@load "_testdata_cmf_he4.jld2"

v = FermiCG.BSTstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=5)


lambda_grid = [.0, .1, .2]
lvec, evec, dvec, dims, times = FermiCG.compute_ac(v, cluster_ops, clustered_ham, lambda_grid, 
                                                   thresh_var = 1e-1,
                                                   thresh_foi = 1e-5,
                                                   thresh_pt = 1e-4
                                                  )
dfit, efit = FermiCG.quadratic_fits(lvec, dvec, evec)
