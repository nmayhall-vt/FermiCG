using FermiCG
using Printf
using Test
using JLD2 

if false
    @load "_testdata_cmf_h12.jld2"

    v = FermiCG.BSTstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=1)


    lambda_grid = [.0, .1, .2]
    lvec, evec, dvec, dims, times = FermiCG.compute_ac(v, cluster_ops, clustered_ham, lambda_grid, 
                                                       thresh_var = 1e-1,
                                                       thresh_foi = 1e-5,
                                                       thresh_pt = 1e-4
                                                      )
    dfit, efit = FermiCG.quadratic_fits(lvec, dvec, evec)
end


@load "_testdata_cmf_he4.jld2"

v = FermiCG.BSTstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=5)
FermiCG.add_single_excitons!(v, FermiCG.FockConfig(init_fspace), cluster_bases)
FermiCG.randomize!(v)
FermiCG.orthonormalize!(v)
e_ci, v = FermiCG.ci_solve(v, cluster_ops, clustered_ham);


lambda_grid = [.0, .05, .1]
lambda_grid = [.0, .1, .2, .3, .4, .5]
lambda_grid = [collect(0:10)...]/10.0
lvec, evec, dvec, dims, times = FermiCG.compute_ac(v, cluster_ops, clustered_ham, lambda_grid, 
                                                   thresh_var = 1e-2,
                                                   thresh_foi = 1e-5,
                                                   thresh_pt = 1e-4
                                                  )
dfit, efit = FermiCG.quadratic_fits(lvec, dvec, evec)
