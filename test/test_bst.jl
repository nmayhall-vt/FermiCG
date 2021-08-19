using FermiCG
using Printf
using Test

#@testset "BSTstate" begin
    @load "_testdata_cmf_h6.jld2"
    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    
    e_ci, v_ci = FermiCG.tucker_ci_solve(v, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.31710895, atol=1e-8)

   
    v = FermiCG.BSTstate(v,R=1)
    xspace  = FermiCG.build_compressed_1st_order_state(v, cluster_ops, clustered_ham, nbody=2, thresh=1e-2)
    xspace = FermiCG.compress(xspace, thresh=1e-2)
    display(size(xspace))
    FermiCG.zero!(xspace)
    FermiCG.nonorth_add!(v, xspace)
    v = FermiCG.BSTstate(v,R=2)

        
    FermiCG.eye!(v)
    vv = FermiCG.get_vectors(v)
    FermiCG.orthonormalize!(v)
    e_ci, v = FermiCG.tucker_ci_solve(v, cluster_ops, clustered_ham)
    e_var, v_var = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               max_iter_pt = 200, 
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = sqrt(1e-5),
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4)
       
#    for i in 1:4
#        v = FermiCG.compress(v, thresh=1e-3)
#        FermiCG.orthonormalize!(v)
#        xspace  = FermiCG.build_compressed_1st_order_state(v, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
#
#        display(size(xspace))
#        xspace = FermiCG.compress(xspace, thresh=1e-3)
#        display(size(xspace))
#
#        FermiCG.zero!(xspace)
#        FermiCG.nonorth_add!(v, xspace)
#        FermiCG.orthonormalize!(v)
#        e, v = FermiCG.tucker_ci_solve(v, cluster_ops, clustered_ham)
#    end
#
#end

if false
@testset "BST" begin


    @load "_testdata_cmf_h6.jld2"

    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)

    e_var, v_var = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               max_iter_pt = 200, 
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = sqrt(1e-5),
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = true,
                                               tol_tucker  = 1e-4)

    @test isapprox(e_var[1], -18.329455008361652, atol=1e-8)
    

    e_cepa, v_cepa = FermiCG.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_cepa)
    @test isapprox(e_cepa[1], -18.32979791111852, atol=1e-8)
    
    e_pt, v_pt = FermiCG.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_pt)
    @test isapprox(e_pt[1], -18.32697072976005, atol=1e-8)

    e_ci, v_ci = FermiCG.tucker_ci_solve(v_cepa, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.329649399280648, atol=1e-8)

end
end
