using FermiCG
using Printf
using Test
using JLD2 

#@testset "BSTstate" begin
if false 
    @load "_testdata_cmf_h12_64bit.jld2"
    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    
    e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.31710895, atol=1e-8)

   
    v = FermiCG.BSTstate(v,R=1)
    xspace  = FermiCG.build_compressed_1st_order_state(v, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
    xspace = FermiCG.compress(xspace, thresh=1e-3)

    FermiCG.nonorth_add!(v, xspace)
    v = FermiCG.BSTstate(v,R=4)
    FermiCG.randomize!(v)
    FermiCG.orthonormalize!(v)
    
    e_ci, v = FermiCG.ci_solve(v, cluster_ops, clustered_ham)
    
    e_pt, v_pt = FermiCG.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    
    if true 
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
    end
       
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
#        e, v = FermiCG.ci_solve(v, cluster_ops, clustered_ham)
#    end
#
#end
end

@testset "BST" begin


    @load "_testdata_cmf_h12_64bit.jld2"
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

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

    @test isapprox(e_var[1], -18.32945552731658, atol=1e-8)
    

    e_cepa, v_cepa = FermiCG.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_cepa)
    @test isapprox(e_cepa[1], -18.329789530070542, atol=1e-8)
    
    e_pt = FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    
    e_pt, v_pt = FermiCG.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_pt)
    @test isapprox(e_pt[1], -18.326971095822113, atol=1e-8)

    e_ci, v_ci = FermiCG.ci_solve(v_cepa, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.329641438975646, atol=1e-8)

end
if false
end
