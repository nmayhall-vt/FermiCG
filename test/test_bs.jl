using FermiCG
using Printf
using Test
using JLD2 

@testset "BST vs BS" begin

    @load "_testdata_cmf_h12.jld2"
  
    v = FermiCG.BSstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=7)
    FermiCG.add_single_excitons!(v)
    FermiCG.add_1electron_transfers!(v)
    FermiCG.eye!(v)

    display(v)
    e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham, solver="davidson");
    #e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    v_bst = FermiCG.BSTstate(v_ci, thresh=1e-5)

    display(v_bst)
    FermiCG.randomize!(v_bst)
    FermiCG.orthonormalize!(v_bst)
    e_ci2, v_ci2 = FermiCG.ci_solve(v_bst, cluster_ops, clustered_ham, solver="davidson");
    #e_ci2, v_ci2 = FermiCG.ci_solve(v_bst, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    for r in 1:FermiCG.nroots(v)
        @test isapprox(e_ci[r], e_ci2[r], atol=1e-8)
    end
end

@testset "BST vs BS 2" begin
    @load "_testdata_cmf_he4.jld2"
  
    v = FermiCG.BSstate(clusters, FermiCG.FockConfig(init_fspace), cluster_bases, R=5)
    FermiCG.add_single_excitons!(v)
    FermiCG.add_1electron_transfers!(v)
    FermiCG.randomize!(v)
    FermiCG.orthonormalize!(v)

    #FermiCG.eye!(v)

    display(v)
    e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham);
    #e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    v_bst = FermiCG.BSTstate(v_ci, thresh=1e-5)

    display(v_bst)
    #e_ci, v_ci = FermiCG.ci_solve(v, cluster_ops, clustered_ham, solver="davidson");
    FermiCG.randomize!(v_bst)
    FermiCG.orthonormalize!(v_bst)
    e_ci2, v_ci2 = FermiCG.ci_solve(v_bst, cluster_ops, clustered_ham);
    #e_ci2, v_ci2 = FermiCG.ci_solve(v_bst, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    for r in 1:FermiCG.nroots(v)
        @test isapprox(e_ci[r], e_ci2[r], atol=1e-8)
    end
end
