using FermiCG
using InCoreIntegrals
using Printf
using Test
using JLD2

@testset "Clusters" begin

    @load "_testdata_hf_h6.jld2"

    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1)

    #
    #   Just test that the sum of all the basis vector matrices is reproduced
    println("")
    tst1 = 0
    for ci in clusters
        display(cluster_bases[ci.idx])
        for (sector,vecs) in cluster_bases[ci.idx].basis
            tst1 += sum(abs.(vecs.vectors))
        end
    end
    println(tst1)
    @test isapprox(tst1, 66.05063700792823, atol=1e-10)
   
    # now try with restrictions on fock space, and dimensions
    cluster_bases = Vector{ClusterBasis}()
    max_roots=2
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, 
                                                       max_roots=2, init_fspace=init_fspace, delta_elec=1)

    println("")
    tst1 = 0
    for ci in clusters
        display(cluster_bases[ci.idx])
        for (sector,vecs) in cluster_bases[ci.idx].basis
            tst1 += sum(abs.(vecs.vectors))
        end
    end
    println(tst1)
    @test isapprox(tst1, 43.466561233596934, atol=1e-10)
   
end

