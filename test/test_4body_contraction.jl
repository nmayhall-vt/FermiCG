using FermiCG
using Printf
using Test

@testset "4-Body Term Contraction Bug" begin
    @load "_testdata_cmf_h8cmf_4c.jld2"

    ints = deepcopy(ints_cmf)
    ecore = ints.h0

    M = 500

    n_clusters = 4
    cluster_list = [[1,2], [3,4], [5,6], [7,8]]
    clusters = [MOCluster(i,collect(cluster_list[i])) for i = 1:length(cluster_list)]
    init_fspace = [ (2,2),(0,0), (2,2), (0,0)]
    display(clusters)
    ansatze = [FCIAnsatz(2, 2, 2), FCIAnsatz(2,0,0), FCIAnsatz(2,2,2), FCIAnsatz(2,0,0)]
    display(ansatze)

    ref_fock = FockConfig(init_fspace)

    cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [4,4,4,4], ref_fock, max_roots=M, verbose=1)

    nroots = 10
    ci_vector = FermiCG.TPSCIstate(clusters, FermiCG.FockConfig(init_fspace), R=nroots);
    FermiCG.expand_to_full_space!(ci_vector, cluster_bases, 4,4)
    FermiCG.eye!(ci_vector)

    # Build ClusteredOperator
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters);

    #
    # Build Cluster Operators
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    #
    # Add cmf hamiltonians for doing MP-style PT2 
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b, verbose=0);

    e0b, v0b = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham, max_iter=200, conv_thresh=1e-8);

    println()
    println("   *======TPSCI results======*")
    @printf("TPS CI Direct, Dim:%8d\n", size(v0b)[1])
    println()
    @printf("TCI %5s %12s\n", "Root", "E(0)")
    for r in 1:nroots
        @printf("TCI %5s %12.8f\n",r, e0b[r] + ecore)
    end

    clustered_S2 = FermiCG.extract_S2(ci_vector.clusters)

    println()
    println("   *======TPSCI S2 results======*")
    @printf(" %-50s", "Compute FINAL S2 expectation values: ")
    @time s2 = FermiCG.compute_expectation_value_parallel(v0b, cluster_ops, clustered_S2)

    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2")
    for r in 1:nroots
        @printf(" %5s %12.8f %12.8f\n",r, e0b[r]+ecore, abs(s2[r]))
    end

    ref_e = [-10.93391515595535
           -10.845837264644105
           -10.830535812785245
           -10.74991021289338
           -10.745066082039546
           -10.73667251242854
           -10.731093629439172
           -10.699595181161815
           -10.680721989881086
           -10.671162598999715]

    ref_s2 = [-3.541228720641168e-17
              1.9999999999999987
              2.000000000000003
              -4.7156592829574237e-17
              2.0000000000000058
              5.999999999999998
              1.9999999999999858
              -6.758642248484838e-17
              2.000000000000007
              -8.809660204725874e-17]

    @test isapprox(e0b, ref_e, atol=1e-12)
    @test isapprox(s2, ref_s2, atol=1e-12)
end


