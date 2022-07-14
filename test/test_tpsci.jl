using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5
using Random
using PyCall
using Arpack
using JLD2

if false 
@testset "tpsci" begin
    atoms = []

    r = 1
    a = 1 
    push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
    push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
    push!(atoms,Atom(3,"H", [0, 1*a, 2*r]))
    push!(atoms,Atom(4,"H", [0, 1*a, 3*r]))
    push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
    push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
    push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
    push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
    push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
    push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
    push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
    push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))


    clusters    = [(1:2),(3:4),(5:6),(7:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1),(2,2)]
    clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(2,2),(2,2),(2,2)]
    na = 6
    nb = 6


    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)

    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
	
    @printf(" Do FCI\n")
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 200 
    cisolver.conv_tol = 1e-8
    nelec = na + nb
    norb = size(ints.h1,1)
    #e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)

    #e_fci = [-18.33022092,
    #         -18.05457644]
    e_fci  = [-18.33022092,
              -18.05457645,
              -18.02913047,
              -17.99661027
             ]

    for i in 1:length(e_fci)
        @printf(" %4i %12.8f %12.8f\n", i, e_fci[i], e_fci[i]+ints.h0)
    end


    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    #FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))

    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    #FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 100
    
    
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=Da, rdm1b=Db, T=Float64)

    
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);


    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);
        
    ref_fock = FermiCG.FockConfig(init_fspace)


    if true 

        ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)


        @time e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, 
                                  thresh_cipsi=1e-3, thresh_foi=1e-9, thresh_asci=1e-4, conv_thresh=1e-5);

        ref = [-18.32973618]

        @test isapprox(abs.(ref), abs.(e0), atol=1e-8)
    end
   
    nroots = 4

    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1])] = [0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1])] = [0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2])] = [0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                              thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    
    ci_vector4 = FermiCG.extract_roots(v0,[4])
    # todo now add root following for tpsci

    e2, v1 = FermiCG.compute_pt1_wavefunction(v0, cluster_ops, clustered_ham, thresh_foi=1e-8)

    ref = [-18.32932467
           -18.05349474
           -18.02775313
           -17.99514933
          ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-7)


    rotations = FermiCG.hosvd(v0, cluster_ops)
    for ci in clusters
        FermiCG.rotate!(cluster_ops[ci.idx], rotations[ci.idx])
        FermiCG.rotate!(cluster_bases[ci.idx], rotations[ci.idx])
        FermiCG.check_basis_orthogonality(cluster_bases[ci.idx])
    end

    #cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    #FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);


    e0a, v0a = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, 
                                thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2);

    

    H = FermiCG.build_full_H(v0a, cluster_ops, clustered_ham)
    sig1 = H*FermiCG.get_vector(v0a)
    sig2 = FermiCG.tps_ci_matvec(v0a, cluster_ops, clustered_ham)

    @test isapprox(norm(sig1-sig2), 0.0, atol=1e-12) 
    
    guess = deepcopy(v0a)
    FermiCG.randomize!(guess)
    FermiCG.orthonormalize!(guess)
    e0b, v0b = FermiCG.tps_ci_direct(guess, cluster_ops, clustered_ham, conv_thresh=1e-9);
    e0c, v0c = FermiCG.tps_ci_davidson(guess, cluster_ops, clustered_ham, conv_thresh=1e-9);

    @test isapprox(abs.(e0a), abs.(e0b), atol=1e-9)
    @test isapprox(abs.(e0a), abs.(e0c), atol=1e-9)
   
    println(" Now test pt2 correction")

    ref = [-18.32916288
           -18.05357935
           -18.02800015
           -17.99499973]

    e2a, v1a = FermiCG.compute_pt1_wavefunction(v0a, cluster_ops, clustered_ham, thresh_foi=1e-8)
    @test isapprox(abs.(ref), abs.(e0a+e2a), atol=1e-7)
    
    e2b = FermiCG.compute_pt2_energy(v0a, cluster_ops, clustered_ham, thresh_foi=1e-8)
    @test isapprox(abs.(ref), abs.(e0a+e2b), atol=1e-7)

        
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)
    ci_vector[ref_fock][ClusterConfig([2,1,1])] = [0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1])] = [0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2])] = [0,0,0,1]
    e0, ci_vector = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham, conv_thresh=1e-9);

    sig1 = FermiCG.open_matvec_serial(ci_vector, cluster_ops, clustered_ham, nbody=4, thresh=1e-9)
    sig2 = FermiCG.open_matvec_thread(ci_vector, cluster_ops, clustered_ham, nbody=4, thresh=1e-9)
        
    @test isapprox(norm(sig1), norm(sig2), atol=1e-12)

end
end


@testset "tpsci he 64bit" begin
    @load "_testdata_cmf_he4.jld2"
    
    nroots = 5

    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1,1])] = [0,1,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1])] = [0,0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1])] = [0,0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,2])] = [0,0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, ci_conv=1e-8,
                              thresh_cipsi=1e-3, thresh_foi=1e-8, thresh_asci=-1, conv_thresh=1e-7);
    
    e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-10)
    
    display(e0)
    display(e2)
    display(e0+e2)

    ref = [
           -16.886058282127408
           -15.435804238762836
           -15.42280922860447
           -15.422679313623284
           -15.409353983787529
          ]
    @test isapprox(abs.(ref), abs.(e0), atol=1e-8)
    
    ref = [
           -16.886190528051184
           -15.43619659959889
           -15.423267329074774
           -15.423025783287512
           -15.4097340230104
          ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-8)


end
@testset "tpsci h12 64bit" begin
    @load "_testdata_cmf_h12.jld2"
    
    nroots = 7

    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1,1,1])] = [0,1,0,0,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1,1])] = [0,0,1,0,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1,1])] = [0,0,0,1,0,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,3,1,1])] = [0,0,0,0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,2,1])] = [0,0,0,0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,1,1,2])] = [0,0,0,0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, ci_conv=1e-8,
                              thresh_cipsi=1e-2, thresh_foi=1e-5, thresh_asci=-1, conv_thresh=1e-7);
    
    e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-10)
    
    display(e0)
    display(e2)
    display(e0+e2)

    ref = [
           -18.32512226024639
           -18.04260833429895
           -18.016245886981604
           -17.986259649774958
           -17.95388664714469
           -17.92637656089058
           -17.909347539008866
          ]
    @test isapprox(abs.(ref), abs.(e0), atol=1e-8)
    
    ref = [
           -18.329242607660643
           -18.05229946775759
           -18.026861793675902
           -17.994775613520986
           -17.962143890203432
           -17.934857273405683
           -17.91769596347915
          ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-8)


end
@testset "tpsci h12 32bit" begin
    @load "_testdata_cmf_h12.jld2"
    
    ints = InCoreInts(ints, Float32)
    cluster_ops = [FermiCG.ClusterOps(co, Float32) for co in cluster_ops]
    cluster_bases = [ClusterBasis(cb, Float32) for cb in cluster_bases]
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)


    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);


    nroots = 4

    ref_fock = FermiCG.FockConfig(init_fspace)
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots, T=Float32)

    #1 excitons 
    ci_vector[ref_fock][ClusterConfig([2,1,1,1,1])] = [0,1,0,0]
    ci_vector[ref_fock][ClusterConfig([1,2,1,1,1])] = [0,0,1,0]
    ci_vector[ref_fock][ClusterConfig([1,1,2,1,1])] = [0,0,0,1]

    #e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=false,
    #                          thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                              thresh_cipsi=1e-2, thresh_foi=1e-4, thresh_asci=1e-2, conv_thresh=1e-4);
    
    e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-8)

    display(e0)
    display(e2)
    display(e0+e2)
    ref = [
           -18.32923698
           -18.05237389
           -18.02698708
           -17.99495125
          ]
    @test isapprox(abs.(ref), abs.(e0+e2), atol=1e-4)
end
