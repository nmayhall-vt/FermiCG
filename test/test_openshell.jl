using QCBase
using RDM
using FermiCG
using ClusterMeanField
using Printf
using Test
using LinearAlgebra
using Profile 
using Random
using PyCall
using JLD2

function generate_h9_data()
    
    atoms = []
    push!(atoms,Atom(1,"H", [-2.20391,  2.04660, -0.33566]))
    push!(atoms,Atom(2,"H", [-1.80047,  1.62359,  1.09361]))
    push!(atoms,Atom(3,"H", [-2.18537,  0.75717, -0.13175]))
    push!(atoms,Atom(4,"H", [ 0.58616,  1.66196, -0.17070]))
    push!(atoms,Atom(5,"H", [ 1.42470,  0.57741, -0.82172]))
    push!(atoms,Atom(6,"H", [ 0.85300, -0.46223,  1.06591]))
    push!(atoms,Atom(7,"H", [ 2.06134,  1.54289,  0.16175]))
    push!(atoms,Atom(8,"H", [-0.32939,  0.23178,  1.71554]))
    push!(atoms,Atom(9,"H", [-0.54605, -0.66268,  0.50978]))

    atoms = []
    push!(atoms,Atom(1,"H", [1, 0, 0]))
    push!(atoms,Atom(2,"H", [2, 0, 0]))
    push!(atoms,Atom(3,"H", [3, 0, 0]))
    push!(atoms,Atom(4,"H", [1, 2, 0]))
    push!(atoms,Atom(5,"H", [2, 2, 0]))
    push!(atoms,Atom(6,"H", [3, 2, 0]))
    push!(atoms,Atom(7,"H", [1, 4, 0]))
    push!(atoms,Atom(8,"H", [2, 4, 0]))
    push!(atoms,Atom(9,"H", [3, 4, 0]))


    clusters    = [(1:3),(4:6),(7:9)]
    init_fspace = [(2,1),(1,2),(2,1)]
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    na = 5
    nb = 4
    
    basis = "sto-3g"
    mol     = Molecule(0,2,atoms,basis)

    nroots = 3 

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
    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)
    for e in e_fci
        @printf(" FCI Energy: %12.8f\n", e)
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
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin_h3_3.molden")
    flush(stdout)


    # Do CMF
    e_cmf, U, d1 = ClusterMeanField.cmf_oo(  ints, clusters, init_fspace, RDM1(n_orb(ints)),
                            verbose=0, gconv=1e-6, method="bfgs",sequential=true)
    ints = orbital_rotation(ints, U)

    @save "_testdata_cmf_h9.jld2" ints d1 clusters init_fspace e_fci
end

@testset "openshell_tpsci" begin
#function run_tpsci()
    # load data
    @load "_testdata_cmf_h9.jld2"
    
    ref_fock = FockConfig(init_fspace)

    # Do TPS
    M=100
    cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], ref_fock, max_roots=M, verbose=1);
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=M, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    nroots=3
    ci_vector = FermiCG.TPSCIstate(clusters, ref_fock, R=nroots)

    ci_vector = FermiCG.add_spin_focksectors(ci_vector)

    display(ci_vector)
    eci, v = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

    e0a, v0a = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
        thresh_cipsi = 1e-3, 
        thresh_foi   = 1e-5,
        thresh_asci  = -1);
    
    ept = FermiCG.compute_pt2_energy(v0a, cluster_ops, clustered_ham, thresh_foi=1e-8)
   
    tpsci_ref = [-14.05014150
                 -14.02155292
                 -14.00595447]
                 
    e2_ref = [-14.05028658
              -14.02164792
              -14.00602933]

    @test all(isapprox(tpsci_ref, e0a, atol=1e-8)) 
    @test all(isapprox(e2_ref, ept+e0a, atol=1e-8)) 
end

@testset "openshell_bst" begin
#function run_bst()
    # load data
    @load "_testdata_cmf_h9.jld2"
    
    
    ref_fock = FockConfig(init_fspace)

    # Do TPS
    M=100
    cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], ref_fock, max_roots=M, verbose=1);
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=M, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    nroots=3

    # BST
    #
    
    # start by defining P/Q spaces
    p_spaces = Vector{ClusterSubspace}()
   
    for ci in clusters
        ssi = ClusterSubspace(clusters[ci.idx])

        num_states_in_p_space = 1
        # our clusters are near triangles, with degenerate gs, so keep two states
        add_subspace!(ssi, ref_fock[ci.idx], 1:num_states_in_p_space)
        add_subspace!(ssi, (ref_fock[ci.idx][2], ref_fock[ci.idx][1]), 1:num_states_in_p_space) # add flipped spin
        push!(p_spaces, ssi)
    end

    ci_vector = BSTstate(clusters, p_spaces, cluster_bases, R=3) 
    
    na = 5
    nb = 4
    FermiCG.fill_p_space!(ci_vector, na, nb)
    FermiCG.eye!(ci_vector)
    e_ci, v = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham)

    ept = FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-8)
    
    if true 
        e_var, v_var = block_sparse_tucker( v, cluster_ops, clustered_ham,
                                           max_iter    = 20,
                                           max_iter_pt = 200,
                                           nbody       = 4,
                                           H0          = "Hcmf",
                                           thresh_var  = 1e-1,
                                           thresh_foi  = 1e-6,
                                           thresh_pt   = 1e-3,
                                           ci_conv     = 1e-5,
                                           ci_max_iter = 100,
                                           do_pt       = true,
                                           resolve_ss  = false,
                                           tol_tucker  = 1e-4,
                                           solver      = "davidson")

        e_ref = [-14.050153133150385
                 -14.021579538798385
                 -14.00597811927653]
        @test all(isapprox(e_ref, e_var, atol=1e-8)) 
    end
end

