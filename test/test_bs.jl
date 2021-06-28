using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5
using Random
using PyCall
using Arpack

    atoms = []

    r = 1
    a = 1 
    push!(atoms,Atom(1,"H", [0, 1*a, 0*r]))
    push!(atoms,Atom(2,"H", [0, 1*a, 1*r]))
    push!(atoms,Atom(3,"H", [0, 1*a, 2*r]))
    push!(atoms,Atom(4,"H", [0, 1*a, 3*r]))
    push!(atoms,Atom(5,"H", [0, 1*a, 4*r]))
    push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
    push!(atoms,Atom(7,"H", [0, 2*a, 6*r]))
    push!(atoms,Atom(8,"H", [0, 2*a, 7*r]))


    clusters    = [(1:3),(4:5),(6:8)]
    init_fspace = [(2,1),(1,2),(1,2)]
    na = 4
    nb = 5

    clusters    = [(1:8)]
    init_fspace = [(4,4)]
    na = 4
    nb = 4

    init_fspace = [(2,2),(1,1),(1,1)]
    na = 4
    nb = 4

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
    if false 
        e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =200)


        for i in 1:length(e_fci)
            @printf(" %4i %12.8f %12.8f\n", i, e_fci[i], e_fci[i]+ints.h0)
        end
    end
    
   

    e, d1a,d1b, d2 = FermiCG.pyscf_fci(ints,na,nb);
    etest = FermiCG.compute_energy(ints, d1a+d1b, d2)
    @test isapprox(e+ints.h0, etest, atol=1e-12)

    d1s = Dict()
    d2s = Dict()
    d1s[1] = [d1a,d1b]
    d2s[1] = d2
    clusters_full = [Cluster(1,collect(1:8))]
    e += ints.h0
    etest = FermiCG.compute_cmf_energy(ints, d1s, d2s, clusters_full) 
    display(e)
    display(etest)
    @test isapprox(e, etest, atol=1e-12)



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
                                       max_iter_oo=40, verbose=0, gconv=1e-7, method="bfgs")
    #FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    Da = (Da + Db) / 2.0
    Db = Da
    e_ref = e_cmf - ints.h0

    max_roots = 100
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=Da, rdm1b=Db)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);


    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);
        
    ref_fock = FermiCG.FockConfig(init_fspace)


    ci_vector = FermiCG.ClusteredState(clusters, ref_fock, R=nroots)


    @time e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, 
                                    thresh_cipsi=1e-3, thresh_foi=1e-9, thresh_asci=1e-4, conv_thresh=1e-5, 
                                    matvec=3,
                                    do_s2=false, thresh_s2=1e-12);




