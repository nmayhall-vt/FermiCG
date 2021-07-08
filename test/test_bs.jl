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
    push!(atoms,Atom(4,"H", [0, 1.5*a, 3*r]))
    push!(atoms,Atom(5,"H", [0, 1.5*a, 4*r]))
    push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
    push!(atoms,Atom(7,"H", [0, 2*a, 6*r]))
    push!(atoms,Atom(8,"H", [0, 2*a, 7*r]))

    clusters    = [(1:8)]
    init_fspace = [(4,4)]


    clusters    = [(1:3),(4:5),(6:8)]
    init_fspace = [(2,0),(2,0),(2,0)]
    init_fspace = [(2,2),(1,1),(1,1)]
    init_fspace = [(2,1),(2,1),(2,1)]
    init_fspace = [(1,2),(1,2),(1,2)]


    na,nb = sum(init_fspace)
    
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
    
   

    e, d1a,d1b, d2aa, d2ab, d2bb = FermiCG.pyscf_fci_spin(ints,na,nb);
    e += ints.h0
    d2 = d2aa + d2bb + d2ab + permutedims(d2ab,(3,4,1,2))
    etest = FermiCG.compute_energy(ints, d1a+d1b, d2)
    @test isapprox(e, etest, atol=1e-12)

    d1sa = Dict()
    d1sb = Dict()
    d2saa = Dict()
    d2sab = Dict()
    d2sbb = Dict()
    d1sa[1] = d1a
    d1sb[1] = d1b
    d2saa[1] = d2aa
    d2sab[1] = d2ab
    d2sbb[1] = d2bb
    clusters_full = [Cluster(1,collect(1:8))]
    etest = FermiCG.compute_cmf_energy(ints, d1sa, d1sb, d2saa, d2sab, d2sbb, clusters_full) 
    #etest -= ints.h0
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

    rdm1a = zeros(size(ints.h1)) 
    rdm1b = zeros(size(ints.h1)) 
    for ci in clusters
        for p in ci.orb_list
            rdm1a[p,p] = init_fspace[ci.idx][1]./length(ci)
            rdm1b[p,p] = init_fspace[ci.idx][2]./length(ci)
        end
        #rdm1a[ci.orb_list,ci.orb_list] .= Matrix(I,length(ci),length(ci)).*init_fspace[ci.idx][1]./length(ci)
        #rdm1b[ci.orb_list,ci.orb_list] .= Matrix(I,length(ci),length(ci)).*init_fspace[ci.idx][2]./length(ci)
    end

    display(rdm1a)
    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1a, rdm1b, 
                                       max_iter_oo=0, verbose=1, gconv=1e-7, method="bfgs")
    #FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    #ints = FermiCG.orbital_rotation(ints,U)

    #Da = (Da + Db) / 2.0
    #Db = Da
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

    FermiCG.add_1excitonic_basis!(ci_vector, cluster_bases)

    H1X = FermiCG.build_full_H_parallel(ci_vector, cluster_ops, clustered_ham)
    display(H1X)
    #display(H1X[1:9,1:9])

    @time e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true, 
                                    thresh_cipsi=1e-3, thresh_foi=1e-9, thresh_asci=1e-4, conv_thresh=1e-5, 
                                    matvec=3,
                                    do_s2=false, thresh_s2=1e-12,
                                    max_iter=1);



