using LinearAlgebra
using FermiCG
using Printf
using Arpack 
using Test

    atoms = []
    push!(atoms,Atom(1,"H",[0, 0, 0.1]))
    push!(atoms,Atom(2,"H",[0, 1,-1]))
    push!(atoms,Atom(3,"H",[0, 1, 1]))
    push!(atoms,Atom(4,"H",[0, 2, 0]))
    push!(atoms,Atom(5,"H",[0, 4, 0]))
    push!(atoms,Atom(6,"H",[0, 5,-1]))
    push!(atoms,Atom(7,"H",[0, 5, 1]))
    push!(atoms,Atom(8,"H",[0, 6, 0]))
    #basis = "6-31g"
    basis = "sto-3g"

    na = 4
    nb = 4

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

    FermiCG.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    clusters    = [(1:4),(5:8)]
    init_fspace = [(2,2),(2,2)]

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf*.5
    rdm1b = rdm_mf*.5

    rdm1a = rdm1
    rdm1b = rdm1

    display(rdm1a)
    display(rdm1b)
    

    #e_cmf, D1,D2,temp,temp2 = FermiCG.cmf_ci(ints, clusters, init_fspace, rdm1a, verbose=0)
    #Da = D1
    #Db = D1

    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    ints = FermiCG.orbital_rotation(ints,U)

    rdm1a = Da*.5
    rdm1b = Db*.5

    display(Da)
    println()
    display(Db)


    #for ci in clusters
    #    ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 
    #	print(ints_i.h1)
    #end


    cluster_bases = FermiCG.compute_cluster_est_basis2(ints, clusters, rdm1a, rdm1b, thresh_schmidt=5e-3, init_fspace=init_fspace)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, max_roots=1,rdm1a=Da,rdm1b=Db,init_fspace=init_fspace) 

    
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    ci_vector = FermiCG.ClusteredState(clusters)
    FermiCG.expand_to_full_space!(ci_vector, cluster_bases, na, nb)
    display(ci_vector,thresh=-1)
    display(cluster_bases[1][(2,1)])
    

    H = FermiCG.build_full_H(ci_vector, cluster_ops, clustered_ham)
    display(size(H))
    display(H)
    println()

    display(ci_vector,root=1)
    e,v = Arpack.eigs(H, nev = 15, which=:SR)
    for ei in e
        @printf(" Energy: %18.12f\n",real(ei))
    end
    
