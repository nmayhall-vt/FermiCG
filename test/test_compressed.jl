using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5
using Random
using PyCall

@testset "BSTstate" begin
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


    clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(2,2),(2,2),(2,2)]
    na = 6
    nb = 6


    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)


    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=1)
    e_fci = -18.33022092

    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    
    #
    # do CMF
    d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, 
                                       method="bfgs")
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 20

    #
    # form Cluster data
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, 
                                                       max_roots=max_roots, 
                                                       init_fspace=init_fspace, 
                                                       rdm1a=d1.a, rdm1b=d1.b)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);


    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)

    ref_vec = v

    e_var, v_var = FermiCG.block_sparse_tucker(ref_vec, cluster_ops, clustered_ham, nbody=2, 

                                               resolve_ss  = true,
                                               thresh_foi=sqrt(1e-7), 
                                               thresh_pt =sqrt(1e-5),
                                               thresh_var=sqrt(1e-4), 
                                               tol_ci=1e-6, 
                                               tol_tucker=1e-4, 
                                               do_pt=true)

    @test isapprox(e_var[1], -18.33000292416142, atol=1e-8)

    e_cepa, v_cepa = FermiCG.do_fois_cepa(ref_vec, cluster_ops, clustered_ham, thresh_foi=1e-4, max_iter=50, tol=1e-8)
    @test isapprox(e_cepa[1], -18.330092885975663, atol=1e-8)
    
    #e_cepa, v_cepa = FermiCG.do_fois_cepa(v_var, cluster_ops, clustered_ham, thresh_foi=1e-10, max_iter=50, tol=1e-8)
    #@test isapprox(e_cepa[1], -18.330092885975663, atol=1e-8)
    
    e_pt, v_pt = FermiCG.do_fois_pt2(ref_vec, cluster_ops, clustered_ham, thresh_foi=1e-4, max_iter=50, tol=1e-8)
    @test isapprox(e_pt, -18.32863783512617, atol=1e-8)

    e_ci, v_ci = FermiCG.tucker_ci_solve(v_cepa, cluster_ops, clustered_ham)
    @test isapprox(e_ci[1], -18.330044872505518, atol=1e-8)

    #v_pt, e_pt = FermiCG.hylleraas_compressed_mp2(v_cepa, ref_vec, cluster_ops, clustered_ham; tol=1e-8, max_iter=100, H0="Hcmf", verbose=1)
    #@test isapprox(e_pt, -18.32863783512617, atol=1e-8)
end

