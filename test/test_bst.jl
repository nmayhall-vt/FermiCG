using FermiCG
using Printf
using Test

#@testset "BST" begin

    molecule = "
    H   0.0     0.0     0.0
    H   0.0     0.0     1.0
    H   0.0     1.0     2.0
    H   0.0     1.0     3.0
    H   0.0     2.0     4.0
    H   0.0     2.0     5.0
    H   0.0     3.0     6.0
    H   0.0     3.0     7.0
    H   0.0     4.0     8.0
    H   0.0     4.0     9.0
    H   0.0     5.0     10.0
    H   0.0     5.0     11.0
    "

    atoms = []
    for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
        l = split(line)
        push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
    end


    clusters    = [(1:2), (3:4), (5:8), (9:10), (11:12)]
    init_fspace = [(1, 1),(1, 1),(2, 2),(1, 1),(1, 1)]

    (na,nb) = sum(init_fspace)


    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)

    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=1);
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
    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)


    #
    # do CMF
    rdm1 = zeros(size(ints.h1))
    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
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
                                                       rdm1a=Da, rdm1b=Db)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);


    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)

    e_var, v_var = FermiCG.block_sparse_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               max_iter_pt = 200, 
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = sqrt(1e-5),
                                               tol_ci      = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = true,
                                               tol_tucker  = 1e-4)

    @test isapprox(e_var[1], -18.329455008361652, atol=1e-8)
    

    e_cepa, v_cepa = FermiCG.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_cepa)
    @test isapprox(e_cepa[1], -18.32979791111852, atol=1e-8)
    
    e_pt, v_pt = FermiCG.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_pt)
    @test isapprox(e_pt, -18.32697072976005, atol=1e-8)

    e_ci, v_ci = FermiCG.tucker_ci_solve(v_cepa, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.329649399280648, atol=1e-8)

#end
