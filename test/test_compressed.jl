using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5
using Random
using PyCall

@testset "CompressedTuckerState" begin
    atoms = []

    r = 1
    a = 0
    push!(atoms,Atom(1,"H", [0,.0*a,0*r]))
    push!(atoms,Atom(2,"H", [0,.0*a,1*r]))
    push!(atoms,Atom(3,"H", [0,.1*a,2*r]))
    push!(atoms,Atom(4,"H", [0,.1*a,3*r]))
    push!(atoms,Atom(5,"H", [0,.2*a,4*r]))
    push!(atoms,Atom(6,"H", [0,.2*a,5*r]))
    push!(atoms,Atom(7,"H", [0,.3*a,6*r]))
    push!(atoms,Atom(8,"H", [0,.3*a,7*r]))
    push!(atoms,Atom(9,"H", [0,.4*a,8*r]))
    push!(atoms,Atom(10,"H",[0,.4*a,9*r]))
    push!(atoms,Atom(11,"H",[0,.5*a,10*r]))
    push!(atoms,Atom(12,"H",[0,.5*a,11*r]))


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
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=2)


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

    rdm1 = zeros(size(ints.h1))

    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 100
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=Da, rdm1b=Db)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);



    p_spaces = Vector{FermiCG.ClusterSubspace}()
    q_spaces = Vector{FermiCG.ClusterSubspace}()

    # define p spaces
    for ci in clusters
        tss = FermiCG.ClusterSubspace(ci)
        tss[init_fspace[ci.idx]] = 1:1
        push!(p_spaces, tss)
    end

    # define q spaces
    for tssp in p_spaces 
        tss = FermiCG.get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    println(" ================= Cluster P Spaces ===================")
    display.(p_spaces)
    println(" ================= Cluster Q Spaces ===================")
    display.(q_spaces)


    nroots = 1

    #
    # initialize with eye
    ref_vector = FermiCG.TuckerState(clusters, p_spaces, q_spaces, na, nb)
    FermiCG.set_vector!(ref_vector, Matrix(1.0I, length(ref_vector),nroots))

    ref_vec  = FermiCG.CompressedTuckerState(ref_vector, thresh=-1);


    e_var, v_var = FermiCG.solve_for_compressed_space(ref_vec, cluster_ops, clustered_ham, nbody=4, thresh_var=1e-4, thresh_foi=1e-6, tol_ci=1e-5, tol_tucker=1e-5)
    @test isapprox(e_var[1], -19.804923102794756, atol=1e-10)

    
    e_cepa, v_cepa = FermiCG.do_fois_cepa(v_var, cluster_ops, clustered_ham)
    @test isapprox(e_cepa[1], -19.80478393279389, atol=1e-10)
    
    e_ci, v_ci = FermiCG.tucker_ci_solve(v_cepa, cluster_ops, clustered_ham)
    @test isapprox(e_ci[1], -19.804823584163735, atol=1e-10)

end

