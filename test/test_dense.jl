using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5
using Random

using PyCall
pydir = joinpath(dirname(pathof(FermiCG)), "python")
pushfirst!(PyVector(pyimport("sys")."path"), pydir)
ENV["PYTHON"] = Sys.which("python")

#@testset "Clusters" begin

    atoms = []
    clusters = []
    na = 0
    nb = 0
    init_fspace = []
    
    function generate_H_ring(n,radius)
        theta = 2*pi/n

        atoms = []
        for i in 0:n-1
            push!(atoms,Atom(i+1,"H",[radius*cos(theta*i), radius*sin(theta*i), 0]))
        end
        return atoms
    end

    #
    # Test basic Tucker stuff
    Random.seed!(2);
    A = rand(4,6,3,3,5)
    tuck = FermiCG.Tucker(A, thresh=20, verbose=1)
    B = FermiCG.recompose(tuck)
    println()
    println(FermiCG.dims_large(tuck))
    @test all(FermiCG.dims_small(tuck) .== [4, 1, 3, 3, 1])
    @test all(FermiCG.dims_large(tuck) .== [4, 6, 3, 3, 5])
     
    A = rand(4,6,3,3,5)
    tuck = FermiCG.Tucker(A, thresh=-1, verbose=1)
    B = FermiCG.recompose(tuck)
    @test isapprox(abs.(A), abs.(B), atol=1e-12)


    if false 
        r = 1
        push!(atoms,Atom(1,"H",[0,0,0*r]))
        push!(atoms,Atom(2,"H",[0,0,1*r]))
        push!(atoms,Atom(3,"H",[0,1,2*r]))
        push!(atoms,Atom(4,"H",[0,1,3*r]))
        push!(atoms,Atom(5,"H",[0,2,4*r]))
        push!(atoms,Atom(6,"H",[0,2,5*r]))
        push!(atoms,Atom(7,"H",[0,3,6*r]))
        push!(atoms,Atom(8,"H",[0,3,7*r]))
        #push!(atoms,Atom(9,"H",[0,0,8*r]))
        #push!(atoms,Atom(10,"H",[0,0,9*r]))
        #push!(atoms,Atom(11,"H",[0,0,10*r]))
        #push!(atoms,Atom(12,"H",[0,0,11*r]))
    

        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
        na = 6
        nb = 6
        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        na = 4
        nb = 4
    elseif false 
        push!(atoms,Atom(1,"H",[-1.30,0,0.00]))
        push!(atoms,Atom(2,"H",[-1.30,0,1.00]))
        push!(atoms,Atom(3,"H",[ 0.00,0,0.00]))
        push!(atoms,Atom(4,"H",[ 0.00,0,1.00]))
        push!(atoms,Atom(5,"H",[ 1.33,0,0.00]))
        push!(atoms,Atom(6,"H",[ 1.30,0,1.00]))

        clusters    = [(1:2),(3:4),(5:6)]
        init_fspace = [(1,1),(1,1),(1,1)]
        na = 3
        nb = 3
    elseif true
       
        rad = 3
        
        atoms = generate_H_ring(12,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:8),(9:12)]
        init_fspace = [(2,2),(2,2),(2,2)]
        na = 6
        nb = 6
        
        atoms = generate_H_ring(8,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:6),(7:8)]
        init_fspace = [(2,2),(1,1),(1,1)]
        na = 4
        nb = 4
        
        atoms = generate_H_ring(10,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:6),(7:8),(9:10)]
        init_fspace = [(2,2),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:8),(9:10)]
        init_fspace = [(2,2),(2,2),(1,1)]
        na = 5
        nb = 5
        
    end

    basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
   
   
    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=2)
	
    #run fci with pyscf
    if false 
        pyscf = pyimport("pyscf")
        fci = pyimport("pyscf.fci")
        mp = pyimport("pyscf.mp")
        mp2 = mp.MP2(mf)
        cisolver = pyscf.fci.direct_spin1.FCI()
        cisolver.max_cycle = 100 
        cisolver.conv_tol = 1e-10 
        nelec = na + nb
        norb = size(ints.h1)[1]
        e_fci, ci = cisolver.kernel(ints.h1, ints.h2, norb , nelec, ecore=0, nroots = 1, verbose=100)
        e_fci = min(e_fci...)
        @printf(" FCI Energy: %12.8f\n", e_fci)
        
    end
   
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
    #e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
    #                                   max_iter_oo=40, verbose=0, gconv=1e-6, method="gd", alpha=1e-1)
    #ints = FermiCG.orbital_rotation(ints,U)
    
    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 40
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=Da, rdm1b=Db)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);


    
    p_spaces = Vector{FermiCG.ClusterSubspace}()
    q_spaces = Vector{FermiCG.ClusterSubspace}()
   
    #ci_vector = FermiCG.TuckerState(clusters)
    #FermiCG.add_fockconfig!(ci_vector, [(1,1),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(0,1),(2,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    
    #FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    
 
    for ci in clusters
        tss = FermiCG.ClusterSubspace(ci)
        tss[init_fspace[ci.idx]] = 1:1
        #tss[(2,2)] = 1:1
        #tss[(2,1)] = 1:1
        #tss[(1,2)] = 1:1
        #tss[(0,1)] = 1:1
        #tss[(1,0)] = 1:1
        push!(p_spaces, tss)
    end
    
    
    for tssp in p_spaces 
        tss = FermiCG.get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    println(" ================= Cluster P Spaces ===================")
    display.(p_spaces)
    println(" ================= Cluster Q Spaces ===================")
    display.(q_spaces)


    nroots = 1
    ci_vector = FermiCG.TuckerState(clusters, p_spaces, na, nb, nroots=nroots)
    ref_vector = deepcopy(ci_vector)

    # for FOI space 
    ci_vector = FermiCG.get_foi(ci_vector, clustered_ham, q_spaces, nbody=4) 
    FermiCG.print_fock_occupations(ci_vector)
    
    # for n-body Tucker
    #ci_vector = FermiCG.get_nbody_tucker_space(ci_vector, p_spaces, q_spaces, na, nb, nbody=4) 
    #FermiCG.print_fock_occupations(ci_vector)
    
    #
    # initialize with eye
    FermiCG.set_vector!(ref_vector, Matrix(1.0I, length(ref_vector),nroots))
    FermiCG.set_vector!(ci_vector, Matrix(1.0I, length(ci_vector),nroots))
        

    if true
        println(" Length of CI Vector: ", length(ci_vector))
        @time e_nb2, x_nb2 = FermiCG.tucker_ci_solve!(ci_vector, cluster_ops, clustered_ham)
        @printf(" E(CI):   Electronic %16.12f Total %16.12f\n", e_nb2[1], e_nb2[1]+ints.h0)
        FermiCG.print_fock_occupations(ci_vector)
    end
    
    if false 
        e_ref = FermiCG.tucker_ci_solve!(ref_vector, cluster_ops, clustered_ham)
        println(" Reference State:" )
        FermiCG.print_fock_occupations(ref_vector)

        @time e_cepa, x_cepa = FermiCG.tucker_cepa_solve!(ref_vector, ci_vector, cluster_ops, clustered_ham)
        @printf(" E(CEPA): Electronic %16.12f Total %16.12f\n", e_cepa, e_cepa+ints.h0)
        println(e_cepa)
        FermiCG.print_fock_occupations(ci_vector)
    end

    #FermiCG.compress_blocks(ci_vector)
    println(length(ci_vector))
    cts = FermiCG.CompressedTuckerState(ci_vector, thresh=1e-4)
    println(length(cts))

    display(cts)
    FermiCG.print_fock_occupations(cts)
    println(" Norm of projected state:           ", FermiCG.dot(cts,cts))
    println(" Norm of projected state: (nonorth) ", FermiCG.nonorth_dot(cts,cts))
    FermiCG.scale!(cts, 1.0/sqrt(FermiCG.dot(cts,cts)))
    println(" Norm of normalized state: ", FermiCG.dot(cts,cts))

    #@time e_nb2, v_nb2 = FermiCG.tucker_ci_solve!(cts, cluster_ops, clustered_ham)
    #@printf(" E(CI):   Electronic %16.12f Total %16.12f\n", e_nb2[1], e_nb2[1]+ints.h0)
    #FermiCG.print_fock_occupations(cts)

#end

