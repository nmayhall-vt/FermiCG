using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using HDF5

using PyCall
pydir = joinpath(dirname(pathof(FermiCG)), "python")
pushfirst!(PyVector(pyimport("sys")."path"), pydir)
ENV["PYTHON"] = Sys.which("python")

#@testset "Clusters" begin

out_radii = []
out_fci = []
out_ref = []
out_nb2 = []
out_cepa = []
out_rhf = []
out_mp2 = []
n_steps = 40 
start = 1
stop = 5
stepsize = (stop-start)/n_steps

for step in 1:n_steps
    rad = start + (step-1) * stepsize 
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

    if false 
        r = 1
        push!(atoms,Atom(1,"H",[0,0,0*r]))
        push!(atoms,Atom(2,"H",[0,0,1*r]))
        push!(atoms,Atom(3,"H",[0,0,2*r]))
        push!(atoms,Atom(4,"H",[0,0,3*r]))
        push!(atoms,Atom(5,"H",[0,0,4*r]))
        push!(atoms,Atom(6,"H",[0,0,5*r]))
        push!(atoms,Atom(7,"H",[0,0,6*r]))
        push!(atoms,Atom(8,"H",[0,0,7*r]))
        push!(atoms,Atom(9,"H",[0,0,8*r]))
        push!(atoms,Atom(10,"H",[0,0,9*r]))
        push!(atoms,Atom(11,"H",[0,0,10*r]))
        push!(atoms,Atom(12,"H",[0,0,11*r]))
    

        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        na = 4
        nb = 4
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
        na = 6
        nb = 6
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
        atoms = generate_H_ring(8,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        na = 4
        nb = 4
        
        atoms = generate_H_ring(10,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1)]
        na = 5
        nb = 5
        
        atoms = generate_H_ring(12,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:8),(9:12)]
        init_fspace = [(2,2),(2,2),(2,2)]
        na = 6
        nb = 6
    end

    basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
   
   
    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    push!(out_rhf, mf.e_tot)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=2)
	
    #run fci with pyscf
    if false 
        pyscf = pyimport("pyscf")
        fci = pyimport("pyscf.fci")
        mp = pyimport("pyscf.mp")
        mp2 = mp.MP2(mf)
        push!(out_mp2, mp2.kernel()[1])
        cisolver = pyscf.fci.direct_spin1.FCI()
        cisolver.max_cycle = 100 
        cisolver.conv_tol = 1e-10 
        nelec = na + nb
        norb = size(ints.h1)[1]
        e_fci, ci = cisolver.kernel(ints.h1, ints.h2, norb , nelec, ecore=0, nroots = 1, verbose=100)
        e_fci = min(e_fci...)
        @printf(" FCI Energy: %12.8f\n", e_fci)

        push!(out_fci, e_fci + ints.h0)
    end
    push!(out_radii, rad)
   
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
    
    # define clusters
    
    #fname = "job.scr" 
    #fid = h5open(fname, "r")
    #Cl = read(fid["mo_coeffs"]) 
    #close(fid)
    #ints = FermiCG.pyscf_build_ints(mol,Cl, zeros(nbas,nbas));


    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    d1 = RDM1(n_orb(ints)) 
    #e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
    #                                   max_iter_oo=40, verbose=0, gconv=1e-6, method="gd", alpha=1e-1)
    #ints = FermiCG.orbital_rotation(ints,U)
    
    e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1, 
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    #fname = "job.scr" 
    #fid = h5open(fname, "w")
    #fid["mo_coeffs"] = Cl*U
    #close(fid)
  
    #continue
    #cmf_out = FermiCG.cmf_ci(ints, clusters, init_fspace, rdm1, verbose=1)
    #e_ref = cmf_out[1]
    
    e_ref = e_cmf - ints.h0

    max_roots = 20
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);


    
    p_spaces = Vector{FermiCG.ClusterSubspace}()
    q_spaces = Vector{FermiCG.ClusterSubspace}()
   
    #ci_vector = FermiCG.BSstate(clusters)
    #FermiCG.add_fockconfig!(ci_vector, [(1,1),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(0,1),(2,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    
    #FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    
 
    for ci in clusters
        tss = FermiCG.ClusterSubspace(ci)
        tss[(2,2)] = 1:1
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
    ci_vector = FermiCG.BSstate(clusters, p_spaces, na, nb, nroots=nroots)
    
    ref_vector = deepcopy(ci_vector)
    if true 
        for ci in clusters
            tmp_spaces = copy(p_spaces)
            tmp_spaces[ci.idx] = q_spaces[ci.idx]
            FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, nb))
        end
    end
    if true 
        for ci in clusters
            for cj in clusters
                ci.idx < cj.idx || continue
                tmp_spaces = copy(p_spaces)
                tmp_spaces[ci.idx] = q_spaces[ci.idx]
                tmp_spaces[cj.idx] = q_spaces[cj.idx]
                FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, na))
            end
        end
    end
    if false 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    ci.idx < cj.idx || continue
                    cj.idx < ck.idx || continue
                    tmp_spaces = copy(p_spaces)
                    tmp_spaces[ci.idx] = q_spaces[ci.idx]
                    tmp_spaces[cj.idx] = q_spaces[cj.idx]
                    tmp_spaces[ck.idx] = q_spaces[ck.idx]
                    FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, na))
                end
            end
        end
    end
    if false
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    for cl in clusters
                        ci.idx < cj.idx || continue
                        cj.idx < ck.idx || continue
                        ck.idx < cl.idx || continue
                        tmp_spaces = copy(p_spaces)
                        tmp_spaces[ci.idx] = q_spaces[ci.idx]
                        tmp_spaces[cj.idx] = q_spaces[cj.idx]
                        tmp_spaces[ck.idx] = q_spaces[ck.idx]
                        tmp_spaces[cl.idx] = q_spaces[cl.idx]
                        FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, nb))
                    end
                end
            end
        end
    end
   

    
    #S = FermiCG.dot(q_vector, q_vector)
    ##display(S - I)
    #@test isapprox(S-I, zeros(size(S)), atol=1e-10)

   

    # initialize with eye
    FermiCG.set_vector!(ref_vector, Matrix(1.0I, length(ref_vector),nroots))
    FermiCG.set_vector!(ci_vector, Matrix(1.0I, length(ci_vector),nroots))
   
    #FermiCG.randomize!(ci_vector, scale=1e-4)

    #FermiCG.orthogonalize!(ci_vector)
    
    S = FermiCG.dot(ci_vector, ci_vector)
    @test isapprox(S-I, zeros(size(S)), atol=1e-12)



    if true 
        #FermiCG.print_fock_occupations(ci_vector)
        println(" Length of CI Vector: ", length(ci_vector))
        @time e_nb2 = FermiCG.tucker_ci_solve!(ci_vector, cluster_ops, clustered_ham)
        push!(out_nb2, min(e_nb2[1]...) + ints.h0)
        #@time FermiCG.tucker_ci_solve!(ci_vector, cluster_ops, clustered_ham)
        FermiCG.print_fock_occupations(ci_vector)
        #display(ci_vector, thresh=-1)
        #display(FermiCG.get_vector(ci_vector))
    end
    
    if true 
        #FermiCG.print_fock_occupations(ref_vector)
        e_ref = FermiCG.tucker_ci_solve!(ref_vector, cluster_ops, clustered_ham)
        println(" Reference State:" )
        push!(out_ref,  min(e_ref[1]...) + ints.h0)
        FermiCG.print_fock_occupations(ref_vector)
        #FermiCG.print_fock_occupations(ci_vector)

        @time e_cepa = FermiCG.tucker_cepa_solve!(ref_vector, ci_vector, cluster_ops, clustered_ham)
        println(e_cepa[1])
        push!(out_cepa, e_cepa[1] + ints.h0)
        FermiCG.print_fock_occupations(ci_vector)
        display(ci_vector)
    end
end

            

#end
