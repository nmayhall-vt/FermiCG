using FermiCG
using Printf
using Test
using LinearAlgebra

#@testset "Clusters" begin
    atoms = []
    clusters = []
    na = 0
    nb = 0
    init_fspace = []
    if false
        push!(atoms,Atom(1,"H",[0,0,0]))
        push!(atoms,Atom(2,"H",[0,0,1]))
        push!(atoms,Atom(3,"H",[0,0,2]))
        push!(atoms,Atom(4,"H",[0,0,3]))
        push!(atoms,Atom(5,"H",[0,0,4]))
        push!(atoms,Atom(6,"H",[0,0,5]))
        push!(atoms,Atom(7,"H",[0,0,6]))
        push!(atoms,Atom(8,"H",[0,0,7]))
    

        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        na = 4
        nb = 4
    elseif true
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
    end

    basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
   
   
    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100)
    @printf(" FCI Energy: %12.8f\n", e_fci)
    
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
    


    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, 
                                       max_iter_oo=0, verbose=0, gconv=1e-6, method="cg")
    ints = FermiCG.orbital_rotation(ints,U)
    #cmf_out = FermiCG.cmf_ci(ints, clusters, init_fspace, rdm1, verbose=1)
    #e_ref = cmf_out[1]
    
    e_ref = e_cmf - ints.h0

    max_roots = 20
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=Da, rdm1b=Db)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    
    p_spaces = Vector{FermiCG.TuckerSubspace}()
    q_spaces = Vector{FermiCG.TuckerSubspace}()
   
    #ci_vector = FermiCG.TuckerState(clusters)
    #FermiCG.add_fockconfig!(ci_vector, [(1,1),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(0,1),(2,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    
    #FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    
 
    for ci in clusters
        tss = FermiCG.TuckerSubspace(ci)
        tss[(1,1)] = 1:1
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
    if true 
        for ci in clusters
            tmp_spaces = copy(p_spaces)
            tmp_spaces[ci.idx] = q_spaces[ci.idx]
            FermiCG.add!(ci_vector, FermiCG.TuckerState(clusters, tmp_spaces, na, nb))
        end
    end
    if true 
        for ci in clusters
            for cj in clusters
                ci.idx < cj.idx || continue
                tmp_spaces = copy(p_spaces)
                tmp_spaces[ci.idx] = q_spaces[ci.idx]
                tmp_spaces[cj.idx] = q_spaces[cj.idx]
                FermiCG.add!(ci_vector, FermiCG.TuckerState(clusters, tmp_spaces, na, na))
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
                    FermiCG.add!(ci_vector, FermiCG.TuckerState(clusters, tmp_spaces, na, na))
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
                        FermiCG.add!(q_vector, FermiCG.TuckerState(clusters, tmp_spaces, 3, 3))
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
    
    FermiCG.randomize!(ci_vector, scale=1e-4)

    FermiCG.orthogonalize!(ci_vector)
    
    S = FermiCG.dot(ci_vector, ci_vector)
    @test isapprox(S-I, zeros(size(S)), atol=1e-12)



    FermiCG.print_fock_occupations(ci_vector)
    @time FermiCG.tucker_ci_solve!(ci_vector, cluster_ops, clustered_ham)
    FermiCG.print_fock_occupations(ci_vector)
    display(ci_vector)
    
    FermiCG.print_fock_occupations(ref_vector)
    @time FermiCG.tucker_ci_solve!(ref_vector, cluster_ops, clustered_ham)
    FermiCG.print_fock_occupations(ref_vector)
    println(" Reference State:" )
    display(ref_vector)
    
    @time FermiCG.tucker_cepa_solve!(ref_vector, ci_vector, cluster_ops, clustered_ham)
    FermiCG.print_fock_occupations(ci_vector)
    display(ci_vector)


#    @time FermiCG.build_sigma!(sigma_vector, p_vector, cluster_ops, clustered_ham)
#    #println(p_vector.data)
#    #println(sigma_vector.data)
#    
#    #v = FermiCG.get_vector(p_vector)
#    #s = FermiCG.get_vector(sigma_vector)
#
#    H = FermiCG.dot(p_vector, sigma_vector)
#    
#    #display(diag(H))
#    dim = size(H,1)
#
#
#    F = eigen(H)
#    for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
#        @printf(" %4i %18.13f\n", idx, Fi)
#    end
#        
#    println()

            

#end
