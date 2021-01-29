using FermiCG
using Printf
using Test
using LinearAlgebra

#@testset "Clusters" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))
    push!(atoms,Atom(5,"H",[0,0,4]))
    push!(atoms,Atom(6,"H",[0,0,5]))
    push!(atoms,Atom(7,"H",[0,0,6]))
    push!(atoms,Atom(8,"H",[0,0,7]))
    basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    
   
    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    
    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)
    
    # define clusters
    clusters    = [(1:2),(3:4),(5:6)]
    clusters    = [(1:2),(3:4),(5:6),(7:8)]
    


    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    max_roots = 20
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots)

    # create reference Tucker Block
    init_fspace = [(1,1),(1,1),(1,1)]
    p_space = [1,4,2]
    
    p_spaces = Vector{FermiCG.TuckerSubspace}()
    q_spaces = Vector{FermiCG.TuckerSubspace}()
   
    ci_vector = FermiCG.TuckerState(clusters)
    #FermiCG.add_fockconfig!(ci_vector, [(1,1),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(0,1),(2,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    
    FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    
    display(length(ci_vector))
    display(ci_vector, thresh=-1)
 
    for ci in clusters
        tss = FermiCG.TuckerSubspace(ci)
        tss[(1,1)] = 1:1
        #tss[(2,1)] = 1:1
        #tss[(1,2)] = 1:1
        #tss[(0,1)] = 1:1
        #tss[(1,0)] = 1:1
        push!(p_spaces, tss)
    end
    
    display.(p_spaces)
    
    for tssp in p_spaces 
        tss = FermiCG.get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    display.(q_spaces)


    p_vector = FermiCG.TuckerState(clusters, p_spaces, 3, 2, nroots=1)

    q_vector = FermiCG.TuckerState(clusters)
    for ci in clusters
        tmp_spaces = copy(p_spaces)
        tmp_spaces[ci.idx] = q_spaces[ci.idx]
        FermiCG.add!(q_vector, FermiCG.TuckerState(clusters, tmp_spaces, 3, 3))
    end
    if false
        for ci in clusters
            for cj in clusters
                ci.idx < cj.idx || continue
                tmp_spaces = copy(p_spaces)
                tmp_spaces[ci.idx] = q_spaces[ci.idx]
                tmp_spaces[cj.idx] = q_spaces[cj.idx]
                FermiCG.add!(q_vector, FermiCG.TuckerState(clusters, tmp_spaces, 3, 3))
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
                    FermiCG.add!(q_vector, FermiCG.TuckerState(clusters, tmp_spaces, 3, 3))
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
   
    if true 
        FermiCG.expand_to_full_space!(p_vector, cluster_bases, 3, 2)
        FermiCG.expand_each_fock_space!(p_vector, cluster_bases)
   
        println(" Length of Vector: ", length(p_vector))
        v = FermiCG.get_vector(p_vector)
        v = Matrix(1.0I,length(p_vector),length(p_vector))
        
        FermiCG.set_vector!(p_vector,v)

        #display(p_vector, thresh=-1)
    end
    FermiCG.randomize!(p_vector)
    FermiCG.randomize!(q_vector)


    FermiCG.orthogonalize!(p_vector)
    FermiCG.orthogonalize!(q_vector)

    S = FermiCG.dot(p_vector, p_vector)
    @test isapprox(S-I, zeros(size(S)), atol=1e-9)
    
    #S = FermiCG.dot(q_vector, q_vector)
    ##display(S - I)
    #@test isapprox(S-I, zeros(size(S)), atol=1e-10)

   
    display.(clusters)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    sigma_vector = deepcopy(p_vector)
    FermiCG.zero!(sigma_vector)

    S = FermiCG.dot(p_vector, sigma_vector)

    @time FermiCG.build_sigma!(sigma_vector, p_vector, cluster_ops, clustered_ham)
    #println(p_vector.data)
    #println(sigma_vector.data)
    
    #v = FermiCG.get_vector(p_vector)
    #s = FermiCG.get_vector(sigma_vector)

    H = FermiCG.dot(p_vector, sigma_vector)
    
    #display(diag(H))
    dim = size(H,1)


    F = eigen(H)
    for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
        @printf(" %4i %18.13f\n", idx, Fi)
    end
        
    println()

            

#end
