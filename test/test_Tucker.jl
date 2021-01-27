using FermiCG
using Printf
using Test

#@testset "Clusters" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))
    push!(atoms,Atom(5,"H",[0,0,4]))
    push!(atoms,Atom(6,"H",[0,0,5]))
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
    


    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    max_roots = 20
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots)

    # create reference Tucker Block
    init_fspace = [(1,1),(1,1),(1,1)]
    p_space = [1,4,2]
    tucker_blocks = Vector{FermiCG.TuckerBlock}()
   
    ci_vector = FermiCG.TuckerState(clusters)
    FermiCG.add_fockconfig!(ci_vector, [(1,1),(1,1),(1,1)])
    FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    FermiCG.add_fockconfig!(ci_vector, [(0,1),(2,1),(1,1)])
    
    FermiCG.add_fockconfig!(ci_vector, [(2,1),(0,1),(1,1)])
    
    FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    
    display(length(ci_vector))
    display(ci_vector, thresh=-1)
    
    FermiCG.expand_to_full_space!(ci_vector, cluster_bases, 3, 3)
    
    display(length(ci_vector))
    display(ci_vector, thresh=-1)
    #display(FermiCG.dim(FermiCG.FockConfig(init_fspace)))

#    tmp = Vector{FermiCG.ClusterSubspace}()
#    for ci in clusters
#        push!(tmp, FermiCG.ClusterSubspace(ci,init_fspace[ci.idx][1],init_fspace[ci.idx][2],1,p_space[ci.idx]))
#    end
#    
#    
#    tb = FermiCG.TuckerBlock(tmp)
#
#    display(tb)
#    display(length(tb))
        

#end
