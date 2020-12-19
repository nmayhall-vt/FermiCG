using FermiCG
using Printf

#@testset "Clusters" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[10,100,2]))
    push!(atoms,Atom(4,"H",[10,100,3]))
    push!(atoms,Atom(5,"H",[20,200,4]))
    push!(atoms,Atom(6,"H",[20,200,5]))
    basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    

    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,1,1)
    @printf(" FCI Energy: %12.8f\n", e_fci)
    
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)


    clusters    = [(1:2),(3:4),(5:6)]
    #clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1)]

    max_roots = 20

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    cluster_bases = Vector{ClusterBasis}
    for ci in clusters
        println("")
        display(ci)
        sectors = FermiCG.possible_focksectors(ci)
   
        basis_i = ClusterBasis(ci) 
        for sec in sectors
            #@printf("(α,β) = (%i,%-i)\n",sec[1],sec[2])
            basis_i.basis[sec] = FermiCG.compute_cluster_eigenbasis(ints, ci, sec[1], sec[2], max_roots=max_roots)
        end
    end
#end

