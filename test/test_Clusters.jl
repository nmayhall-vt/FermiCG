using FermiCG
using Printf

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
    cluster_bases = Vector{ClusterBasis}()
    for ci in clusters
        println("")
        display(ci)
        sectors = FermiCG.possible_focksectors(ci)
   
        basis_i = ClusterBasis(ci) 
        for sec in sectors
            v = Vector{Matrix{Float64}}()
            push!(v,FermiCG.compute_cluster_eigenbasis(ints, ci, sec[1], sec[2], max_roots=max_roots))
            basis_i.basis[sec] = v 
        end
        push!(cluster_bases,basis_i)
    end
    println("")
    for ci in clusters
        display(cluster_bases[ci.idx])
        for cbi in cluster_bases[ci.idx].basis
            #display(cbi)
        end
    end
#end

