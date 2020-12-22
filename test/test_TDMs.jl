using FermiCG
using Printf
using Test

@testset "TDMs" begin
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

    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, 
    #                                                   max_roots=2, init_fspace=init_fspace, delta_elec=2)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters) 


    norbs = size(ints.h1,1)
    for ci in clusters
        println("")
        display(ci)
        ci_basis = cluster_bases[ci.idx]
        for na in 0:norbs
            for nb in 0:norbs
                fockbra = (na+1,nb)
                fockket = (na,nb)
                focktrans = (fockbra,fockket)

                if haskey(ci_basis, fockbra) && haskey(ci_basis, fockket)
                    basis_bra = ci_basis[fockbra]
                    basis_ket = ci_basis[fockket]
                    println(fockbra, "<-",fockket)
                    op = compute_A(norbs, fockbra, fockket) 
                end
            end
        end
    end

end

