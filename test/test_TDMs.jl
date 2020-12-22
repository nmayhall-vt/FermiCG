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

    clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters    = [(1:2),(3:4),(5:6)]
    init_fspace = [(1,1),(1,1),(1,1)]
    
    clusters    = [(1:4),(5:6)]
    init_fspace = [(2,2),(1,1)]

    max_roots = 20

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, 
    #                                                   max_roots=2, init_fspace=init_fspace, delta_elec=2)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, max_roots=100) 

    cluster_ops = Vector{Dict}(undef,length(clusters))
    for ci in clusters
        cluster_ops[ci.idx] = Dict() 
    end

    display(cluster_ops)
    for ci in clusters
        println("")
        display(ci)
        ci_basis = cluster_bases[ci.idx]
        norbs = length(ci)

        dicti = Dict{Tuple,Array}()
        for na in 0:norbs
            for nb in 0:norbs
                fockbra = (na+1,nb)
                fockket = (na,nb)
                focktrans = (fockbra,fockket)
        

                if haskey(ci_basis, fockbra) && haskey(ci_basis, fockket)
                    basis_bra = ci_basis[fockbra]
                    basis_ket = ci_basis[fockket]
                    println(fockbra, "<-",fockket)
                    dicti[focktrans] = FermiCG.StringCI.compute_annihilation(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, "alpha")
                end
            end
        end
        tmp = Dict{String,Dict}()
        tmp["A"] = dicti
        merge!(cluster_ops[ci.idx],tmp)
    end
  
    ref1 =  [-0.0063314  -0.0497708  -0.156291  -0.156764
             -0.0497708  -0.131177   -0.343145  -0.383806
             -0.156291   -0.343145   -0.344804  -0.32713
             -0.156764   -0.383806   -0.32713   -0.129477]
    test =cluster_ops[1]["A"][((2,1),(1,1))][:,:,1] 
    @test isapprox(ref1, test, atol=1e-5)

end

