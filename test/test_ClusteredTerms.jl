using FermiCG
using Printf
using Test

#@testset "ClusteredTerms" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[1,0,2]))
    push!(atoms,Atom(4,"H",[1,0,3]))
    push!(atoms,Atom(5,"H",[2,0,4]))
    push!(atoms,Atom(6,"H",[2,0,5]))
    push!(atoms,Atom(7,"H",[3,0,6]))
    push!(atoms,Atom(8,"H",[3,0,7]))
    push!(atoms,Atom(9,"H",[0,0,8]))
    push!(atoms,Atom(10,"H",[0,0,9]))
    push!(atoms,Atom(11,"H",[0,0,10]))
    push!(atoms,Atom(12,"H",[0,0,11]))
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    

    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,6,6)
    #@printf(" FCI Energy: %12.8f\n", e_fci)
    
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

    clusters    = [(1:4),(5:8),(9:10),(11:12)]
    init_fspace = [(2,2),(2,2),(1,1),(1,1)]


    max_roots = 4

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots) 

    display.(clusters)
    terms = FermiCG.extract_1e_terms(ints.h1, clusters)

    for t in keys(terms)
        #FermiCG.print_fock_sectors(collect(t))
        for tt in terms[t]
            display(tt)
        end
    end

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases);
   
    fock_bra = [(3,2),(1,2),(1,1),(1,1)]
    fock_ket = [(2,2),(2,2),(1,1),(1,1)]
    bra = [1,1,1,1]
    ket = [2,1,1,1]
    
    ci_vector = FermiCG.ClusteredState(clusters)
    FermiCG.add_fockconfig!(ci_vector,init_fspace)
    FermiCG.add_fockconfig!(ci_vector,fock_bra)
    FermiCG.add_fockconfig!(ci_vector,fock_ket)
    ci_vector[fock_ket][ket] = 1.1
    println(length(ci_vector))
    display(ci_vector)
    FermiCG.print_configs(ci_vector)
    
    FermiCG.normalize!(ci_vector)
    FermiCG.clip!(ci_vector)
    display(ci_vector)
    FermiCG.print_configs(ci_vector)
    
    FermiCG.zero!(ci_vector)
    FermiCG.clip!(ci_vector)
    display(ci_vector)
    FermiCG.print_configs(ci_vector)
    
    FermiCG.add_fockconfig!(ci_vector,[(2,2),(2,2),(1,1),(1,1)])
    FermiCG.add_fockconfig!(ci_vector,[(3,2),(1,2),(1,1),(1,1)])

    FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    println(" length: ", length(ci_vector))
    display(ci_vector)


    function build(ci_vector, cluster_ops)
        for (fock_bra, configs_bra) in ci_vector.data
            for (fock_ket, configs_ket) in ci_vector.data
                fock_trans = Tuple(fock_bra .- fock_ket)
                for (config_bra, coeff_bra) in configs_bra
                    for (config_ket, coeff_ket) in configs_ket
                        for term in terms[fock_trans]
                            me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, bra, fock_ket, ket)
                            println(me)
                        end
                    end
                end
            end
        end
    end

    build(ci_vector, cluster_ops)
#end
