using FermiCG
using Printf
using Test
using LinearAlgebra

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

    #clusters    = reverse([(1:4),(5:8),(9:10),(11:12)])
    #init_fspace = reverse([(2,2),(2,2),(1,1),(1,1)])
    clusters    = [(1:4),(5:8),(9:10),(11:12)]
    init_fspace = [(2,2),(2,2),(1,1),(1,1)]


    max_roots = 8

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots) 

    display.(clusters)
    terms = FermiCG.extract_ClusteredTerms(ints, clusters)

    for t in keys(terms)
        #FermiCG.print_fock_sectors(collect(t))
        for tt in terms[t]
            display(tt)
        end
    end

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    fock_bra = FermiCG.FockConfig([(3,2),(1,2),(1,1),(1,1)])
    fock_ket = FermiCG.FockConfig([(2,2),(2,2),(1,1),(1,1)])
    bra = FermiCG.ClusterConfig([1,1,1,1])
    ket = FermiCG.ClusterConfig([2,1,1,1])


    ci_vector = FermiCG.ClusteredState(clusters)
    FermiCG.add_fockconfig!(ci_vector,init_fspace)
    FermiCG.print_configs(ci_vector)
    FermiCG.add_fockconfig!(ci_vector,fock_bra)
   
    
    FermiCG.add_fockconfig!(ci_vector,fock_ket)
    ci_vector[fock_ket][ket] = 1.1
    println("length(ci_vector):", length(ci_vector))
    println("length(clusters):", length(ci_vector.clusters))
    display(ci_vector)
    FermiCG.print_configs(ci_vector)
  
    display(ci_vector)
    FermiCG.normalize!(ci_vector)
    FermiCG.clip!(ci_vector)
    display(ci_vector)
    FermiCG.print_configs(ci_vector)
    
    FermiCG.zero!(ci_vector)
    FermiCG.clip!(ci_vector)
    display(ci_vector)
    FermiCG.print_configs(ci_vector)
    


    FermiCG.add_fockconfig!(ci_vector,[(2,2),(2,2),(0,1),(1,0)])
    FermiCG.add_fockconfig!(ci_vector,[(3,2),(1,2),(0,1),(1,0)])
    FermiCG.add_fockconfig!(ci_vector,[(3,2),(2,2),(0,1),(0,0)])
    #FermiCG.add_fockconfig!(ci_vector,reverse([(2,2),(2,2),(1,1),(0,0)]))
    #FermiCG.add_fockconfig!(ci_vector,reverse([(3,2),(1,2),(1,1),(0,0)]))

    FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
   
    display(ci_vector)
    #display(cluster_bases[1][(2,2)])
    
    function build(ci_vector, cluster_ops, terms)
        dim = length(ci_vector)
        H = zeros(dim, dim)
    
        bra_idx = 0
        for (fock_bra, configs_bra) in ci_vector.data
            for (config_bra, coeff_bra) in configs_bra
                bra_idx += 1
                ket_idx = 0
                for (fock_ket, configs_ket) in ci_vector.data
                    fock_trans = fock_bra - fock_ket
                  
                    # check if transition is connected by H
                    haskey(terms, fock_trans) || continue

                    for (config_ket, coeff_ket) in configs_ket
                        ket_idx += 1
                        ket_idx <= bra_idx || continue
        
                        for term in terms[fock_trans]
                            H[bra_idx, ket_idx] += FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                        end

                        H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                    end
                end
            end
        end
        return H
    end
    
    H = build(ci_vector, cluster_ops, terms)

    F = eigen(H)
    for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
        @printf(" %4i %12.8f\n", idx, Fi)
    end
#    display(H)
#    fock_bra = FermiCG.FockConfig([(2,2),(2,2),(1,1),(0,0)])
#    fock_ket = FermiCG.FockConfig([(3,2),(1,2),(1,1),(0,0)])
#    fock_trans = fock_bra - fock_ket
#    term = terms[fock_trans][1]
#    config_bra = [2,1,1,1]
#    config_ket = [1,1,1,1]
#    test = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
#    println(test)
#    FermiCG.print_configs(ci_vector)
#end

