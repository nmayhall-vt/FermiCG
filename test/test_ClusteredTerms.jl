using FermiCG
using Printf
using Test
using LinearAlgebra

#@testset "ClusteredTerms" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,1,0]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))
    push!(atoms,Atom(5,"H",[0,0.4,4]))
    push!(atoms,Atom(6,"H",[0,0,5]))
    push!(atoms,Atom(7,"H",[0,0,6]))
    push!(atoms,Atom(8,"H",[0,0,7]))
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


    max_roots = 3 

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots) 

    display.(clusters)
    terms = FermiCG.extract_ClusteredTerms(ints, clusters)

#    for t in keys(terms)
#        #FermiCG.print_fock_sectors(collect(t))
#        print(t)
#        for tt in terms[t]
#            display(tt) 
#            display(typeof(tt))
#        end
#    end

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    ci_vector = FermiCG.TPSCIstate(clusters)
    FermiCG.add_fockconfig!(ci_vector,[(2,2),(2,2),(1,1),(1,1)])
    FermiCG.add_fockconfig!(ci_vector,[(1,2),(3,2),(2,1),(1,1)])
    FermiCG.add_fockconfig!(ci_vector,[(2,2),(2,2),(2,1),(1,1)])
    FermiCG.add_fockconfig!(ci_vector,[(2,1),(2,2),(1,2),(1,1)])
    FermiCG.add_fockconfig!(ci_vector,[(2,1),(2,2),(1,2),(1,1)])
    
    #FermiCG.add_fockconfig!(ci_vector,[(2,2),(0,2),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(0,2),(2,2),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(2,2),(0,1),(1,2),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(0,2),(2,1),(1,2),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(3,1),(1,3),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(1,3),(3,1),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(3,1),(2,2),(0,2),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(1,3),(2,2),(2,0),(1,1)])
    
    #FermiCG.add_fockconfig!(ci_vector,[(1,3),(2,1),(3,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(0,2),(4,2),(1,1),(1,1)])
    #FermiCG.add_fockconfig!(ci_vector,[(3,2),(1,2),(0,1),(1,0)])
    #FermiCG.add_fockconfig!(ci_vector,[(3,2),(2,2),(0,1),(0,0)])
    #FermiCG.add_fockconfig!(ci_vector,reverse([(2,2),(2,2),(1,1),(0,0)]))
    #FermiCG.add_fockconfig!(ci_vector,reverse([(3,2),(1,2),(1,1),(0,0)]))

    #FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)
    FermiCG.expand_to_full_space!(ci_vector, cluster_bases)
    
    display(ci_vector)
    #display(cluster_bases[2][(2,2)])
    
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
                    if haskey(terms, fock_trans) == false
                        ket_idx += length(configs_ket)
                        continue
                    end

                    for (config_ket, coeff_ket) in configs_ket
                        ket_idx += 1
                        #ket_idx <= bra_idx || continue
        
                        for term in terms[fock_trans]
                            H[bra_idx, ket_idx] += FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                        end
                    
                        #H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                    end
                end
            end
        end
        return H
    end

    if false 
        display(keys(terms))
        tmp = FermiCG.TransferConfig([(2,0),(-2,0),(0,0),(0,0)])
        display(haskey(terms,tmp))
        for i in terms[tmp]
            display(i)
        end

        throw(Exception)
    end
    @time H = build(ci_vector, cluster_ops, terms)


    maximum(abs.(H-H')) < 1e-14 || error("Hamiltonian not symmetric: ",maximum(abs.(H-H'))) 
    F = eigen(H)
    for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
        @printf(" %4i %18.13f\n", idx, Fi)
    end
    #display(round.(H[1:16,17:32]; digits=8))
    #display(round.(H[1:16,1:16]; digits=8))
    #    for i in 1:size(H,1)
#        #@printf("%18.12f\n", H[i,1])
#    end
#    fock = FermiCG.FockConfig([(2,2),(2,2),(1,1),(1,1)])
##    fock_bra = FermiCG.FockConfig([(2,2),(2,2),(1,1),(0,0)])
##    fock_ket = FermiCG.FockConfig([(3,2),(1,2),(1,1),(0,0)])
##    fock_trans = fock_bra - fock_ket
#    fock_trans = fock - fock
#    config_bra = [2,2,1,1]
#    config_ket = [1,1,1,1]
#    test = 0
#    for term in  terms[fock_trans]
#        test1 = FermiCG.contract_matrix_element(term, cluster_ops, fock, config_bra, fock, config_ket)
#        println(test1/2, term.ops)
#        global test += test1
#    end
#    println(test)
#    FermiCG.print_configs(ci_vector)
#end

