using FermiCG
using Printf
using Test
using LinearAlgebra

@testset "full_hbuild" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,1,0]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))
    push!(atoms,Atom(5,"H",[0,0.4,4]))
    push!(atoms,Atom(6,"H",[0,0,5]))
    push!(atoms,Atom(7,"H",[0,0,6]))
    push!(atoms,Atom(8,"H",[0,0,7]))
    #push!(atoms,Atom(9,"H",[0,0,8]))
    #push!(atoms,Atom(10,"H",[0,0,9]))
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    

    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,3,2,conv_tol=1e-10,max_cycle=100)
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

    
    clusters    = [(1:4),(5:8)]
    clusters    = [(1:4),(5:6),(7:10)]
    clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10)]
    clusters    = [(1:4),(5:6),(7:8)]

    max_roots = 100 

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots) 

    display.(clusters)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    ci_vector = FermiCG.ClusteredState(clusters)
#    FermiCG.add_fockconfig!(ci_vector,[(1,1),(1,0),(0,1)])
#    FermiCG.add_fockconfig!(ci_vector,[(1,1),(0,1),(1,0)])
#    
#    #FermiCG.add_fockconfig!(ci_vector,[(2,1),(0,0),(0,1)])
#    FermiCG.add_fockconfig!(ci_vector,[(1,1),(1,1),(0,0)])
#    FermiCG.add_fockconfig!(ci_vector,[(2,0),(1,0),(1,0)])
#    FermiCG.add_fockconfig!(ci_vector,[(2,2),(0,0),(0,0)])

    FermiCG.expand_to_full_space!(ci_vector, cluster_bases, 3, 2)
    
    display(ci_vector)
    #display(cluster_bases[2][(2,2)])
    
    function build(ci_vector, cluster_ops, clustered_ham)
        dim = length(ci_vector)
        H = zeros(dim, dim)
   
        zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
        bra_idx = 0
        for (fock_bra, configs_bra) in ci_vector.data
            for (config_bra, coeff_bra) in configs_bra
                bra_idx += 1
                ket_idx = 0
                for (fock_ket, configs_ket) in ci_vector.data
                    fock_trans = fock_bra - fock_ket

                    # check if transition is connected by H
                    if haskey(clustered_ham, fock_trans) == false
                        ket_idx += length(configs_ket)
                        continue
                    end

                    for (config_ket, coeff_ket) in configs_ket
                        ket_idx += 1
                        ket_idx <= bra_idx || continue
         
                        
                        for term in clustered_ham[fock_trans]
                            me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                            H[bra_idx, ket_idx] += me 
                        end
                    
                        H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                    end
                end
            end
        end
        return H
    end

    @time H = build(ci_vector, cluster_ops, clustered_ham)
    dim = size(H,1)


    F = eigen(H)
    for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
        @printf(" %4i %18.13f\n", idx, Fi)
    end
        
    println()
    
    @test isapprox(F.values[1], -9.2156766772454, atol=1e-10)
#
#    #FermiCG.print_configs(ci_vector)
#    for i in 1:dim
#        @printf("%12.8f",H[2,i])
#        for j in 1:dim
#            #@printf("%12.8f",H[i,j])
#        end
#        println()
#    end
    #@printf(" sum of H %12.8f\n", sum(abs.(H)))
    maximum(abs.(H-H')) < 1e-14 || error("Hamiltonian not symmetric: ",maximum(abs.(H-H'))); 
end


