using FermiCG
using Printf
using Test
using LinearAlgebra
using Arpack 
using StatProfilerHTML

#@testset "full_hbuild" begin
    atoms = []
  
    clusters = []
if true 
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
    clusters    = [(1:6),(7:10)]
    #clusters    = [(1:4),(5:6),(7:10)]
    clusters    = [(1:4),(5:8)]
    clusters    = [(1:4),(5:6),(7:8)]
    clusters    = [(1:2),(3:4),(5:6),(7:8)]
elseif true
    push!(atoms,Atom(1,"H",[-1.30,00,0.00]))
    push!(atoms,Atom(2,"H",[-1.30,00,1.00]))
    push!(atoms,Atom(3,"H",[ 0.00,30,0.00]))
    push!(atoms,Atom(4,"H",[ 0.00,30,1.00]))
    push!(atoms,Atom(5,"H",[ 1.33,60,0.00]))
    push!(atoms,Atom(6,"H",[ 1.30,60,1.00]))
    clusters    = [(1:2),(3:4),(5:6)]
end
    
#    push!(atoms,Atom(1,"H",[0,0,0]))
#    push!(atoms,Atom(2,"H",[0,0,1]))
#    push!(atoms,Atom(3,"H",[0,0,2]))
#    push!(atoms,Atom(4,"H",[0,0,3]))
#    push!(atoms,Atom(5,"H",[0,0,4]))
#    push!(atoms,Atom(6,"H",[0,0,5]))
#    push!(atoms,Atom(7,"H",[0,0,6]))
#    push!(atoms,Atom(8,"H",[0,0,7]))
    #push!(atoms,Atom(9,"H",[0,0,8]))
    #push!(atoms,Atom(10,"H",[0,0,9]))
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
   
    na = 4
    nb = 4

    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb,conv_tol=1e-10,max_cycle=100)
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

    

    max_roots = 400 

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots) 

    display.(clusters)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    

    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    ci_vector = FermiCG.ClusteredState(clusters)

    FermiCG.expand_to_full_space!(ci_vector, cluster_bases, na, nb)
    
    display(ci_vector)
    #display(cluster_bases[2][(2,2)])
    

#    if true 
#    @time H = FermiCG.build_full_H_serial(ci_vector, cluster_ops, clustered_ham)
#    dim = size(H,1)
#    
#        F = eigen(H)
#        for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
#            @printf(" %4i %18.13f\n", idx, Fi)
#        end
#    end
#    println()
    
    #@time H = FermiCG.build_full_H_serial(ci_vector, cluster_ops, clustered_ham)
    #e,v = Arpack.eigs(H, nev = 1, which=:SR)
    #@printf(" Energy: %18.12f\n",real(e[1]))
    

    #@profilehtml H = FermiCG.build_full_H(ci_vector, cluster_ops, clustered_ham)
    @time H = FermiCG.build_full_H(ci_vector, cluster_ops, clustered_ham)
    e,v = Arpack.eigs(H, nev = 1, which=:SR)
    @printf(" Energy: %18.12f\n",real(e[1]))
   
    #if false
        #F = eigen(H)
        #for (idx,Fi) in enumerate(F.values[1:min(10,length(F.values))])
        #    @printf(" %4i %18.13f\n", idx, Fi)
        #end
    #end
    

    #FermiCG.set_vector!(ci_vector, F.vectors[:,1])
    #display(ci_vector)
    println()
    
    #@test isapprox(F.values[1], -5.066833300762457, atol=1e-10)
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
#end


