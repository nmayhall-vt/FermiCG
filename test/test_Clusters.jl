using FermiCG
using ClusterMeanField
using InCoreIntegrals
using Printf
using Test

#function tmp()
@testset "Clusters" begin
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
    

    mf = pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = pyscf_fci(ints,1,1)
    @printf(" FCI Energy: %12.8f\n", e_fci)
    
    C = mf.mo_coeff
    Cl = localize(mf.mo_coeff,"lowdin",mf)
    pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)


    clusters    = [(1:2),(3:4),(5:6)]
    #clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1)]

    max_roots = 20

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1)

    #
    #   Just test that the sum of all the basis vector matrices is reproduced
    println("")
    tst1 = 0
    for ci in clusters
        display(cluster_bases[ci.idx])
        for (sector,vecs) in cluster_bases[ci.idx].basis
            tst1 += sum(abs.(vecs.vectors))
        end
    end
    println(tst1)
    @test isapprox(tst1, 66.05063700792823, atol=1e-10)
   
    # now try with restrictions on fock space, and dimensions
    cluster_bases = Vector{ClusterBasis}()
    max_roots=2
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, 
                                                       max_roots=2, init_fspace=init_fspace, delta_elec=1)

    println("")
    tst1 = 0
    for ci in clusters
        display(cluster_bases[ci.idx])
        for (sector,vecs) in cluster_bases[ci.idx].basis
            tst1 += sum(abs.(vecs.vectors))
        end
    end
    println(tst1)
    @test isapprox(tst1, 43.466561233596934, atol=1e-10)
   
end

