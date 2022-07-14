using FermiCG
using Printf
using Test
using JLD2 

if true 
@testset "He4" begin

    # start with a square, then add some noise to break symmetries
    molecule = "
    He  -1.5    0.0     0.0
    He   1.5    0.0     0.5
    He   0.0   -1.5     0.0
    He   0.0    1.5     0.0
    "

    atoms = []
    for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
        l = split(line)
        push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
    end


    clusters    = [(1:5), (6:10), (11:15), (16:20)]
    init_fspace = [(1, 1), (1, 1), (1, 1), (1, 1)]
    (na,nb) = sum(init_fspace)


    basis = "cc-pvdz"
    mol     = Molecule(0, 1, atoms, basis)

    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    display(mf.energy_tot())
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=4, do_rdm1=false, do_rdm2=false);
    #e_fci_states = [-18.33022092, -18.05457645, -18.02913048, -17.99661028]
    

    #@test isapprox(mf.energy_tot(), -11.416159557959963, atol=1e-9)

    # localize orbitals
    C = mf.mo_coeff
    FermiCG.pyscf_write_molden(mol, C, filename="he4_rhf.molden")
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol, Cl, filename="he4_loc.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)


    #
    # do CMF
    rdm1 = zeros(size(ints.h1))
    e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, rdm1, 
                                       max_iter_oo=60, verbose=0, gconv=1e-10, 
                                       method="bfgs")
    ints = FermiCG.orbital_rotation(ints,U)
    C = Cl*U

    FermiCG.pyscf_write_molden(mol, C, filename="he4_cmf.molden")

    @test isapprox(e_cmf, -11.545601384796, atol=1e-9)
    @save "_testdata_cmf_he4.jld2" ints Da Db e_cmf clusters init_fspace C
end
end

@testset "He4_basis" begin
    
    @load "_testdata_cmf_he4.jld2" 
    
    max_roots = 20

    #
    # form Cluster data
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, 
                                                       max_roots=max_roots, 
                                                       delta_elec=1,
                                                       init_fspace=init_fspace, 
                                                       rdm1a=Da, rdm1b=Db)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);

    check = 0.0
    for ci_ops in cluster_ops
        for (opstr, ops) in ci_ops 
            for (ftrans, op) in ops 
                check += sum(abs.(op))
            end
        end
    end
    println(check)
    @test isapprox(check, 51116.898762307974, atol=1e-8)
    @save "_testdata_cmf_he4.jld2" ints C Da Db e_cmf clusters init_fspace cluster_bases  clustered_ham cluster_ops
end

