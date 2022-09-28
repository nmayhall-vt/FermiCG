using QCBase
using ClusterMeanField
using FermiCG
using Printf
using Test
using JLD2 

@testset "H12_CMF" begin

    molecule = "
    H   0.0     0.0     0.0
    H   0.0     0.0     1.0
    H   0.0     1.0     2.0
    H   0.0     1.0     3.0
    H   0.0     2.0     4.0
    H   0.0     2.0     5.0
    H   0.0     3.0     6.0
    H   0.0     3.0     7.0
    H   0.0     4.0     8.0
    H   0.0     4.0     9.0
    H   0.0     5.0     10.0
    H   0.0     5.0     11.0
    "

    atoms = []
    for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
        l = split(line)
        push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
    end


    clusters    = [(1:2), (3:4), (5:8), (9:10), (11:12)]
    init_fspace = [(1, 1),(1, 1),(2, 2),(1, 1),(1, 1)]

    (na,nb) = sum(init_fspace)


    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)

    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=4, do_rdm1=false, do_rdm2=false);
    e_fci = -18.33022092
    e_fci_states = [-18.33022092, -18.05457645, -18.02913048, -17.99661028]

    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)


    #
    # do CMF
    d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1, 
                                       max_iter_oo=60, verbose=0, gconv=1e-10, 
                                       method="bfgs")
    ints = FermiCG.orbital_rotation(ints,U)

    @test isapprox(e_cmf, -6.5218473576915414, atol=1e-9)
    @save "_testdata_cmf_h12.jld2" ints d1 e_cmf clusters init_fspace
end


@testset "H12_CMF_basis" begin
    
    @load "_testdata_cmf_h12.jld2" ints d1 e_cmf clusters init_fspace
    
    max_roots = 20

    #
    # form Cluster data
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, 
                                                       max_roots=max_roots, 
                                                       init_fspace=init_fspace, 
                                                       rdm1a=d1.a, rdm1b=d1.b)

    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    check = 0.0
    for ci_ops in cluster_ops
        for (opstr, ops) in ci_ops 
            for (ftrans, op) in ops 
                check += sum(abs.(op))
            end
        end
    end
    println(check)
    #@test isapprox(check, 159955.9925735096, atol=1e-6)
    @save "_testdata_cmf_h12.jld2" ints d1 e_cmf clusters init_fspace cluster_bases  clustered_ham cluster_ops
end

