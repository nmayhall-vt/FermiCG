using FermiCG
using Printf
using Test

#@testset "TDMs" begin
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
    init_fspace = [(2,1),(1,1)]

    max_roots = 20

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    #cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, 
    #                                                   max_roots=2, init_fspace=init_fspace, delta_elec=2)
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=100) 

    #for cb in cluster_bases
    #    display(cb)
    #end

    cluster_ops = Vector{FermiCG.ClusterOps}()
    for ci in clusters
        push!(cluster_ops, FermiCG.ClusterOps(ci)) 
    end


    for ci in clusters
        cb = cluster_bases[ci.idx]

        cluster_ops[ci.idx]["A"], cluster_ops[ci.idx]["a"] = FermiCG.tdm_A(cb) 
        cluster_ops[ci.idx]["B"], cluster_ops[ci.idx]["b"] = FermiCG.tdm_B(cb)
    end

    ref1 = [     0.760602    0.194328
                -0.356922   -0.610415
                 0.540039   -0.637967
                -0.0494895   0.42735]
    test =cluster_ops[2]["A"][((1,1),(0,1))][1,:,:] 
    display(ref1)
    display(test)
    @test isapprox(ref1, test, atol=1e-5)
    
    #for (ftrans,op) in cluster_ops[1]["B"]
    #    println(ftrans, "  :  ", size(op))
    #    display(op[1,:,:])
    #end


    ref2 = [ 0.692476    0.168256   -0.0394252    0.00396003
             0.633938    0.0369774   0.0770702   -0.0153966
            -0.0742221   0.646856   -0.636296     0.103912
            -0.308786   -0.0175901   0.00515862  -0.0633258
            -0.0776733   0.444367   -0.0201446   -0.621285
             0.0130287   0.0582555   0.348725    -0.653307]

    display(ref2)
    display(cluster_ops[1]["A"][((2, 4), (1, 4))][1,:,:]) 
    test =cluster_ops[1]["B"][((4, 2), (4, 1))][1,:,:] 
    @test isapprox(ref2, test, atol=1e-5)
#end

