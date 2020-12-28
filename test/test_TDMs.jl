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

        cluster_ops[ci.idx]["A"], cluster_ops[ci.idx]["a"] = FermiCG.tdm_A(cb,"alpha") 
        cluster_ops[ci.idx]["B"], cluster_ops[ci.idx]["b"] = FermiCG.tdm_A(cb,"beta")
        cluster_ops[ci.idx]["AA"], cluster_ops[ci.idx]["aa"] = FermiCG.tdm_AA(cb,"alpha") 
        cluster_ops[ci.idx]["BB"], cluster_ops[ci.idx]["bb"] = FermiCG.tdm_AA(cb,"beta") 
        cluster_ops[ci.idx]["Aa"] = FermiCG.tdm_Aa(cb,"alpha") 
        cluster_ops[ci.idx]["Bb"] = FermiCG.tdm_Aa(cb,"beta") 
    end

    ref1 = [     0.760602    0.194328
                -0.356922   -0.610415
                 0.540039   -0.637967
                -0.0494895   0.42735]
    test =cluster_ops[2]["A"][((1,1),(0,1))][1,:,:] 
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

    #display(ref2)
    #display(cluster_ops[1]["A"][((2, 4), (1, 4))][1,:,:]) 
    test =cluster_ops[1]["B"][((4, 2), (4, 1))][1,:,:] 
    @test isapprox(ref2, test, atol=1e-5)
   
    ref = [  0.997916  0.064526
           -0.064526  0.997916]
    test =cluster_ops[2]["AA"][((2, 1), (0, 1))][1,2,:,:] 
    @test isapprox(ref, test, atol=1e-5)
  
    ref = [ -0.210308     0.134555     -0.0105751   -0.0130626
           -0.146011     0.459975      0.0957622   -0.0409076
           -0.121296     0.404645      0.201137    -0.0233472
           -0.398129     0.129795     -0.0889375   -0.0373247
           0.160396    -0.276295      0.317716     0.172581
           -0.116331    -0.373659     -0.164785     0.0129931
           0.0401054   -0.17791      -0.248367    -0.0897069
           -0.12651      0.301717     -0.0496348   -0.248452
           -0.149806     0.0853372     0.433886     0.1877
           0.481946     0.0435557     0.128998     0.111449
           0.0926449    0.195847      0.134642     0.084457
           0.507474     0.109186      0.144895    -0.1903
           -0.0439681   -0.0443377    -0.338921    -0.364227
           0.307398     0.331592     -0.268884     0.0109789
           -0.00297574   0.0595918     0.112011     0.0919267
           -0.0601778    0.0736758    -0.022481     0.0812163
           0.247002     0.109472     -0.00769421  -0.383945
           0.126592     0.178113     -0.442779     0.222488
           0.0362439    0.0233627    -0.0142876   -0.190231
           -0.0222505    0.000279442   0.0915151    0.0481816
           0.0659491    0.159962     -0.310095     0.627363
           0.0115114    7.89468e-5   -0.0468136   -0.0231196
           -0.0121033   -0.0291859     0.0530118   -0.164273
           -0.00148844  -0.00618908    0.00372897  -0.0336697]
    test =cluster_ops[1]["aa"][((2, 1), (4, 1))][1,2,:,:] 
    @test isapprox(ref, test, atol=1e-5)


    #for (ftrans,op) in cluster_ops[1]["aa"]
    #    println(ftrans, "  :  ", size(op))
    #    display(op[1,2,:,:])
    #end
    #for (ftrans,op) in cluster_ops[2]["AA"]
    #    println(ftrans, "  :  ", size(op))
    #    display(op[1,2,:,:])
    #end
   
    #for (ftrans,op) in cluster_ops[1]["Bb"]
    #    println(ftrans, "  :  ", size(op))
    #    display(op[1,2,:,:])
    #end
    ref = [ 0.46354     0.394923     -0.416441   -0.268368
           -0.199213    1.17757e-16   0.33466     0.590183
            0.356483    0.557353     -0.087595    0.503239
           -0.0456762  -0.18274      -0.0908969  -0.375945]
    test =cluster_ops[2]["Aa"][((1, 1), (1, 1))][1,2,:,:] 
    @test isapprox(ref, test, atol=1e-5)
   
    ref = [ 0.370772    -0.0457363    0.0432994   0.264432    0.0339364   0.00585283
            0.510164    -1.20899e-5   0.17079     0.420218   -0.0816309   0.0788461
           -0.319318     0.237434     0.312764   -0.0502967  -0.433148    0.217789
           -0.356189     0.0186202   -0.0863461  -0.276715    0.0190341  -0.0341083
           -0.181094     0.0902997    0.0989763  -0.0682649  -0.155185    0.0736075
            0.00357835  -0.224128    -0.39496    -0.19786     0.456546   -0.251624]
    test =cluster_ops[1]["Bb"][((0, 2), (0, 2))][1,2,:,:] 
    @test isapprox(ref, test, atol=1e-5)
#end

