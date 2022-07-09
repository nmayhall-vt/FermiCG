using FermiCG
using Printf
using Test

@testset "TDMs" begin
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

    cluster_ops = Vector{FermiCG.ClusterOps{Float64}}()
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
        cluster_ops[ci.idx]["Ab"], cluster_ops[ci.idx]["Ba"] = FermiCG.tdm_Ab(cb) 
        # remove BA and ba account for these terms 
        @time cluster_ops[ci.idx]["AB"], cluster_ops[ci.idx]["ba"], cluster_ops[ci.idx]["BA"], cluster_ops[ci.idx]["ab"] = FermiCG.tdm_AB(cb)
        @time cluster_ops[ci.idx]["AAa"], cluster_ops[ci.idx]["Aaa"] = FermiCG.tdm_AAa(cb,"alpha")
        @time cluster_ops[ci.idx]["BBb"], cluster_ops[ci.idx]["Bbb"] = FermiCG.tdm_AAa(cb,"beta")
        #@time cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"] = FermiCG.tdm_ABa(cb,"alpha")
        #@time cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"] = FermiCG.tdm_ABa(cb,"beta")

    end

    ref1 = [     0.760602    0.194328
                -0.356922   -0.610415
                 0.540039   -0.637967
                -0.0494895   0.42735]
    test =cluster_ops[2]["A"][((1,1),(0,1))][1,:,:] 
    @test isapprox(abs.(ref1), abs.(test), atol=1e-5)
    
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
    @test isapprox(abs.(ref2), abs.(test), atol=1e-5)
   
    ref = [  0.997916  0.064526
           -0.064526  0.997916]
    test =cluster_ops[2]["AA"][((2, 1), (0, 1))][1,2,:,:] 
    @test isapprox(abs.(ref), abs.(test), atol=1e-5)
  
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
    @test isapprox(abs.(ref), abs.(test), atol=1e-5)

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
    @test isapprox(abs.(ref), abs.(test), atol=1e-5)
   
    ref = [ 0.370772    -0.0457363    0.0432994   0.264432    0.0339364   0.00585283
            0.510164    -1.20899e-5   0.17079     0.420218   -0.0816309   0.0788461
           -0.319318     0.237434     0.312764   -0.0502967  -0.433148    0.217789
           -0.356189     0.0186202   -0.0863461  -0.276715    0.0190341  -0.0341083
           -0.181094     0.0902997    0.0989763  -0.0682649  -0.155185    0.0736075
            0.00357835  -0.224128    -0.39496    -0.19786     0.456546   -0.251624]
    test =cluster_ops[1]["Bb"][((0, 2), (0, 2))][1,2,:,:] 
    @test isapprox(abs.(ref), abs.(test), atol=1e-5)
    
    #for (ftrans,op) in cluster_ops[1]["Ab"]
    #    println(ftrans, "  :  ", size(op))
    #    display(op[1,2,:,:])
    #end
    ref = [-0.369922      0.0414125   0.0253631    0.00504684
           -0.41114      -0.0696597   0.12167     -0.00578434
           -0.0297536    -0.352606    0.298661    -0.0198366
           -8.22928e-5    0.280202   -0.218175     0.0411312
            0.0182344     0.0406758  -0.0132163    0.0410473
           -0.100527     -0.224248    0.072862    -0.226296
           -0.488761      0.025401   -0.00741915  -0.102493
            0.414459     -0.141834   -0.0148663   -0.119597
           -0.0286128     0.0514988  -0.0944393   -0.0892154
            0.065136      0.289568   -0.0509547    0.346402
            0.115824      0.25837    -0.0839488    0.260729
            0.16495       0.0135669  -0.0310352    0.0110329
            0.173482      0.136309   -0.339572    -0.318982
           -0.359717      0.324396   -0.232706    -0.0128002
           -0.167805     -0.0148749   0.0542493    0.0245395
            0.0704252     0.157099   -0.0510441    0.158533
           -0.0074902     0.0978906  -0.280457    -0.323457
            0.00575834   -0.0406361   0.0847375    0.0828477
           -0.0123439    -0.179338    0.0875409   -0.115216
           -0.145663      0.0755987   0.0108951    0.0881159
           -0.0112101    -0.113188   -0.131103    -0.380087
           -0.0261508     0.0420964  -0.261946    -0.377486
            0.000736687  -0.0341842   0.0390686    0.0156487
            0.00543429    0.0115098   0.0303084    0.0676828]
    test =cluster_ops[1]["Ab"][((2, 3), (1, 4))][1,2,:,:] 
    @test isapprox(abs.(ref), abs.(test), atol=1e-5)
    
    display(cluster_ops[1]["AA"][((3,1),(1,1))][1,2,1,1])
    display(cluster_ops[1]["Aa"][((3,1),(3,1))][1,2,1,1])
    display(cluster_ops[1]["Ab"][((3,1),(2,2))][1,2,1,1])
    display(cluster_ops[1]["Ba"][((1,3),(2,2))][1,3,1,1])
    display(cluster_ops[1]["AB"][((3,3),(2,2))][1,3,1,2])
    display(cluster_ops[1]["AAa"][((3,2),(2,2))][1,2,3,1,2])
    display(cluster_ops[1]["Bbb"][((2,1),(2,2))][1,2,3,1,2])
    println(length(cluster_ops[1].data))
end

