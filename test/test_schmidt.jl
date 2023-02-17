using FermiCG
using RDM
using LinearAlgebra
using Printf
using Arpack 
using Test
using OrderedCollections

@testset "EST" begin
    @load "_testdata_cmf_h8.jld2" 

    na = 4
    nb = 4

    cb_cmf = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, max_roots=5,rdm1a=d1.a, rdm1b=d1.b,init_fspace=init_fspace) 
    cb_est = FermiCG.compute_cluster_est_basis(ints, clusters, d1.a, d1.b, thresh_schmidt=1e-4, init_fspace=init_fspace)

    
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cb_est, ints);

    ci_vector = FermiCG.TPSCIstate(clusters)
    FermiCG.expand_to_full_space!(ci_vector, cb_est, na, nb)
    display(ci_vector,thresh=-1)



    ovlp = OrderedDict()
    for ci in clusters
    	for (key,value) in cb_est[ci.idx]
            #println(key)
            #display(cb_est[ci.idx][key])    
            ovlp[ci.idx,key] = cb_cmf[ci.idx].basis[key]'*cb_est[ci.idx].basis[key]
        end
    end
    	
    

    H = FermiCG.build_full_H(ci_vector, cluster_ops, clustered_ham)
    display(size(H))
    #display(H)
    println()

    display(ci_vector,root=1)
    e,v = Arpack.eigs(H, nev = 8, which=:SR)
    for ei in e
        @printf(" Energy: %18.12f\n",real(ei))
    end

    ref_e = [-10.18725871
    -10.15805362
    -10.14789239
    -10.12277607
    -10.11982911
    -10.11553898
    -10.03259372
    -10.02109892]

    @test isapprox(e, ref_e, atol=1e-6)

    S_1_42 =[0.616641;
	  0.766495;
	  0.027335;
	  0.176888;
	  0.013518]
    @test isapprox(abs.(ovlp[1,(4,2)]), abs.(S_1_42), atol=1e-4)

    S_1_41 = [ 0.935067 -0.307226;
          0.208181  0.478093;
         -0.272371 -0.777974;
         -0.090131  0.267947]
    @test isapprox(abs.(ovlp[1,(4,1)]), abs.(S_1_41), atol=1e-4)


    S_1_33 = [-0.32085  -0.        0.851868;
         -0.        0.616641  0.      ;
          0.       -0.766495 -0.      ;
         -0.711982 -0.       -0.110697;
         -0.137815  0.       -0.329198]
    @test isapprox(abs.(ovlp[1,(3,3)]), abs.(S_1_33), atol=1e-4)

    S_1_23 = [ 0.674829  0.229042 -0.228784  0.656271 -0.061039  0.       -0.024558 -0. ;
         -0.635893  0.420242  0.073237  0.548461  0.306119 -0.       -0.023685 -0.      ;
         -0.       -0.        0.        0.       -0.       -0.935067 -0.       -0.307226;
          0.036266 -0.099898  0.164285  0.035816 -0.0381   -0.       -0.387746 -0.      ;
          0.226091  0.792939  0.421423 -0.369864 -0.06511  -0.       -0.002342  0.      ]
    @test isapprox(abs.(ovlp[1,(2,3)]) , abs.(S_1_23), atol=1e-4)
    S_2_42 = [-0.032297;
          0.981557;
         -0.012785;
         -0.187085;
         -0.004896]
    @test isapprox(abs.(ovlp[2,(4,2)]) , abs.(S_2_42), atol=1e-4)
    S_2_41 = [ 0.944586  0.281996;
          0.011955 -0.030939;
         -0.318901  0.908279;
          0.07692   0.30749 ;]
    @test isapprox(abs.(ovlp[2,(4,1)]) , abs.(S_2_41), atol=1e-4)
    S_2_24 = [ 0.032297;
         -0.981557;
         -0.012785;
          0.187085;
         -0.004896]
    @test isapprox(abs.(ovlp[2,(2,4)]) , abs.(S_2_24), atol=1e-4)


end



    
