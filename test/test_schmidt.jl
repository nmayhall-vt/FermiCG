using FermiCG
using ClusterMeanField
using LinearAlgebra
using Printf
using Arpack 
using Test
using OrderedCollections

@testset "EST" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0, 0, 0.1]))
    push!(atoms,Atom(2,"H",[0, 1,-1]))
    push!(atoms,Atom(3,"H",[0, 1, 1]))
    push!(atoms,Atom(4,"H",[0, 2, 0]))
    push!(atoms,Atom(5,"H",[0, 4, 0]))
    push!(atoms,Atom(6,"H",[0, 5,-1]))
    push!(atoms,Atom(7,"H",[0, 5, 1]))
    push!(atoms,Atom(8,"H",[0, 6, 0]))
    #basis = "6-31g"
    basis = "sto-3g"

    na = 4
    nb = 4

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

    FermiCG.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
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
    init_fspace = [(2,2),(2,2)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf*.5
    rdm1b = rdm_mf*.5

    rdm1a = rdm1
    rdm1b = rdm1

    display(rdm1a)
    display(rdm1b)
    

    d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = cmf_oo(ints, clusters, init_fspace, d1,
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    ints = FermiCG.orbital_rotation(ints,U)


    #for ci in clusters
    #    ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 
    #	print(ints_i.h1)
    #end


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
            println(key)
            #display(cb_est[ci.idx][key])    
            ovlp[ci.idx,key] = cb_cmf[ci.idx].basis[key]'*cb_est[ci.idx].basis[key]
        end
    end
    	
    

    H = FermiCG.build_full_H(ci_vector, cluster_ops, clustered_ham)
    display(size(H))
    display(H)
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



    
