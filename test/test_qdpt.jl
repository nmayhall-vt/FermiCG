using QCBase
using RDM
using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using JLD2

# load data
@load "_testdata_cmf_cr2_19.jld2"

M=20
init_fspace =  FockConfig(init_fspace)

cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3,3,3], init_fspace, max_roots=M, verbose=1);

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters);
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=4


ci_vector = FermiCG.TPSCIstate(clusters, init_fspace, R=nroots)

ci_vector = FermiCG.add_spin_focksectors(ci_vector)

eci, v = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);


e2a = FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham)
e_qdpt=FermiCG.compute_qdpt_energy(v, cluster_ops, clustered_ham, nbody=4, H0="Hcmf",  thresh_foi=1e-8,  verbose=1, threaded=true)

println(" PT2 - ROCMF-PT2")
display(e2a)
println(" PT2 - ROCMF-QDPT")
display(e_qdpt)
@test isapprox(e_qdpt, [-108.13989647019216, -108.13975457069576, -108.1394893503325, -108.1390882840833], atol=1e-12)

