using FermiCG
using Printf
using Random
using Test
using LinearAlgebra
using JLD2
  
Random.seed!(2)
@load "_testdata_cmf_h12_64bit.jld2"

clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots = 3

ref_fock = FermiCG.FockConfig(init_fspace)
ψ0 = FermiCG.single_excitonic_basis(clusters, ref_fock, R=nroots, Nk=4)

# Initialize vectors:
e0, ψ0 = FermiCG.tps_ci_direct(ψ0, cluster_ops, clustered_ham, conv_thresh=1e-10);
e2 = FermiCG.compute_pt2_energy(ψ0, cluster_ops, clustered_ham, thresh_foi=1e-10)
FermiCG.set_vector!(ψ0, rand(size(ψ0)...))

# compute residual

# slow 
σ0 = FermiCG.open_matvec_thread(ψ0, cluster_ops, clustered_ham, thresh=1e-3)
display(size(σ0))

H = FermiCG.build_full_H(σ0, cluster_ops, clustered_ham);
FermiCG.set_vector!(σ0, H*FermiCG.get_vector(σ0));
Hss1 = FermiCG.overlap(σ0, ψ0);
display(Hss1) 

# new
σ0 = FermiCG.open_matvec_thread(ψ0, cluster_ops, clustered_ham, thresh=1e-3)
display(size(σ0))
Hss2 = FermiCG.matrix_element(σ0, ψ0, cluster_ops, clustered_ham);
display(Hss2) 
display(Hss2 - Hss1) 
@test isapprox(norm(Hss2-Hss1), 0.0, atol=1e-10)