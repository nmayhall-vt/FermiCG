e_ci, v = FermiCG.ci_solve(ci_vector, cluster_ops, clustered_ham);
@save "data_ci.jld2" v e_ci

#v = BSTstate(v,R=2)

v1 = BSTstate(v,R=1)
v2 = BSTstate(v,R=1)
FermiCG.set_vector!(v2, FermiCG.get_vector(v)[:,8])
σ1 = FermiCG.build_compressed_1st_order_state(v1, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-7,
                                    compress_twice=false)

σ2 = FermiCG.build_compressed_1st_order_state(v2, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-7,
                                    compress_twice=false)

e_ci, v = FermiCG.ci_solve(v1, cluster_ops, clustered_ham);
e_ci, v = FermiCG.ci_solve(v2, cluster_ops, clustered_ham);
display(FermiCG.nonorth_overlap(σ1,v1))
display(FermiCG.nonorth_overlap(σ2,v2))

#σ = FermiCG.compress(σ, thresh=1e-2)
#
#FermiCG.zero!(σ)
#FermiCG.nonorth_add!(v, σ)
#
#e_ci, v = FermiCG.ci_solve(v, cluster_ops, clustered_ham);
#
