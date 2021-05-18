using LinearAlgebra

    



#
#
#
#"""
#    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#
#Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
#"""
#function build_full_H2(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
##={{{=#
#    dim = length(ci_vector)
#    H = zeros(dim, dim)
#
#    jobs = []
#
#    zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
#    bra_idx = 0
#    for (fock_bra, configs_bra) in ci_vector.data
#        for (config_bra, coeff_bra) in configs_bra
#            bra_idx += 1
#            ket_idx = 0
#            for (fock_ket, configs_ket) in ci_vector.data
#                fock_trans = fock_bra - fock_ket
#
#                # check if transition is connected by H
#                if haskey(clustered_ham, fock_trans) == false
#                    ket_idx += length(configs_ket)
#                    continue
#                end
#
#                for (config_ket, coeff_ket) in configs_ket
#                    ket_idx += 1
#                    ket_idx <= bra_idx || continue
#
#                    push!(jobs, (bra_idx, ket_idx, fock_trans, fock_bra, config_bra, fock_ket, config_ket))
#
#                end
#            end
#        end
#    end
#
#    function do_job(job)
#        #return FermiCG.contract_matrix_element(job[3], cluster_ops, job[4:7]...)
#        me = 0.0
#        for term in clustered_ham[job[3]]
#            me += FermiCG.contract_matrix_element(term, cluster_ops, job[4:7]...)
#        end
#        return me
#    end
#
#    
#    #Threads.@threads for job in jobs
#    for job in jobs
#        me = do_job(job)
#        H[job[1], job[2]] += me
#        H[job[2], job[1]] = H[job[1], job[2]]
#    end
#    return H
#end
##=}}}=#
#
