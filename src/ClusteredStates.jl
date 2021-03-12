using LinearAlgebra

    



"""
    expand_each_fock_space!(s::ClusteredState, bases)

For each fock space sector defined, add all possible basis states
- `basis::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::ClusteredState{T,N,R}, bases::Vector{ClusterBasis}) where {T,N,R}
    # {{{
    println("\n Make each Fock-Block the full space")
    # create full space for each fock block defined
    for (fblock,configs) in s.data
        #println(fblock)
        dims::Vector{UnitRange{Int16}} = []
        #display(fblock)
        for c in s.clusters
            # get number of vectors for current fock space
            dim = size(bases[c.idx][fblock[c.idx]], 2)
            push!(dims, 1:dim)
        end
        for newconfig in product(dims...)
            #display(newconfig)
            #println(typeof(newconfig))
            #
            # this is not ideal - need to find a way to directly create key
            config = ClusterConfig(collect(newconfig))
            s.data[fblock][config] = zeros(SVector{R,T}) 
            #s.data[fblock][[i for i in newconfig]] = 0
        end
    end
end
# }}}

"""
    expand_to_full_space(s::ClusteredState, bases)

Define all possible fock space sectors and add all possible basis states
- `basis::Vector{ClusterBasis}` 
- `na`: Number of alpha electrons total
- `nb`: Number of alpha electrons total
"""
function expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis}, na, nb)
    # {{{
    println("\n Expand to full space")
    ns = []

    for c in s.clusters
        nsi = []
        for (fspace,basis) in bases[c.idx]
            push!(nsi,fspace)
        end
        push!(ns,nsi)
    end
    for newfock in product(ns...)
        nacurr = 0
        nbcurr = 0
        for c in newfock
            nacurr += c[1]
            nbcurr += c[2]
        end
        if (nacurr == na) && (nbcurr == nb)
            config = FockConfig(collect(newfock))
            add_fockconfig!(s,config) 
        end
    end
    expand_each_fock_space!(s,bases)

    return
end
# }}}


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
