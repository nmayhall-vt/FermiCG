#"""
#    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#
#Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
#"""
#function build_full_H_serial(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
##={{{=#
#    dim = length(ci_vector)
#    H = zeros(dim, dim)
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
#
#                    for term in clustered_ham[fock_trans]
#                    
#                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
#                        
#                        me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
#                        H[bra_idx, ket_idx] += me 
#                    end
#
#                    H[ket_idx, bra_idx] = H[bra_idx, ket_idx]
#
#                end
#            end
#        end
#    end
#    return H
#end
##=}}}=#


"""
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    jobs = []

    zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    N = length(ci_vector.clusters)
    

    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            #push!(jobs, (bra_idx, fock_bra, config_bra) )
            #push!(jobs, (bra_idx, fock_bra, config_bra, H[bra_idx,:]) )
            push!(jobs, (bra_idx, fock_bra, config_bra, zeros(dim)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        Hrow = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                ket_idx += 1
                ket_idx <= job[1] || continue

                for term in clustered_ham[fock_trans]
                       
                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    
                    me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime FermiCG.contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end

            end

        end
    end

    # because @threads divides evenly the loop, let's distribute thework more fairly
    #mid = length(jobs) ÷ 2
    #r = collect(1:length(jobs))
    #perm = [r[1:mid] reverse(r[mid+1:end])]'[:]
    #jobs = jobs[perm]
    
    #for job in jobs
    Threads.@threads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        H[job[1],:] .= job[4]
    end

    for i in 1:dim
        @simd for j in i+1:dim
            @inbounds H[i,j] = H[j,i]
        end
    end


    return H
end
#=}}}=#


"""
    open_matvec(ci_vector::ClusteredState, cluster_ops, clustered_ham; thresh=1e-9, nbody=4)

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.
"""
function open_matvec(ci_vector::ClusteredState, cluster_ops, clustered_ham; thresh=1e-9, nbody=4)
#={{{=#
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    #sig = ClusteredState(clusters)
    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket
           
            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_bra) || continue 
            all(f[2] >= 0 for f in fock_bra) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
        
            haskey(sig, fock_bra) || add_fockconfig!(sig, fock_bra)
            for term in terms

                length(term.clusters) <= nbody || continue

                for (config_ket, coeff_ket) in configs_ket
                    sig_i = FermiCG.contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)
                    #if term isa ClusteredTerm2B
                    #    @btime sig_i = FermiCG.contract_matvec($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, thresh=$thresh)
                    #end
                    sig[fock_bra] = merge(+, sig[fock_bra], sig_i)
                    #for (config,coeff) in sig_i
                    #    #display(coeff[1])
                    #    #display(sig[fock_bra][config][1])
                    #    sig[fock_bra][config][1] += coeff[1]
                    #    #sig[fock_bra][config] = sig[fock_bra][config] + coeff
                    #end
                end
            end
        end
    end

    return sig
end
#=}}}=#



function compute_diagonal(vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}
    Hd = zeros(size(vector)[1])
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]

                Hd[idx] = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
            end
        end
    end
    return Hd
end

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




"""
    project_out!(v::ClusteredState, w::ClusteredState)

Project w out of v 
|v'> = |v> - |w><w|v>
"""
function project_out!(v::ClusteredState, w::ClusteredState)
    for (fock,configs) in v.data 
        for (config, coeff) in configs
            if haskey(w, fock)
                if haskey(w[fock], config)
                    delete!(v.data[fock], config)
                end
            end
        end
    end
end
