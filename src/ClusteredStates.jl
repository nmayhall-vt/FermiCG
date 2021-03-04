using LinearAlgebra

    

"""
    Base.display(s::ClusteredState; thresh=1e-3)

Pretty print
"""
function Base.display(s::ClusteredState; thresh=1e-3)
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- Fockspaces in state ------: Dim = %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "Fock space(α,β)...") 
    @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
    for (fock,configs) in s.data
        prob = 0
        for (config, coeff) in configs 
            prob += coeff*coeff 
        end
        if prob > thresh
            @printf(" %-20.3f%-20i", prob,length(s.data[fock]))
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()
        end
    end
    print(" --------------------------------------------------\n")
end

"""
    print_configs(s::ClusterState; thresh=1e-3)

Pretty print
"""
function print_configs(s::ClusteredState; thresh=1e-3)
    #display(keys(s.data))
    idx = 1
    for (fock,configs) in s.data
        #display(s.clusters)
        #display(s.data)
        length(s.clusters) == length(fock) || throw(Exception)
        length(s.data[fock]) > 0 || continue
        @printf(" Dim %4i fock_space: ",length(s.data[fock]))
        [@printf(" %-2i(%i:%i) ",fii,fi[1],fi[2]) for (fii,fi) in enumerate(fock)] 
        println()
        for (config, value) in s.data[fock]
            @printf(" %5i",idx)
            for c in config
                @printf("%3i",c)
            end
            @printf(":%12.8f\n",value)
            idx += 1
        end
    end
end

"""
    norm(s::ClusteredState)
"""
function LinearAlgebra.norm(s::ClusteredState)
    norm = 0
    for (fock,configs) in s.data
        for (config,coeff) in configs
            norm += coeff*coeff
        end
    end
    return sqrt(norm)
end

"""
    normalize!(s::AbstractState)
"""
function normalize!(s::AbstractState)
    scale!(s,1/sqrt(dot(s,s))) 
end

"""
    scale!(s::ClusteredState,c)
"""
function scale!(s::ClusteredState,c)
    for (fock,configs) in s.data
        for (config,coeff) in configs
            s[fock][config] = coeff*c
        end
    end
end
    
"""
    prune_empty_fock_spaces!(s::ClusteredState)
        
remove fock_spaces that don't have any configurations 
"""
function prune_empty_fock_spaces!(s::ClusteredState)
    keylist = keys(s.data)
    for fock in keylist
        if length(s[fock]) == 0
            delete!(s.data, fock)
        end
    end
end

"""
    zero!(s::ClusteredState)

set all elements to zero
"""
function zero!(s::ClusteredState)
    for (fock,configs) in s.data
        for (config,coeffs) in configs                
            s.data[fock][config] = 0
        end
    end
end

"""
    clip!(s::ClusteredState; thresh=1e-5)
"""
function clip!(s::ClusteredState; thresh=1e-5)
#={{{=#
    for (fock,configs) in s.data
        for (config,coeff) in configs      
            if abs(coeff) < thresh
                delete!(s.data[fock], config)
            end
        end
    end
    prune_empty_fock_spaces!(s)
end
#=}}}=#


"""
    expand_each_fock_space!(s::ClusteredState, bases)

For each fock space sector defined, add all possible basis states
- `basis::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::ClusteredState, bases::Vector{ClusterBasis})
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
            s.data[fblock][config] = 0
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
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H_serial(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
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
                    ket_idx <= bra_idx || continue


                    for term in clustered_ham[fock_trans]
                        me = FermiCG.contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                        H[bra_idx, ket_idx] += me 
                    end

                    H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                end
            end
        end
    end
    return H
end
#=}}}=#


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

    
    #Threads.@threads for job in shuffle(jobs)
    for job in reverse(jobs)
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
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H2(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    jobs = []

    zero_fock = FermiCG.TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
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
                    ket_idx <= bra_idx || continue

                    push!(jobs, (bra_idx, ket_idx, fock_trans, fock_bra, config_bra, fock_ket, config_ket))

                end
            end
        end
    end

    function do_job(job)
        #return FermiCG.contract_matrix_element(job[3], cluster_ops, job[4:7]...)
        me = 0.0
        for term in clustered_ham[job[3]]
            me += FermiCG.contract_matrix_element(term, cluster_ops, job[4:7]...)
        end
        return me
    end

    
    #Threads.@threads for job in jobs
    for job in jobs
        me = do_job(job)
        H[job[1], job[2]] += me
        H[job[2], job[1]] = H[job[1], job[2]]
    end
    return H
end
#=}}}=#

