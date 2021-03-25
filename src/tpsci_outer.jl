using ThreadPools

"""
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    zero_fock = TransferConfig([(0,0) for i in ci_vector.clusters])
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
                    
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                        
                        me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
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
    tpsci_ci(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                thresh_cipsi = 1e-2,
                thresh_foi   = 1e-6,
                thresh_asci  = 1e-2,
                max_iter     = 10,
                conv_thresh  = 1e-4) where {T,N,R}

Run TPSCI 
"""
function tpsci_ci(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
    thresh_cipsi = 1e-2,
    thresh_foi   = 1e-6,
    thresh_asci  = 1e-2,
    max_iter     = 10,
    conv_thresh  = 1e-4) where {T,N,R}
#={{{=#
    vec_var = deepcopy(ci_vector)
    vec_pt = deepcopy(ci_vector)
    zero!(vec_pt)
    e0 = zeros(T,R) 
    e2 = zeros(T,R) 
    e0_last = zeros(T,R)
    
    clustered_S2 = extract_S2(ci_vector.clusters)

    println(" ci_vector     :", size(ci_vector) ) 
    println(" thresh_cipsi  :", thresh_cipsi ) 
    println(" thresh_foi    :", thresh_foi   ) 
    println(" thresh_asci   :", thresh_asci  ) 
    println(" max_iter      :", max_iter     ) 
    println(" conv_thresh   :", conv_thresh  ) 
    for it in 1:max_iter

        println()
        println(" ===================================================================")
        @printf("     Selected CI Iteration: %4i epsilon: %12.8f\n", it,thresh_cipsi)
        println(" ===================================================================")

        if it > 1
            var_clip = thresh_cipsi/10
            l1 = length(vec_var)
            clip!(vec_var, thresh=var_clip)
            l2 = length(vec_var)
            @printf(" Clip values > %8.1e         %6i → %6i\n", var_clip, l1, l2)
            
            l1 = length(vec_var)
            add!(vec_var, vec_pt)
            l2 = length(vec_var)
            @printf(" Add pt vector to current space %6i → %6i\n", l1, l2)
        end

        @printf(" Build Hamiltonian matrix with dimension: %5i\n", length(vec_var))
        flush(stdout)
        #@time H = build_full_H(vec_var, cluster_ops, clustered_ham)
        @time H = build_full_H_parallel(vec_var, cluster_ops, clustered_ham)
        if length(vec_var) > 1000
            e0,v = Arpack.eigs(H, nev = R, which=:SR)
        else
            F = eigen(H)
            e0 = F.values[1:R]
            v = F.vectors[:,1:R]
        end
        set_vector!(vec_var, v)
        
        s2 = compute_expectation_value(vec_var, cluster_ops, clustered_S2)
        @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, e0[r], abs(s2[r]))
        end

        for r in 1:R
            display(vec_var, root=r)
        end

        flush(stdout)
        vec_asci = deepcopy(vec_var)
        l1 = length(vec_asci)
        clip!(vec_asci, thresh=thresh_asci)
        l2 = length(vec_asci)
        #do_asci_rediag = true 
        #if do_asci_rediag
        #    println(" Re-diagonalize in ASCI subspace:")
        #    @time Hasci = build_full_H_parallel(vec_asci, cluster_ops, clustered_ham)
        #    F = eigen(Hasci)
        #    e0_asci = F.values[1:R]
        #    v_asci = F.vectors[:,1:R]
        #    set_vector!(vec_asci, v_asci)
        #
        #    s2_asci = compute_expectation_value(vec_asci, cluster_ops, clustered_S2)
        #    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
        #    for r in 1:R
        #        @printf(" %5s %12.8f %12.8f\n",r, e0_asci[r], abs(s2_asci[r]))
        #    end
        #end
        @printf(" Length of ASCI vector %8i → %8i \n", l1, l2)
        @time e2, vec_pt = compute_pt2(vec_asci, cluster_ops, clustered_ham, thresh_foi=thresh_foi)
        flush(stdout)


        l1 = length(vec_pt)
        clip!(vec_pt, thresh=thresh_cipsi)
        l2 = length(vec_pt)
        @printf(" Length of PT1  vector %8i → %8i \n", l1, l2)
        #add!(vec_var, vec_pt)

        if maximum(abs.(e0_last .- e0)) < conv_thresh
            print_tpsci_iter(vec_var, it, e0, true)
            break
        else
            print_tpsci_iter(vec_var, it, e0, false)
            e0_last .= e0
        end
    end
    return e0, e2, vec_var, vec_pt 
end
#=}}}=#


function print_tpsci_iter(ci_vector::ClusteredState{T,N,R}, it, e0, converged) where {T,N,R}
#={{{=#
    if converged 
        @printf("*TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    else
        @printf(" TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    end
    @printf(" E(var): ")
    for i in 1:R
        @printf("%13.8f ", e0[i])
    end
#    @printf(" E(pt2): ")
#    for i in 1:R
#        @printf("%13.8f ", e2[i])
#    end
    println()
end
#=}}}=#

"""
    build_full_H(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H_parallel(ci_vector::ClusteredState, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    jobs = []

    zero_fock = TransferConfig([(0,0) for i in ci_vector.clusters])
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
                    
                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
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
"""
function compute_pt2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf", 
        thresh_foi=1e-8, 
        verbose=1) where {T,N,R}
    #={{{=#

    e2 = zeros(T,R)
    
    println()
    println(" Compute FOIS vector")
    norms = norm(ci_vector)
    @time sig = open_matvec_parallel(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    #@btime sig = open_matvec_parallel($ci_vector, $cluster_ops, $clustered_ham, nbody=$nbody, thresh=$thresh_foi)
    #@time sig = open_matvec(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    println(" Length of FOIS vector: ", length(sig))

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    project_out!(sig, ci_vector)
    

    println(" Compute diagonal")
    @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)
    

    println(" Compute <0|H0|0>")
    E0 = compute_expectation_value(ci_vector, cluster_ops, clustered_ham_0)
    #[@printf(" %4i %12.8f\n", i, E0[i]) for i in 1:length(E0)]
    
    println(" Compute <0|H|0>")
    Evar = compute_expectation_value(ci_vector, cluster_ops, clustered_ham)
    #[@printf(" %4i %12.8f\n", i, Evar[i]) for i in 1:length(Evar)]
    

    sig_v = get_vectors(sig)
    v_pt  = zeros(size(sig_v))

    println()
    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)") 
    for r in 1:R
        denom = 1.0 ./ (E0[r]/(norms[r]*norms[r]) .- Hd)  
        v_pt[:,r] .= denom .* sig_v[:,r] 
        e2[r] = sum(sig_v[:,r] .* v_pt[:,r])
   
        @printf(" %5s %12.8f %12.8f\n",r, Evar[r]/norms[r], Evar[r]/(norms[r]*norms[r]) + e2[r])
    end

    set_vector!(sig,v_pt)

    return e2, sig 
end
#=}}}=#

"""
"""
function compute_expectation_value(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}
    #={{{=#

    out = zeros(T,R)

    for (fock_bra, configs_bra) in ci_vector.data

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                continue
            end

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket

                    me = 0.0
                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                        me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    end

                    out += coeff_bra .* coeff_ket .* me

                end

            end
        end
    end

    return out 
end
#=}}}=#


"""
    open_matvec(ci_vector::ClusteredState, cluster_ops, clustered_ham; thresh=1e-9, nbody=4)

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.
"""
function open_matvec(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    #sig = ClusteredState(clusters)
    #sig = OrderedDict{FockConfig{N}, OrderedDict{NTuple{N,Int16}, MVector{T} }}()
    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket
           
            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_bra) || continue 
            all(f[2] >= 0 for f in fock_bra) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
        
            #if haskey(sig, fock_bra) == false
            #    sig[fock_bra] = OrderedDict{NTuple{N,Int16}, MVector{T}}()
            #end
            haskey(sig, fock_bra) || add_fockconfig!(sig, fock_bra)
            for term in terms

                length(term.clusters) <= nbody || continue

                for (config_ket, coeff_ket) in configs_ket
                    
                    sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)
                    #if term isa ClusteredTerm2B
                    #    @btime sig_i = contract_matvec($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, thresh=$thresh)
                    #end
                    #typeof(sig_i) == typeof(sig[fock_bra]) || println(typeof(sig_i), "\n",  typeof(sig[fock_bra]), "\n")
                    
                    merge!(+, sig[fock_bra], sig_i)
                    #sig[fock_bra] = merge(+, sig[fock_bra], sig_i)
                    
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


"""
    open_matvec_parallel(ci_vector::ClusteredState, cluster_ops, clustered_ham; thresh=1e-9, nbody=4)

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_parallel(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    #sig = ClusteredState(clusters)
    #sig = OrderedDict{FockConfig{N}, OrderedDict{NTuple{N,Int16}, MVector{T} }}()

    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_bra) || continue 
            all(f[2] >= 0 for f in fock_bra) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_bra)
                push!(jobs[fock_bra], job_input)
            else
                jobs[fock_bra] = [job_input]
            end
            
        end
    end

    jobs_vec = []
    for (fock_bra, job) in jobs
        push!(jobs_vec, (fock_bra, job))
    end

    jobs_out = Vector{ClusteredState{T,N,R}}()
    for tid in 1:Threads.nthreads()
        push!(jobs_out, ClusteredState(clusters, T=T, R=R))
    end


    #println(" Number of jobs:    ", length(jobs))
    #println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    #Threads.@threads for job in jobs_vec
   

    #for job in jobs_vec
    @qthreads for job in jobs_vec
        fock_bra = job[1]
        sigi = _open_matvec_job(job[2], fock_bra, cluster_ops, nbody, thresh, N, R, T)
        tmp = jobs_out[Threads.threadid()]
        jobs_out[Threads.threadid()][fock_bra] = sigi
    end

    for threadid in 1:Threads.nthreads()
        add!(sig, jobs_out[threadid])
    end

    BLAS.set_num_threads(Threads.nthreads())
    return sig
end
#=}}}=#

function _open_matvec_job(job, fock_bra, cluster_ops, nbody, thresh, N, R, T)
#={{{=#
    sigfock = OrderedDict{ClusterConfig{N}, MVector{R, T} }()

    for jobi in job 

        terms, fock_ket, configs_ket = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (config_ket, coeff_ket) in configs_ket

                sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)
                #if term isa ClusteredTerm4B
                #    @btime sig_i = contract_matvec($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, thresh=$thresh)
                #    error("here")
                #end
                merge!(+, sigfock, sig_i)
            end
        end
    end
    return sigfock
end
#=}}}=#


"""
    compute_diagonal(vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal(vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}
    Hd = zeros(size(vector)[1])
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
                Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
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



"""
    hosvd(ci_vector::ClusteredState, cluster_ops; hshift=1e-8, truncate=-1)

Peform HOSVD aka Tucker Decomposition of ClusteredState
"""
function hosvd(ci_vector::ClusteredState{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}
#={{{=#
   
    cluster_rotations = []
    for ci in ci_vector.clusters
        println()
        println(" --------------------------------------------------------")
        println(" Density matrix: Cluster ", ci.idx)
        println()
        println(" Compute BRDM")
        println(" Hshift = ",hshift)
        
        dims = Dict()
        for (fock, mat) in cluster_ops[1]["H"]
            dims[fock[1]] = size(mat,1)
        end
        
        rdms = build_brdm(ci_vector, ci, dims)
        norm = 0
        entropy = 0
        rotations = Dict{Tuple,Matrix{T}}() 
        for (fspace,rdm) in rdms
            fspace_norm = 0
            fspace_entropy = 0
            @printf(" Diagonalize RDM for Cluster %2i in Fock space: ",ci.idx)
            println(fspace)
            F = eigen(Symmetric(rdm))

            idx = sortperm(F.values, rev=true) 
            n = F.values[idx]
            U = F.vectors[:,idx]


            # Either truncate the unoccupied cluster states, or remix them with a hamiltonian to be unique
            if truncate < 0
                remix = []
                for ni in 1:length(n)
                    if n[ni] < 1e-8
                        push!(remix, ni)
                    end
                end
                U2 = U[:,remix]
                Hlocal = U2' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U2
                
                F = eigen(Symmetric(Hlocal))
                n2 = F.values
                U2 = U2 * F.vectors
                
                U[:,remix] .= U2[:,:]
            
            else
                keep = []
                for ni in 1:length(n) 
                    if abs(n[ni]) > truncate
                        push!(keep, ni)
                    end
                end
                @printf(" Truncated Tucker space. Starting: %5i Ending: %5i\n" ,length(n), length(keep))
                U = U[:,keep]
            end
        

           
            
            n = diag(U' * rdm * U)
            Elocal = diag(U' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U)
            
            norm += sum(n)
            fspace_norm = sum(n)
            @printf("                 %4s:    %12s    %12s\n", "","Population","Energy")
            for (ni_idx,ni) in enumerate(n)
                if abs(ni/norm) > 1e-16
                    fspace_entropy -= ni*log(ni/norm)/norm
                    entropy -=  ni*log(ni)
                    @printf("   Rotated State %4i:    %12.8f    %12.8f\n", ni_idx,ni,Elocal[ni_idx])
                end
           end
           @printf("   ----\n")
           @printf("   Entanglement entropy:  %12.8f\n" ,fspace_entropy) 
           @printf("   Norm:                  %12.8f\n" ,fspace_norm) 

           #
           # let's just be careful that our vectors remain orthogonal
           F = svd(U)
           U = F.U * F.Vt
           check_orthogonality(U) 
           rotations[fspace] = U
        end
        @printf(" Final entropy:.... %12.8f\n",entropy)
        @printf(" Final norm:....... %12.8f\n",norm)
        @printf(" --------------------------------------------------------\n")

        flush(stdout) 

        #ci.rotate_basis(rotations)
        #ci.check_basis_orthogonality()
        push!(cluster_rotations, rotations)
    end
    return cluster_rotations
end
#=}}}=#




"""
    build_brdm(ci_vector::ClusteredState, ci, dims)
    
Build block reduced density matrix for `Cluster`,  `ci`
- `ci_vector::ClusteredState` = input state
- `ci` = Cluster type for whihch we want the BRDM
- `dims` = list of dimensions for each fock sector
"""
function build_brdm(ci_vector::ClusteredState, ci, dims)
    # {{{
    rdms = OrderedDict()
    for (fspace, configs) in ci_vector.data
        curr_dim = dims[fspace[ci.idx]]
        rdm = zeros(curr_dim,curr_dim)
        for (configi,coeffi) in configs
            for cj in 1:curr_dim

                configj = [configi...]
                configj[ci.idx] = cj
                configj = ClusterConfig(configj)

                if haskey(configs, configj)
                    rdm[configi[ci.idx],cj] += sum(coeffi.*configs[configj])
                end
            end
        end


        if haskey(rdms, fspace[ci.idx]) 
            rdms[fspace[ci.idx]] += rdm 
        else
            rdms[fspace[ci.idx]] = rdm 
        end

    end
    return rdms
end
# }}}


