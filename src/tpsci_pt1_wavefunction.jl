"""
    compute_pt1_wavefunction(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        matvec=3) where {T,N,R}
"""
function compute_pt1_wavefunction(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        matvec=3) where {T,N,R}
    #={{{=#

    println()
    println(" |............................do PT2................................")
    println(" thresh_foi    :", thresh_foi   ) 
    println(" prescreen     :", prescreen   ) 
    println(" H0            :", H0   ) 
    println(" nbody         :", nbody   ) 

    e2 = zeros(T,R)
    
    norms = norm(ci_vector);
    println(" Norms of input states")
    [@printf(" %12.8f\n",i) for i in norms]
    println(" Compute FOIS vector")

    if matvec == 1
        #@time sig = open_matvec(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
        sig = open_matvec_serial2(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    elseif matvec == 2
        sig = open_matvec_thread(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    elseif matvec == 3
        sig = open_matvec_thread2(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    else
        error("wrong matvec")
    end
    #@time sig = open_matvec_parallel(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    #@btime sig = open_matvec_parallel($ci_vector, $cluster_ops, $clustered_ham, nbody=$nbody, thresh=$thresh_foi)
    println(" Length of FOIS vector: ", length(sig))

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    project_out!(sig, ci_vector)
    println(" Length of FOIS vector: ", length(sig))
    
    @printf(" %-50s", "Compute diagonal")
    @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)
    
    if E0 == nothing
        @printf(" %-50s", "Compute <0|H0|0>:")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)
        #E0 = diag(E0)
        flush(stdout)
    end

    @printf(" %-50s", "Compute <0|H|0>:")
    @time Evar = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)
    #Evar = diag(Evar)
    flush(stdout)
    

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
    println(" ..................................................................|")

    return e2, sig 
end
#=}}}=#


"""
    open_matvec(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.
"""
function open_matvec(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec")
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    #sig = TPSCIstate(clusters)
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
    open_matvec_thread(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_thread(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec_thread")
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    #sig = TPSCIstate(clusters)
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

    jobs_out = Vector{TPSCIstate{T,N,R}}()
    for tid in 1:Threads.nthreads()
        push!(jobs_out, TPSCIstate(clusters, T=T, R=R))
    end


    #println(" Number of jobs:    ", length(jobs))
    #println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    #Threads.@threads for job in jobs_vec
   

    #for job in jobs_vec
    #@qthreads for job in jobs_vec
    Threads.@threads for job in jobs_vec
        fock_bra = job[1]
        sigi = _open_matvec_job(job[2], fock_bra, cluster_ops, nbody, thresh, N, R, T)
        tmp = jobs_out[Threads.threadid()]
        jobs_out[Threads.threadid()][fock_bra] = sigi
    end

    for threadid in 1:Threads.nthreads()
        #display(size(jobs_out[threadid]))
        add!(sig, jobs_out[threadid])
    end

    #BLAS.set_num_threads(Threads.nthreads())
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
                #if term isa ClusteredTerm2B
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
    open_matvec_parallel(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_parallel(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
   
    println(" Copy data to each worker")
    @sync for pid in procs()
        @spawnat pid eval(:(ci_vector = deepcopy($ci_vector)))
        @spawnat pid eval(:(sig_job = TPSCIstate($clusters, R=$R)))
        @spawnat pid eval(:(cluster_ops = $cluster_ops))
        @spawnat pid eval(:(clusters = $clusters))
        @spawnat pid eval(:(thresh = $thresh))
    end
    flush(stdout)

    println(" Collect jobs")
    @time for (fock_ket, configs_ket) in ci_vector.data
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

    jobs_out = Dict{Int, TPSCIstate{T,N,R}}()
    for pid in procs()
        jobs_out[pid] = TPSCIstate(clusters, T=T, R=R)
    end


    println(" Number of jobs:    ", length(jobs))
   

    #@sync @distributed for job in jobs_vec
   
    futures = []

    println(" Compute all jobs")
    @time @sync begin
        for job in jobs_vec
            fock_bra = job[1]
            future_sigi = @spawnat :any _open_matvec_job_parallel(job[2], fock_bra, nbody, thresh, N, R, T)
            #jobs_out[myid()][fock_bra] = sigi
            push!(futures, future_sigi)
        end
    end

    println(" Combine results")
    flush(stdout)
    @time for pid in procs()
        add!(sig, @fetchfrom pid sig_job)
    end

    #BLAS.set_num_threads(Threads.nthreads())
    return sig
end
#=}}}=#

function _open_matvec_job_parallel(job, fock_bra, nbody, thresh, N, R, T)
#={{{=#
    #sigfock = OrderedDict{ClusterConfig{N}, MVector{R, T} }()

    #sig = TPSCIstate(clusters,R=R)
    add_fockconfig!(sig_job, fock_bra)

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
                merge!(+, sig_job[fock_bra], sig_i)
            end
        end
    end
    return 
end
#=}}}=#

"""
    open_matvec_parallel2(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
"""
function open_matvec_parallel2(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    sig = deepcopy(ci_vector)
    zero!(sig)
    clusters = ci_vector.clusters

    println(" create empty sig vector on each worker")
    @sync for pid in procs()
        @spawnat pid eval(:(ci_vector = deepcopy($ci_vector)))
        @spawnat pid eval(:(sig_job = TPSCIstate($clusters, R=$R)))
        @spawnat pid eval(:(cluster_ops = $cluster_ops))
        @spawnat pid eval(:(clusters = $clusters))
        @spawnat pid eval(:(thresh = $thresh))
    end
    println("done")
    flush(stdout)


    #jobs = Vector{Tuple{TransferConfig{N}, ClusteredTerm}}()
    #println(" Number of jobs: ", length(jobs))

   
    println(" Compute all jobs")
    futures = []
    @time @sync for (ftrans, terms) in clustered_ham
        for term in terms
            length(term.clusters) <= nbody || continue

            future = @spawnat :any _do_job(ftrans, term)
            #future = do_job(ftrans, term)
            push!(futures, future)
        end
    end

    println(" combine results")
    flush(stdout)
    @time @sync for pid in procs()
        add!(sig, @fetchfrom pid sig_job)
    end

#    n = length(futures)
#    @elapsed while n > 0 # print out results
#        add!(sig, take!(futures))
#        n = n - 1
#    end

    return sig
end
#=}}}=#

function _do_job(ftrans, term)
    for (fock_ket, configs_ket) in ci_vector.data
        fock_bra = ftrans + fock_ket

        #
        # check to make sure this fock config doesn't have negative or too many electrons in any cluster
        all(f[1] >= 0 for f in fock_bra) || continue 
        all(f[2] >= 0 for f in fock_bra) || continue 
        all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 
        all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_bra)) || continue 

        haskey(sig_job, fock_bra) || add_fockconfig!(sig_job, fock_bra)


        for (config_ket, coeff_ket) in configs_ket

            sig_i = contract_matvec(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, thresh=thresh)

            merge!(+, sig_job[fock_bra], sig_i)
        end
    end
    return 
end


