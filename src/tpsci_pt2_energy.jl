"""
    compute_pt2_energy(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=true,
        verbose=1) where {T,N,R}
"""
function compute_pt2_energy(ci_vector_in::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0::String ="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-9, 
        prescreen=true,
        verbose=1) where {T,N,R}

    println()
    println(" |..................................do batched PT2......................................")
    println(" thresh_foi    :", thresh_foi   ) 
    println(" prescreen     :", prescreen   ) 
    println(" H0            :", H0   ) 
    println(" nbody         :", nbody   ) 

    e2 = zeros(T,R)
   
    ci_vector = deepcopy(ci_vector_in)
    clusters = ci_vector.clusters
    norms = norm(ci_vector);
    @printf(" Norms of input states:\n")
    [@printf(" %12.8f\n",i) for i in norms]
    orthonormalize!(ci_vector)
    
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
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


    # 
    # define batches (FockConfigs present in resolvant)
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_x) || continue 
            all(f[2] >= 0 for f in fock_x) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
            # 
            # Check to make sure we don't create states that we have already discarded
            found = true
            for c in ci_vector.clusters
                if haskey(cluster_ops[c.idx]["H"], (fock_x[c.idx], fock_x[c.idx])) == false
                    found = false
                    continue
                end
            end
            found == true || continue
 
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
            
        end
    end

    #
    # prepare scratch arrays to help cut down on allocation in the threads
    jobs_vec = []
    for (fock_x, job) in jobs
        push!(jobs_vec, (fock_x, job))
    end

    scr_f = Vector{Vector{Vector{T}} }()
    scr_i = Vector{Vector{Vector{Int16}} }()
    scr_m = Vector{Vector{MVector{N,Int16}} }()
    nscr = 20 

    scr1 = Vector{Vector{T}}()
    scr2 = Vector{Vector{T}}()
    scr3 = Vector{Vector{T}}()
    scr4 = Vector{Vector{T}}()
    tmp1 = Vector{MVector{N,Int16}}()
    tmp2 = Vector{MVector{N,Int16}}()

    e2_thread = Vector{Vector{T}}()
    for tid in 1:Threads.nthreads()
        push!(e2_thread, zeros(T, R))
        push!(scr1, zeros(T, 1000))
        push!(scr2, zeros(T, 1000))
        push!(scr3, zeros(T, 1000))
        push!(scr4, zeros(T, 1000))
        push!(tmp1, zeros(Int16,N))
        push!(tmp2, zeros(Int16,N))

        tmp = Vector{Vector{T}}() 
        [push!(tmp, zeros(T,10000)) for i in 1:nscr]
        push!(scr_f, tmp)

        tmp = Vector{Vector{Int16}}() 
        [push!(tmp, zeros(Int16,10000)) for i in 1:nscr]
        push!(scr_i, tmp)

        tmp = Vector{MVector{N,Int16}}() 
        [push!(tmp, zeros(Int16,N)) for i in 1:nscr]
        push!(scr_m, tmp)
    end



    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)


    tmp = Int(round(length(jobs_vec)/100))
    verbose < 1 || println("   |----------------------------------------------------------------------------------------------------|")
    verbose < 1 || println("   |0%                                                                                              100%|")
    verbose < 1 || print("   |")
    #@profilehtml @Threads.threads for job in jobs_vec
    t = @elapsed begin
        #@qthreads for job in jobs_vec
        #@time for job in jobs_vec
        
        @Threads.threads for (jobi,job) in collect(enumerate(jobs_vec))
        #for (jobi,job) in collect(enumerate(jobs_vec))
            fock_bra = job[1]
            tid = Threads.threadid()
            e2_thread[tid] .+= _pt2_job(job[2], fock_bra, cluster_ops, nbody, thresh_foi,  
                                        scr_f[tid], scr_i[tid], scr_m[tid],  prescreen, verbose, 
                                        ci_vector, H0, E0)
            if verbose > 0
                if  jobi%tmp == 0
                    print("-")
                    flush(stdout)
                end
            end
        end
    end
    verbose < 1 || println("|")
    flush(stdout)
   
    @printf(" Time spent computing E2 %12.1f (s)\n",t)
    flush(stdout)
    e2 = sum(e2_thread) 

    #BLAS.set_num_threads(Threads.nthreads())

    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, Evar[r], Evar[r] + e2[r])
    end
    println(" ......................................................................................|")

    return e2
end


function _pt2_job(job, fock_x, cluster_ops, nbody, thresh, 
                  scr_f, scr_i, scr_m, prescreen, verbose,
                  ci_vector::TPSCIstate{T,N,R}, opstring::String, E0) where {T,N,R}
    #={{{=#

    sig = TPSCIstate(ci_vector.clusters, T=T, R=R)
    add_fockconfig!(sig, fock_x)

    for jobi in job 

        terms, fock_ket, configs_ket = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (config_ket, coeff_ket) in configs_ket

                #term isa ClusteredTerm2B || continue
                #if (length(sig[fock_x]) > 0) && (term isa ClusteredTerm4B)
                if (term isa ClusteredTerm4B)
                    #println("1: ", length(sig[fock_x]))
                    #@btime contract_matvec_thread($term, $cluster_ops, $fock_x, $fock_ket, $config_ket, $coeff_ket, 
                    #                       $sig[$fock_x], 
                    #                       $scr_f, $scr_i, $scr_m, 
                    #                       thresh=$thresh, prescreen=$prescreen)
                    #contract_matvec_thread(term, cluster_ops, fock_x, fock_ket, config_ket, coeff_ket, 
                    #                   sig[fock_x], 
                    #                   scr_f, scr_i, scr_m, 
                    #                   thresh=thresh, prescreen=prescreen)
                    #println("2: ", length(sig[fock_x]))
                    #@profilehtml contract_matvec_thread(term, cluster_ops, fock_x, fock_ket, config_ket, coeff_ket, 
                    #                   sig[fock_x], 
                    #                   scr_f, scr_i, scr_m, 
                    #                   thresh=thresh, prescreen=prescreen)
                    #error("we good?")
                end

                contract_matvec_thread(term, cluster_ops, fock_x, fock_ket, config_ket, coeff_ket, 
                                       sig[fock_x], 
                                       scr_f, scr_i, scr_m, 
                                       thresh=thresh, prescreen=prescreen)
                #if term isa ClusteredTerm3B
                    #@code_warntype contract_matvec_thread(term, cluster_ops, fock_x, fock_ket, config_ket, 
                    #                                coeff_ket,  sig[fock_x],scr1, scr2, thresh=thresh)
                    #@btime contract_matvec_thread($term, $cluster_ops, $fock_x, $fock_ket, $config_ket, 
                    #                        $coeff_ket, $sig[$fock_x], $scr_f, $scr_i, $scr_m, thresh=$thresh)
                #end
            end
        end
    end

    #@btime project_out!($sig, $ci_vector)
    #@time project_out!(sig, ci_vector)
    
    project_out!(sig, ci_vector)
    #verbose > 0 || println(" Fock(X): ", fock_x)
    #verbose > 1 || println(" Length of FOIS vector: ", length(sig))
    #verbose > 1 || println(" Compute diagonal")
    
    nx = length(sig)

    Hd = scr_f[9]
    resize!(Hd, nx)
    Hd = reshape2(Hd, (nx, 1))
    
    #Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)
    fill!(Hd,0.0)
    compute_diagonal!(Hd, sig, cluster_ops, opstring)
    # compute_diagonal!(Hd, sig, cluster_ops, clustered_ham_0)
    #@btime compute_diagonal!($Hd, $sig, $cluster_ops, $clustered_ham_0)
    
    sig_v = scr_f[10]
    resize!(sig_v, nx*R)
    sig_v = reshape2(sig_v, size(sig))
    fill!(sig_v,0.0)
    
    get_vector!(sig_v, sig)
    
    #sig_v = get_vector(sig)

   
    e2 = zeros(T,R)

    
    _sum_pt2(sig_v, e2, Hd, E0, R)

    return e2 
end
#=}}}=#

function _sum_pt2(sig_v, e2, Hd, E0, R)
    # put into a function to let compiler specialize
    nx = length(Hd)
    sig_vx = 0.0
    Hdx = 0.0
    @inbounds for x in 1:nx
        Hdx = Hd[x]
        @simd for r in 1:R
            sig_vx = sig_v[x,r]
            e2[r] += sig_vx*sig_vx / (E0[r] - Hdx) 
            #verbose > 0 || @printf(" %5s %12.8f\n",r, e2[r])
        end
    end
end


