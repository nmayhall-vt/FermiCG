"""
    open_matvec_thread(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_thread(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham; 
                             thresh=1e-9, 
                             prescreen=true,
                             nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec_thread")
    #sig = deepcopy(ci_vector)
    sig = TPSCIstate(ci_vector.clusters, T=T, R=R)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    #sig = TPSCIstate(clusters)
    #sig = OrderedDict{FockConfig{N}, OrderedDict{NTuple{N,Int16}, MVector{T} }}()

    @printf(" %-50s", "Setup threaded jobs: ")
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

    jobs_out = Vector{TPSCIstate{T,N,R}}()
    for tid in 1:Threads.nthreads()
        push!(jobs_out, TPSCIstate(clusters, T=T, R=R))
        push!(scr1, zeros(T, 1000))
        push!(scr2, zeros(T, 1000))
        push!(scr3, zeros(T, 1000))
        push!(scr4, zeros(T, 1000))
        push!(tmp1, zeros(Int16,N))
        push!(tmp2, zeros(Int16,N))

        tmp = Vector{Vector{T}}() 
        [push!(tmp, zeros(T, 10000)) for i in 1:nscr]
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
    #Threads.@threads for job in jobs_vec
  
    flush(stdout)

    #@time for job in jobs_vec
    #@qthreads for job in jobs_vec
    @printf(" %-50s", "Compute matrix-vector: ")
    @time @Threads.threads for job in jobs_vec
        fock_bra = job[1]
        tid = Threads.threadid()
        _open_matvec_thread_job(job[2], fock_bra, cluster_ops, nbody, thresh, 
                                 jobs_out[tid], scr_f[tid], scr_i[tid], scr_m[tid], prescreen)
    end
    flush(stdout)

    @printf(" %-50s", "Now collect thread results : ")
    flush(stdout)
    @time for threadid in 1:Threads.nthreads()
        for (fock, configs) in jobs_out[threadid].data
            haskey(sig, fock) == false || error(" why me")
            sig[fock] = configs
        end
    end

    #BLAS.set_num_threads(Threads.nthreads())
    return sig
end
#=}}}=#

function open_matvec_serial(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                             thresh=1e-9, 
                             prescreen=true,
                             nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec_serial2")
    #sig = deepcopy(ci_vector)
    sig = TPSCIstate(ci_vector.clusters, T=T, R=R)
    zero!(sig)
    clusters = ci_vector.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()

    @printf(" %-50s", "Setup threaded jobs: ")
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

    jobs_out = Vector{TPSCIstate{T,N,R}}()
    for tid in 1:1
        push!(jobs_out, TPSCIstate(clusters, T=T, R=R))
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
    flush(stdout)
    @printf(" %-50s", "Compute matrix-vector: ")
    @time for job in jobs_vec
        fock_bra = job[1]
        tid = 1
        _open_matvec_thread_job(job[2], fock_bra, cluster_ops, nbody, thresh, 
                                 jobs_out[tid], scr_f[tid], scr_i[tid], scr_m[tid],prescreen)
    end
    flush(stdout)

    flush(stdout)
    @printf(" %-50s", "Now collect thread results: ")
    @time for threadid in 1:1
        #add!(sig, jobs_out[threadid])
        for (fock, configs) in jobs_out[threadid].data
            haskey(sig, fock) == false || error(" why me")
            sig[fock] = configs
        end
    end

    return sig
end
#=}}}=#

function _open_matvec_thread_job(job, fock_bra, cluster_ops, nbody, thresh, sig, scr_f, scr_i, scr_m, prescreen)
#={{{=#

    haskey(sig, fock_bra) || add_fockconfig!(sig, fock_bra)
    #sizehint!(sig[fock_bra],1000)

    for jobi in job 

        terms, fock_ket, configs_ket = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (config_ket, coeff_ket) in configs_ket

                #term isa ClusteredTerm2B || continue

                contract_matvec_thread(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, sig[fock_bra], 
                                       scr_f, scr_i, scr_m, thresh=thresh, prescreen=prescreen)
                if term isa ClusteredTerm3B
                    #@code_warntype contract_matvec_thread(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, 
                    #                              sig[fock_bra],scr1, scr2, thresh=thresh)
                    #@btime contract_matvec_thread($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, 
                    #                              $sig[$fock_bra], $scr_f, $scr_i, $scr_m, thresh=$thresh)
                end
            end
        end
    end
    return 
end
#=}}}=#

reshape2(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

"""
    contract_matvec_thread(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9) where {T,R,N}
"""
function contract_matvec_thread(   term::ClusteredTerm1B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9, prescreen=true) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        
    
    #
    # <:|p'|J> h(pq) <:|q|L>
    #
    # <:|p'|J> h(pq) <:|q|L>
    g1a = cluster_ops[c1.idx][term.ops[1]]

    # the following type declarations are needed to prevent allocations. This ultimately is due to the fact 
    # that the ClusterOps type isn't formed in an ideal concrete manner. We just currently use Array. 
    # This should be cleaned up eventually, preferably with a distinction between contracted, and uncontracted
    # operators.
    g1::Array{T,2} = g1a[(fock_bra[c1.idx],fock_ket[c1.idx])]
    
    @views gamma1 = g1[:,conf_ket[c1.idx]]

    scr1 = scr_f[1]
    resize!(scr1, size(gamma1,1))

    scr1 .= gamma1 .* state_sign
    newI = 1:size(scr1,1)
    #new_coeffs = scr1
    #newI = 1:size(new_coeffs,1)

    _collect_significant_thread!(sig, conf_ket, scr1, coef_ket, c1.idx,  newI,  thresh, scr_m[1])
    #_collect_significant_thread!(sig, conf_ket, new_coeffs, coef_ket, c1.idx,  newI,  thresh)
            

    return 
end
#=}}}=#

"""
    contract_matvec_thread(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9) where {T,R,N}
"""
function contract_matvec_thread(   term::ClusteredTerm2B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9, prescreen=true) where {T,R,N}
#={{{=#


    c1 = term.clusters[1]
    c2 = term.clusters[2]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
  

    #
    # <:|p'|J> h(pq) <:|q|L>
    g1a = cluster_ops[c1.idx][term.ops[1]]
    g2a = cluster_ops[c2.idx][term.ops[2]]
    

    # the following type declarations are needed to prevent allocations. This ultimately is due to the fact 
    # that the ClusterOps type isn't formed in an ideal concrete manner. We just currently use Array. 
    # This should be cleaned up eventually, preferably with a distinction between contracted, and uncontracted
    # operators.
           
    haskey(g1a, (fock_bra[c1.idx],fock_ket[c1.idx])) || return
    haskey(g2a, (fock_bra[c2.idx],fock_ket[c2.idx])) || return
    
    g1::Array{T,3} = g1a[(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2::Array{T,3} = g2a[(fock_bra[c2.idx],fock_ket[c2.idx])]
    
    @views gamma1 = g1[:,:,conf_ket[c1.idx]]
    @views gamma2 = g2[:,:,conf_ket[c2.idx]]
    

    scr1 = scr_f[1]
    scr2 = scr_f[2]

    resize!(scr1, size(term.ints,2) * size(gamma1,2))
    resize!(scr2, size(gamma1,2) * size(gamma2,2))
    
    scr1 = reshape2(scr1, (size(term.ints,2), size(gamma1,2)))
    scr2 = reshape2(scr2, (size(gamma1,2), size(gamma2,2)))

  
    mul!(scr1, term.ints', gamma1)
    mul!(scr2, scr1', gamma2)
    new_coeffs = scr2
  
    newI = 1:size(new_coeffs,1)
    newJ = 1:size(new_coeffs,2)

    _collect_significant_thread!(sig, conf_ket, new_coeffs, coef_ket, c1.idx, c2.idx, newI, newJ, thresh, scr_m[1], state_sign)
    #@btime _collect_significant_thread!($sig, $conf_ket, $new_coeffs, $coef_ket, $c1.idx, $c2.idx, $newI, $newJ, $thresh, $scr3)

    return 
end
#=}}}=#

"""
    contract_matvec_thread(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9, prescreen=true) where {T,R,N}

This version should only use M^2N^2 storage, and n^4 scaling n={MN}
"""
function contract_matvec_thread(   term::ClusteredTerm3B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9, prescreen=true) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
    


    #
    # h(pqr) <I|p'|_> <J|q|_> <K|r|_> 
    #
    # X(p,J,K) = h(pqr) <J|q|_> <K|r|_>
    #
    #
    haskey(cluster_ops[c1.idx][term.ops[1]],  (fock_bra[c1.idx],fock_ket[c1.idx])) || return
    haskey(cluster_ops[c2.idx][term.ops[2]],  (fock_bra[c2.idx],fock_ket[c2.idx])) || return
    haskey(cluster_ops[c3.idx][term.ops[3]],  (fock_bra[c3.idx],fock_ket[c3.idx])) || return

    g1::Array{T,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2::Array{T,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    g3::Array{T,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    @views gamma1 = g1[:,:,conf_ket[c1.idx]]
    @views gamma2 = g2[:,:,conf_ket[c2.idx]]
    @views gamma3 = g3[:,:,conf_ket[c3.idx]]
   
    scr1 = scr_f[1]
    scr2 = scr_f[2]
    scr3 = scr_f[3]
    tmp1 = scr_m[1]
    tmp2 = scr_m[2]
    
    newI = UnitRange{Int16}(1,size(g1,2))
    newJ = UnitRange{Int16}(1,size(g2,2))
    newK = UnitRange{Int16}(1,size(g3,2))
    
    if prescreen
        up_bound = upper_bound_thread(term.ints, gamma1, gamma2, gamma3,
                              scr1, scr2, scr3,
                              c=maximum(abs.(coef_ket)))
        if up_bound < thresh
            return 
        end
        #@btime newI, newJ, newK = upper_bound2($term.ints, $gamma1, $gamma2, $gamma3, $thresh, c=maximum(abs.($coef_ket)))
        newI, newJ, newK = upper_bound2_thread(term.ints, gamma1, gamma2, gamma3, 
                                                     scr_f, scr_i, thresh, c=maximum(abs, coef_ket))
        length(newI) > 0 || return
        length(newJ) > 0 || return
        length(newK) > 0 || return
        

        if true
            gamma1 = scr_f[4]
            gamma2 = scr_f[5]
            gamma3 = scr_f[6]

            resize!(gamma1, size(g1,1)*length(newI))
            resize!(gamma2, size(g2,1)*length(newJ))
            resize!(gamma3, size(g3,1)*length(newK))

            gamma1 = reshape2(gamma1, (size(g1,1), length(newI)))
            gamma2 = reshape2(gamma2, (size(g2,1), length(newJ)))
            gamma3 = reshape2(gamma3, (size(g3,1), length(newK)))

            _fill1!(gamma1, newI, g1, conf_ket[c1.idx])
            _fill1!(gamma2, newJ, g2, conf_ket[c2.idx])
            
            #gamma1 .= g1[:,newI,conf_ket[c1.idx]]
            #gamma2 .= g2[:,newJ,conf_ket[c2.idx]]
            #gamma3 .= g3[:,newK,conf_ket[c3.idx]]
        else
            @views gamma1 = g1[:,newI,conf_ket[c1.idx]]
            @views gamma2 = g2[:,newJ,conf_ket[c2.idx]]
            @views gamma3 = g3[:,newK,conf_ket[c3.idx]]
        end
    end

    # 
    # for K in G3
    #   scr1(p,q) = h(p,q,r) * G3(r;K)          N^3M^1
    #   scr2(p,J) = scr1(p,q) * G2(q,J)
    #   scr3(I,J) = G1(p,I)' * scr2(p,J)
    #
    #   collect(scr3(I,J))


    np = size(term.ints,1)
    nq = size(term.ints,2)
    nr = size(term.ints,3)
    nI = length(newI) 
    nJ = length(newJ) 
    nK = length(newK) 
    
    resize!(scr1, np*nq)
    resize!(scr2, np*nJ)
    resize!(scr3, nI*nJ)
    
    scr2 = reshape2(scr2, (np,nJ))
    scr3 = reshape2(scr3, (nI,nJ))
  
    
    v = reshape2(term.ints, (np*nq,nr))


    #a = @allocated begin
    #end; if a > 0 println(a); error("here we are"); end

    tmp1 .= conf_ket.config

    for K::Int16 in newK 
   
        @views gamma3 = g3[:,K,conf_ket[c3.idx]]
    
        tmp1[c3.idx] = K
                
        tmp_conf = ClusterConfig(SVector(tmp1))
   
        mul!(scr1, v, gamma3)
        mul!(scr2, reshape2(scr1, (np,nq)), gamma2)
        #@allocated mul!(scr2, reshape2(scr1, (np,nq)), gamma2) == 0 || error(" we allocated")

            
        if prescreen
            #println("3b")
            #@btime upper_bound_thread($gamma1, $scr2, c=maximum(abs.($coef_ket))) 
            #println(size(gamma1))
            #println(size(scr2))
            #println(typeof(gamma1))
            #println(typeof(scr2))
            upper_bound_thread(gamma1, scr2, c=maximum(abs.(coef_ket))) > thresh || continue
        end
        
        mul!(scr3, gamma1', scr2) 
        #@btime mul!($scr3, $gamma1', $scr2) 
        #a = @allocated begin
        #    mul!(scr3, gamma1', scr2)  
        #end
        #a == 0 || error(" we allocated: ", a)
            
        
        #@btime _collect_significant_thread!($sig, $tmp_conf, $scr3, $coef_ket, $c1.idx, $c2.idx, $newI, $newJ, $thresh, $tmp2, $state_sign)
        _collect_significant_thread!(sig, tmp_conf, scr3, coef_ket, c1.idx, c2.idx, newI, newJ, thresh, tmp2, state_sign)
    end

    return 
end
#=}}}=#

"""
    contract_matvec_thread(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9, prescreen=true) where {T,R,N}

This version should only use M^2N^2 storage, and n^5 scaling n={MN}
"""
function contract_matvec_thread(   term::ClusteredTerm4B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, 
                                    scr_f::Vector{Vector{T}},  
                                    scr_i::Vector{Vector{Int16}},  
                                    scr_m::Vector{MVector{N,Int16}};  
                                    thresh=1e-9, prescreen=true) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
    


    #
    # h(pqr) <I|p'|_> <J|q|_> <K|r|_> 
    #
    # X(p,J,K) = h(pqr) <J|q|_> <K|r|_>
    #
    #
    haskey(cluster_ops[c1.idx][term.ops[1]],  (fock_bra[c1.idx],fock_ket[c1.idx])) || return
    haskey(cluster_ops[c2.idx][term.ops[2]],  (fock_bra[c2.idx],fock_ket[c2.idx])) || return
    haskey(cluster_ops[c3.idx][term.ops[3]],  (fock_bra[c3.idx],fock_ket[c3.idx])) || return
    haskey(cluster_ops[c4.idx][term.ops[4]],  (fock_bra[c4.idx],fock_ket[c4.idx])) || return
    
    g1::Array{T,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2::Array{T,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    g3::Array{T,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    g4::Array{T,3} = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    @views gamma1 = g1[:,:,conf_ket[c1.idx]]
    @views gamma2 = g2[:,:,conf_ket[c2.idx]]
    @views gamma3 = g3[:,:,conf_ket[c3.idx]]
    @views gamma4 = g4[:,:,conf_ket[c4.idx]]
    
    np = size(term.ints,1)
    nq = size(term.ints,2)
    nr = size(term.ints,3)
    ns = size(term.ints,4)
    
    scr1 = scr_f[1]
    scr2 = scr_f[2]
    scr3 = scr_f[3]
    scr4 = scr_f[4]
    tmp1 = scr_m[1]
    tmp2 = scr_m[2]
    
    
    newI = UnitRange{Int16}(1,size(g1,2))
    newJ = UnitRange{Int16}(1,size(g2,2))
    newK = UnitRange{Int16}(1,size(g3,2))
    newL = UnitRange{Int16}(1,size(g4,2))
    if prescreen
        up_bound = upper_bound_thread(term.ints, gamma1, gamma2, gamma3, gamma4,
                              scr1, scr2, scr3, scr4,
                              c=maximum(abs.(coef_ket)))
        if up_bound < thresh
            return 
        end
        #println("a")
        #
        # screen phase 2: ignore indices for each cluster which will produce discarded terms

        #@btime newI, newJ, newK, newL = upper_bound2_thread($term.ints, $gamma1, $gamma2, $gamma3, $gamma4, 
        #                                             $scr_f, $scr_i, $thresh, c=maximum(abs, $coef_ket))

        newI, newJ, newK, newL = upper_bound2_thread(term.ints, gamma1, gamma2, gamma3, gamma4, 
                                                     scr_f, scr_i, thresh, c=maximum(abs, coef_ket))
        length(newI) > 0 || return
        length(newJ) > 0 || return
        length(newK) > 0 || return
        length(newL) > 0 || return
        #println("b")


        #
        # While we could simply use a view, this creates a non-contiguous array and BLAS then does allocations
        
        if true 
            g1t = typeof(gamma1)
            g2t = typeof(gamma1)
            g3t = typeof(gamma1)
            g4t = typeof(gamma1)
            gamma1 = scr_f[5]
            gamma2 = scr_f[6]
            gamma3 = scr_f[7]
            gamma4 = scr_f[8]

            
            #g1t == typeof(gamma1) || println(" here: ", g1t, " vs ", typeof(gamma1))
            #g2t == typeof(gamma2) || println(" here: ", g2t, " vs ", typeof(gamma2))
            #g3t == typeof(gamma3) || println(" here: ", g3t, " vs ", typeof(gamma3))
            #g4t == typeof(gamma4) || println(" here: ", g4t, " vs ", typeof(gamma4))
            
            resize!(gamma1, size(g1,1)*length(newI))
            resize!(gamma2, size(g2,1)*length(newJ))
            resize!(gamma3, size(g3,1)*length(newK))
            resize!(gamma4, size(g4,1)*length(newL))

            gamma1 = reshape2(gamma1, (size(g1,1), length(newI)))
            gamma2 = reshape2(gamma2, (size(g2,1), length(newJ)))
            gamma3 = reshape2(gamma3, (size(g3,1), length(newK)))
            gamma4 = reshape2(gamma4, (size(g4,1), length(newL)))



            _fill1!(gamma1, newI, g1, conf_ket[c1.idx])
            _fill1!(gamma2, newJ, g2, conf_ket[c2.idx])
            #_fill1!(gamma3, newK, g3, conf_ket[c3.idx])
            #_fill1!(gamma4, newL, g4, conf_ket[c4.idx])

            #gamma1 .= g1[:,newI,conf_ket[c1.idx]]
            #gamma2 .= g2[:,newJ,conf_ket[c2.idx]]
            #gamma3 .= g3[:,newK,conf_ket[c3.idx]]
            #gamma4 .= g4[:,newL,conf_ket[c4.idx]]
        else
            @views gamma1 = g1[:,newI,conf_ket[c1.idx]]
            @views gamma2 = g2[:,newJ,conf_ket[c2.idx]]
            @views gamma3 = g3[:,newK,conf_ket[c3.idx]]
            @views gamma4 = g4[:,newL,conf_ket[c4.idx]]
        end
    end

    #
    #   for L in G4
    #       scr1(p,q,r) = h(p,q,r,s) * G4(s;L)          N^4M^1
    #   
    #       for K in G3
    #           scr2(p,q) = scr1(p,q,r) * G3(r;K)       N^3M^2
    #           scr3(p,J) = scr2(p,q) * G2(q,J)         N^2M^3
    #           scr4(I,J) = G1'(p,I) * scr3(p,J)        N^1M^4
    #       
    #           collect(scr4(I,J))


    nI = length(newI) 
    nJ = length(newJ) 
    nK = length(newK) 
    nL = length(newL) 
    
    resize!(scr1, np*nq*nr)
    resize!(scr2, np*nq)
    resize!(scr3, np*nJ)
    resize!(scr4, nI*nJ)
    
    scr3 = reshape2(scr3, (np,nJ))
    scr4 = reshape2(scr4, (nI,nJ))
    
    v = reshape2(term.ints, (np*nq*nr,ns))
    #@btime v = reshape2($term.ints, ($np*$nq*$nr,$ns))



    tmp1 .= conf_ket.config

    a = 0.0
    #@btime begin 
    #    for L::Int16 in $newL 
    #        $a += 1.0
    #    end
    #end

    for L::Int16 in newL 
            
        @views gamma4 = g4[:,L,conf_ket[c4.idx]]
        #@btime @views gamma4 = $g4[:,$L,$conf_ket[$c4.idx]]
        
        tmp1[c4.idx] = L
        #@btime $tmp1[$c4.idx] = $L
            
        mul!(scr1, v, gamma4)
        #@btime mul!($scr1, $v, $gamma4)

        for K::Int16 in newK 

            @views gamma3 = g3[:,K,conf_ket[c3.idx]]
            #@btime @views $gamma3 = $g3[:,$K,$conf_ket[$c3.idx]]

            tmp1[c3.idx] = K
            #@btime $tmp1[$c3.idx] = $K
            
            tmp_conf = ClusterConfig(SVector(tmp1))
            #@btime $tmp_conf = ClusterConfig(SVector($tmp1))

            
            mul!(scr2, reshape2(scr1, (np*nq,nr)), gamma3)
            #@btime mul!($scr2, reshape2($scr1, ($np*$nq,$nr)), $gamma3)
           
            #@views gamma2 = g2[:,:,conf_ket[c2.idx]]

            # 
            # this is allocating :(  .... but only when threaded? actually @allocated doesn't work when threaded
            mul!(scr3, reshape2(scr2, (np,nq)), gamma2)
            #a = @allocated begin
            #    mul!(scr3, reshape2(scr2, (np,nq)), gamma2)
            #end
            #@allocated mul!(scr3, reshape2(scr2, (np,nq)), gamma2) == 0 || error("why allocate?")
            #@btime mul!($scr3, reshape2($scr2, ($np,$nq)), $gamma2)
            
            #display(typeof(gamma3))
            #display(typeof(gamma2))
            #println()
            if prescreen
                #println("4b")
                #@btime upper_bound_thread($gamma1, $scr3, c=maximum(abs.($coef_ket))) 
                #println(size(gamma1))
                #println(size(scr3))
                #println(typeof(gamma1))
                #println(typeof(scr3))
                upper_bound_thread(gamma1, scr3, c=maximum(abs.(coef_ket))) > thresh || continue
            end
            #println("c")
            
            mul!(scr4, gamma1', scr3)
            #a = @allocated begin
            #    mul!(scr4, gamma1', scr3)
            #end
            #a == 0 || error(" we allocated: ", a)
            #@btime mul!($scr4, $gamma1', $scr3)

            _collect_significant_thread!(sig, tmp_conf, scr4, coef_ket, c1.idx, c2.idx, newI, newJ, thresh, tmp2, state_sign)
        end
    end

    return 
end
#=}}}=#

function _fill1!(gamma, idx, g, k)
    @inbounds for i in 1:size(g,1)
        for (II,I) in enumerate(idx) 
            gamma[i,II] = g[i,I,k]
            #gamma[i,II] = deepcopy(g[i,I,k])
        end
    end
end

function _collect_significant_thread!(out, conf_ket, new_coeffs, coeff, c1idx, newI, thresh, tmp1) 
    #={{{=#
    tmp1 .= conf_ket.config
    thresh_curr = thresh / maximum(abs.(coeff))
    for i::Int16 in newI
        if (new_coeffs[i] > thresh_curr) || (new_coeffs[i] < -thresh_curr)
            tmp1[c1idx] = i
                
            tmp = ClusterConfig(SVector(tmp1))
            if haskey(out, tmp)
                out[tmp] .+= new_coeffs[i].*coeff 
            else
                out[tmp] = new_coeffs[i].*coeff 
            end
        end
    end
end
#=}}}=#

function _collect_significant_thread!(out, conf_ket, new_coeffs, coeff, c1idx, c2idx, newI, newJ, thresh, tmp1, sign) 
#={{{=#
    tmp1 .= conf_ket.config
    thresh_curr = thresh / maximum(abs.(coeff))

    ii::Int = 1
    nI = length(newI)
    nJ = length(newJ)
    @inbounds for j::Int16 in 1:nJ 
        tmp1[c2idx] = newJ[j]
        for i::Int16 in 1:nI
            if abs(new_coeffs[i,j]) > thresh_curr
                tmp1[c1idx] = newI[i]
                tmp = ClusterConfig(SVector(tmp1))
                #@btime haskey($out, $tmp)
                if haskey(out, tmp)
                    ii += 1
                    if sign == 1
                        out[tmp] .+= new_coeffs[i,j] .* coeff
                    elseif sign == -1
                        out[tmp] .-= new_coeffs[i,j] .* coeff
                    else
                        error(" only 1 or -1")
                    end
                else
                    if sign == 1
                        out[tmp] = new_coeffs[i,j].*coeff 
                    elseif sign == -1
                        out[tmp] = -new_coeffs[i,j].*coeff 
                    else
                        error(" only 1 or -1")
                    end
                end
            end
        end
    end
end
#=}}}=#

"""
    upper_bound_thread(g1, g2; c::T=1.0)

Return upper bound on the size of matrix elements resulting from matrix multiply 

    V[I,J] =  g1[i,I] * g2[i,J] * c 

    max(|V|) <= sum_i max|g1[i,:]| * max|g2[i,:]| * |c|
"""
function upper_bound_thread(g1, g2; c::T=1.0) where {T}
#={{{=#
    bound::T = 0.0
    n1 = size(g1,1) 
    n2 = size(g2,1) 
    n1 == n2 || throw(DimensionMismatch)

    absc = abs(c)
    @inbounds @simd for p in 1:n1
        @views pmax = maximum(abs, g1[p,:])
        @views qmax = maximum(abs, g2[p,:])
        bound += pmax * qmax
    end
    bound *= absc
    return bound
end
#=}}}=#

"""
    upper_bound_thread(v::AbstractArray{T,3}, g1, g2, g3, scr1, scr2, scr3; c::T=1.0)

Return upper bound on the size of tensor elements resulting from the following contraction

    V[I,J,K] = v[i,j,k] * g1[i,I] * g2[j,J] * g3[k,K] 

    max(|V|) <= sum_ijk |v[ijk]| * |g1[i,:]|_8 * |g2[j,:]|_8 * |g3[k,:]|_8 
"""
function upper_bound_thread(v::AbstractArray{T,3}, g1, g2, g3, scr1, scr2, scr3; c::T=1.0) where {T}
    #={{{=#
        bound = 0
        n1 = size(g1,1) 
        n2 = size(g2,1) 
        n3 = size(g3,1) 
   
        resize!(scr1, n1)
        resize!(scr2, n2)
        resize!(scr3, n3)
        pmax = scr1 
        qmax = scr2
        rmax = scr3

        fill!(pmax, 0.0)
        fill!(qmax, 0.0)
        fill!(rmax, 0.0)
        
        for p in 1:n1
            @views pmax[p] = maximum(abs(i) for i in g1[p,:])
            #@views pmax[p] = maximum(abs.(g1[p,:]))
        end
        for p in 1:n2
            @views qmax[p] = maximum(abs(i) for i in g2[p,:])
        end
        for p in 1:n3
            @views rmax[p] = maximum(abs(i) for i in g3[p,:])
        end
        
        tmp = 0.0
        @inbounds for r in 1:n3
            for q in 1:n2
                tmp = abs(c) * qmax[q] * rmax[r] 
                @simd for p in 1:n1
                    bound += tmp * abs(v[p,q,r]) * pmax[p]
                end
            end
        end
    return bound
end
#=}}}=#


"""
    upper_bound_thread(v::Array{T,4}, g1, g2, g3, g4, scr1, scr2, scr3, scr4; c::T=1.0)

Return upper bound on the size of tensor elements resulting from the following contraction

    V[I,J,K,L] = v[i,j,k,l] * g1[i,I] * g2[j,J] * g3[k,K] * g4[l,L]

    max(|V|) <= sum_ijkl |v[ijkl]| * |g1[i,:]|_8 * |g2[j,:]|_8 * |g3[k,:]|_8 * |g4[l,:]|_8
"""
function upper_bound_thread(v::Array{T,4}, g1, g2, g3, g4, scr1, scr2, scr3, scr4; c::T=1.0) where {T}
    #={{{=#
        bound = 0
        n1 = size(g1,1) 
        n2 = size(g2,1) 
        n3 = size(g3,1) 
        n4 = size(g4,1) 
   
        resize!(scr1, n1)
        resize!(scr2, n2)
        resize!(scr3, n3)
        resize!(scr4, n4)
        pmax = scr1 
        qmax = scr2
        rmax = scr3
        smax = scr4

        fill!(pmax, 0.0)
        fill!(qmax, 0.0)
        fill!(rmax, 0.0)
        fill!(smax, 0.0)
        
        for p in 1:n1
            @views pmax[p] = maximum(abs(i) for i in g1[p,:])
        end
        for p in 1:n2
            @views qmax[p] = maximum(abs(i) for i in g2[p,:])
        end
        for p in 1:n3
            @views rmax[p] = maximum(abs(i) for i in g3[p,:])
        end
        for p in 1:n4
            @views smax[p] = maximum(abs(i) for i in g4[p,:])
        end
        
        tmp = 0.0
        @inbounds for s in 1:n4
            for r in 1:n3
                for q in 1:n2
                    tmp = abs(c) * qmax[q] * rmax[r] * smax[s] 
                    @simd for p in 1:n1
                        bound += tmp * abs(v[p,q,r,s]) * pmax[p]
                    end
                end
            end
        end
    return bound
end
#=}}}=#


"""
    upper_bound2_thread(v::Array{T,4}, g1, g2, g3, 
        scr_f::Vector{Vector{T}}, scr_i::Vector{Vector{Int16}}, thresh; 
        c::T=1.0)

    max(H_IJ(K)|_K <= sum_s (sum_pqr vpqrs max(g1[p,:]) * max(g2[q,:]) * |c| ) * |g3(r,K)|
"""
function upper_bound2_thread(v::Array{T,3}, g1, g2, g3, 
        scr_f::Vector{Vector{T}}, scr_i::Vector{Vector{Int16}}, thresh; 
        c::T=T(1.0)) where {T}
    #={{{=#
        
    
        newI = scr_i[1]
        newJ = scr_i[2]
        newK = scr_i[3]
        resize!(newI,0)
        resize!(newJ,0)
        resize!(newK,0)
      
        n1 = size(v,1)
        n2 = size(v,2)
        n3 = size(v,3)

        n1 == size(g1,1) || throw(DimensionMismatch)
        n2 == size(g2,1) || throw(DimensionMismatch)
        n3 == size(g3,1) || throw(DimensionMismatch)
       
        pmax = scr_f[5]
        qmax = scr_f[6]
        rmax = scr_f[7]
        
        resize!(pmax,n1)
        resize!(qmax,n2)
        resize!(rmax,n3)


        fill!(pmax, 0.0)
        fill!(qmax, 0.0)
        fill!(rmax, 0.0)

        for p in 1:n1
            @views pmax[p] = maximum(abs, g1[p,:])
        end
        for p in 1:n2
            @views qmax[p] = maximum(abs, g2[p,:])
        end
        for p in 1:n3
            @views rmax[p] = maximum(abs, g3[p,:])
        end
        
        tmp = 0.0

        mI = scr_f[9]
        resize!(mI,size(g1,2))
        fill!(mI, 0.0)
        @inbounds for r in 1:n3
            for q in 1:n2
                tmp = qmax[q] * rmax[r] * abs(c) 
                for p in 1:n1
                    @views @. mI += abs(v[p,q,r]) * abs.(g1[p,:]) * tmp  
                end
            end
        end

        mJ = scr_f[10]
        resize!(mJ,size(g2,2))
        fill!(mJ, 0.0)
        @inbounds for r in 1:n3
            for p in 1:n1
                tmp = pmax[p] * rmax[r] * abs(c)
                for q in 1:n2
                    @views @. mJ += abs(v[p,q,r]) * abs.(g2[q,:])  * tmp
                end
            end
        end

        mK = scr_f[11]
        resize!(mK,size(g3,2))
        fill!(mK, 0.0)
        @inbounds for q in 1:n2
            for p in 1:n1
                tmp = pmax[p] * qmax[q] * abs(c)
                for r in 1:n3
                    @views @. mK += abs(v[p,q,r]) * abs.(g3[r,:]) * tmp 
                end
            end
        end

        for I in 1:size(g1,2)
            if abs(mI[I]) > thresh
                push!(newI,I)
            end
        end

        for J in 1:size(g2,2)
            if abs(mJ[J]) > thresh
                push!(newJ,J)
            end
        end

        for K in 1:size(g3,2)
            if abs(mK[K]) > thresh
                push!(newK,K)
            end
        end

    return newI, newJ, newK
end
#=}}}=#


"""
    upper_bound2_thread(v::Array{T,4}, g1, g2, g3, g4, 
        scr_f::Vector{Vector{T}}, scr_i::Vector{Vector{Int16}}, thresh; 
        c::T=1.0)

    max(H_IJK(L)|_L <= sum_s (sum_pqr vpqrs max(g1[p,:]) * max(g2[q,:]) * max(g3[r,:]) * |c| ) * |g4(s,L)|
"""
function upper_bound2_thread(v::Array{T,4}, g1, g2, g3, g4, 
        scr_f::Vector{Vector{T}}, scr_i::Vector{Vector{Int16}}, thresh; 
        c::T=1.0) where {T}
    #={{{=#
        
    
        newI = scr_i[1]
        newJ = scr_i[2]
        newK = scr_i[3]
        newL = scr_i[4]
        resize!(newI,0)
        resize!(newJ,0)
        resize!(newK,0)
        resize!(newL,0)
      
        n1 = size(v,1)
        n2 = size(v,2)
        n3 = size(v,3)
        n4 = size(v,4)

        n1 == size(g1,1) || throw(DimensionMismatch)
        n2 == size(g2,1) || throw(DimensionMismatch)
        n3 == size(g3,1) || throw(DimensionMismatch)
        n4 == size(g4,1) || throw(DimensionMismatch)
       
        pmax = scr_f[5]
        qmax = scr_f[6]
        rmax = scr_f[7]
        smax = scr_f[8]
        
        resize!(pmax,n1)
        resize!(qmax,n2)
        resize!(rmax,n3)
        resize!(smax,n4)


        fill!(pmax, 0.0)
        fill!(qmax, 0.0)
        fill!(rmax, 0.0)
        fill!(smax, 0.0)

        for p in 1:n1
            @views pmax[p] = maximum(abs, g1[p,:])
        end
        for p in 1:n2
            @views qmax[p] = maximum(abs, g2[p,:])
        end
        for p in 1:n3
            @views rmax[p] = maximum(abs, g3[p,:])
        end
        for p in 1:n4
            @views smax[p] = maximum(abs, g4[p,:])
        end
        
        tmp = 0.0

        mI = scr_f[9]
        resize!(mI,size(g1,2))
        fill!(mI, 0.0)
        @inbounds for s in 1:n4
            for r in 1:n3
                for q in 1:n2
                    tmp = qmax[q] * rmax[r] * smax[s] * abs(c) 
                    for p in 1:n1
                        @views @. mI += abs(v[p,q,r,s]) * abs.(g1[p,:]) * tmp  
                    end
                end
            end
        end

        mJ = scr_f[10]
        resize!(mJ,size(g2,2))
        fill!(mJ, 0.0)
        @inbounds for s in 1:n4
            for r in 1:n3
                for p in 1:n1
                    tmp = pmax[p] * rmax[r] * smax[s] * abs(c)
                    for q in 1:n2
                        @views @. mJ += abs(v[p,q,r,s]) * abs.(g2[q,:])  * tmp
                    end
                end
            end
        end

        mK = scr_f[11]
        resize!(mK,size(g3,2))
        fill!(mK, 0.0)
        @inbounds for s in 1:n4
            for q in 1:n2
                for p in 1:n1
                    tmp = pmax[p] * qmax[q] * smax[s] * abs(c)
                    for r in 1:n3
                        @views @. mK += abs(v[p,q,r,s]) * abs.(g3[r,:]) * tmp 
                    end
                end
            end
        end

        mL = scr_f[12]
        resize!(mL,size(g4,2))
        fill!(mL, 0.0)
        @inbounds for r in 1:n3
            for q in 1:n2
                for p in 1:n1
                    tmp =  pmax[p] * qmax[q] * rmax[r] * abs(c) 
                    for s in 1:n4
                        @views @. mL += abs(v[p,q,r,s]) * abs.(g4[s,:]) * tmp
                    end
                end
            end
        end

        for I in 1:size(g1,2)
            if abs(mI[I]) > thresh
                push!(newI,I)
            end
        end

        for J in 1:size(g2,2)
            if abs(mJ[J]) > thresh
                push!(newJ,J)
            end
        end

        for K in 1:size(g3,2)
            if abs(mK[K]) > thresh
                push!(newK,K)
            end
        end

        for L in 1:size(g4,2)
            if abs(mL[L]) > thresh
                push!(newL,L)
            end
        end

    return newI, newJ, newK, newL 
end
#=}}}=#


