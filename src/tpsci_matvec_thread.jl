using ThreadPools
"""
    open_matvec_thread(ci_vector::ClusteredState, cluster_ops, clustered_ham; thresh=1e-9, nbody=4)

Compute the action of the Hamiltonian on a tpsci state vector. Open here, means that we access the full FOIS 
(restricted only by thresh), instead of the action of H on v within a subspace of configurations. 
This is essentially used for computing a PT correction outside of the subspace, or used for searching in TPSCI.

This parallellizes over FockConfigs in the output state, so it's not the most fine-grained, but it avoids data races in 
filling the final vector
"""
function open_matvec_thread2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham; thresh=1e-9, nbody=4) where {T,N,R}
#={{{=#
    println(" In open_matvec_thread2\n")
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

    scr1 = Vector{Vector{Float64}}()
    scr2 = Vector{Vector{Float64}}()
    scr3 = Vector{MVector{N,Int16}}()

    jobs_out = Vector{ClusteredState{T,N,R}}()
    for tid in 1:Threads.nthreads()
        push!(jobs_out, ClusteredState(clusters, T=T, R=R))
        push!(scr1, zeros(1))
        push!(scr2, zeros(1))
        push!(scr3, zeros(Int16,N))
    end



    println(" Number of jobs:    ", length(jobs_vec))
    #println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    #Threads.@threads for job in jobs_vec
   

    #for job in jobs_vec
    #@qthreads for job in jobs_vec
    @Threads.threads for job in jobs_vec
        fock_bra = job[1]
        tid = Threads.threadid()
        _open_matvec_thread2_job(job[2], fock_bra, cluster_ops, nbody, thresh, jobs_out[tid], scr1[tid], scr2[tid], scr3[tid])
        #@btime _open_matvec_thread2_job($job[2], $fock_bra, $cluster_ops, $nbody, $thresh, $jobs_out[$tid], $scr1[$tid], $scr2[$tid], $scr3[$tid])
    end

    for threadid in 1:Threads.nthreads()
        add!(sig, jobs_out[threadid])
    end

    BLAS.set_num_threads(Threads.nthreads())
    return sig
end
#=}}}=#

function _open_matvec_thread2_job(job, fock_bra, cluster_ops, nbody, thresh, sig, scr1, scr2, scr3)
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
                                       scr1, scr2, scr3, thresh=thresh)
                if term isa ClusteredTerm2B
                    #@code_lowered contract_matvec_thread(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, 
                    #                              sig[fock_bra],scr1, scr2, thresh=thresh)
                    #@code_warntype contract_matvec_thread(term, cluster_ops, fock_bra, fock_ket, config_ket, coeff_ket, 
                    #                              sig[fock_bra],scr1, scr2, thresh=thresh)
                    #@btime contract_matvec_thread($term, $cluster_ops, $fock_bra, $fock_ket, $config_ket, $coeff_ket, 
                    #                              $sig[$fock_bra],$scr1, $scr2, $scr3, thresh=$thresh)
                end
            end
        end
    end
    return 
end
#=}}}=#

reshape2(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

function contract_matvec_thread(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, scr1::Vector{Float64}, scr2::Vector{Float64}, scr3::MVector{N,Int16};
                                    thresh=1e-9) where {T,R,N}
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
    g1::Array{Float64,2} = g1a[(fock_bra[c1.idx],fock_ket[c1.idx])]
    
    @views gamma1 = g1[:,conf_ket[c1.idx]]

    resize!(scr1, size(gamma1,1))

    scr1 .= gamma1 .* state_sign
    newI = 1:size(scr1,1)
    #new_coeffs = scr1
    #newI = 1:size(new_coeffs,1)

    _collect_significant_thread!(sig, conf_ket, scr1, coef_ket, c1.idx,  newI,  thresh, scr3)
    #_collect_significant_thread!(sig, conf_ket, new_coeffs, coef_ket, c1.idx,  newI,  thresh)
            

    return 
end
#=}}}=#


function contract_matvec_thread(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T},
                                    sig, scr1::Vector{Float64}, scr2::Vector{Float64}, scr3::MVector{N,Int16};
                                    thresh=1e-9) where {T,R,N}
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
    g1::Array{Float64,3} = g1a[(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2::Array{Float64,3} = g2a[(fock_bra[c2.idx],fock_ket[c2.idx])]
    
    @views gamma1 = g1[:,:,conf_ket[c1.idx]]
    @views gamma2 = g2[:,:,conf_ket[c2.idx]]
    

    resize!(scr1, size(term.ints,2) * size(gamma1,2))
    resize!(scr2, size(gamma1,2) * size(gamma2,2))
    #@btime resize!($scr1, size($term.ints,2) * size($gamma1,2))
    #@btime resize!($scr2, size($gamma1,2) * size($gamma2,2))
    
    scr1 = reshape2(scr1, (size(term.ints,2), size(gamma1,2)))
    scr2 = reshape2(scr2, (size(gamma1,2), size(gamma2,2)))
    #@btime tmp1 = reshape2($scr1, (size($term.ints,2), size($gamma1,2)))
    #@btime new_coeffs = reshape2($scr2, (size($gamma1,2), size($gamma2,2)))

  
    mul!(scr1, term.ints', gamma1)
    mul!(scr2, scr1', gamma2)
    new_coeffs = scr2
  
    #@btime mul!($tmp1, $term.ints', $gamma1)
    #@btime mul!($new_coeffs, $tmp1', $gamma2)
    #tmp1 .= term.ints' * gamma1 
    #new_coeffs .= tmp1' * gamma2
   
    #coeffs = []
    #for r in 1:R
    #    c = coef_ket[r]*state_sign
    #    push!(coeffs, new_coeff .* c)
    #end

    if state_sign < 0
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(new_coeffs,1)
    newJ = 1:size(new_coeffs,2)


    #display(typeof(scr3))
    #tmp = ClusterConfig{N}(Tuple(i for i in scr3))
    #tmp = ClusterConfig(@SVector [i for i in scr3])
    #a = @MVector [i for i in scr3]
    #tmp = ClusterConfig(@SVector [i for i in scr3])
    #tmp = ClusterConfig{N}(ntuple(i -> convert(Int16, scr3[i]), length(scr3)))
    #tmp2 = ntuple(i -> convert(Int16, scr3[i]), length(scr3))

    _collect_significant_thread!(sig, conf_ket, new_coeffs, coef_ket, c1.idx, c2.idx, newI, newJ, thresh, scr3)
    #@btime _collect_significant_thread!($sig, $conf_ket, $new_coeffs, $coef_ket, $c1.idx, $c2.idx, $newI, $newJ, $thresh, $scr3)

    return 
end
#=}}}=#



function _collect_significant_thread!(out, conf_ket, new_coeffs, coeff, c1idx, newI, thresh, scr3) 
    #={{{=#
    N = length(conf_ket)
    #cket = MVector{N,Int16}([conf_ket.config...])
    #cket = [conf_ket.config...]
    tmp = deepcopy(conf_ket)
    scr3 .= conf_ket.config
    thresh_curr = thresh / maximum(abs.(coeff))
    for i::Int16 in newI
       # if any((abs(new_coeffs[i]*s) > thresh for s in coeff))
        if (new_coeffs[i] > thresh_curr) || (new_coeffs[i] < -thresh_curr)
            scr3[c1idx] = i
                
            tmp = ClusterConfig(SVector(scr3))
            if haskey(out, tmp)
                out[tmp] .+= new_coeffs[i]*coeff 
            else
                out[tmp] = new_coeffs[i]*coeff 
            end
        end
    end
end
#=}}}=#

function _collect_significant_thread!(out, conf_ket, new_coeffs, coeff, c1idx, c2idx, newI, newJ, thresh, scr3) 
#={{{=#
    N = length(conf_ket)
    R = length(coeff)
    tmp = deepcopy(conf_ket)
    scr3 .= conf_ket.config
    #cket = MVector{N,Int16}([conf_ket.config...])
    thresh_curr = thresh / maximum(abs.(coeff))
    for j::Int16 in newJ
        scr3[c2idx] = j
        for i::Int16 in newI
            if (new_coeffs[i,j] > thresh_curr) || (new_coeffs[i,j] < -thresh_curr)
                scr3[c1idx] = i
                tmp = ClusterConfig(SVector(scr3))
                #if haskey(out, ClusterConfig(scr3))
                if haskey(out, tmp)
                    out[tmp] .+= new_coeffs[i,j]*coeff 
                else
                    out[tmp] = new_coeffs[i,j]*coeff 
                end
            end
        end
    end
end
#=}}}=#

