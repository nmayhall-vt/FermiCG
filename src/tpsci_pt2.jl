using Distributed
using ThreadPools
using JLD2
using LinearMaps

"""
    compute_batched_pt2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        matvec=3) where {T,N,R}
"""
function compute_batched_pt2(ci_vector::ClusteredState{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        matvec=3) where {T,N,R}
    #={{{=#

    println()
    println(" |........................do batched PT2............................")

    e2 = zeros(T,R)
    
    norms = norm(ci_vector);
    println(" Norms of input states")
    [@printf(" %12.8f\n",i) for i in norms]
    println(" Compute FOIS vector")

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
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
            
        end
    end




    @time sig = open_matvec_batched(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    println(" Length of FOIS vector: ", length(sig))

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    project_out!(sig, ci_vector)
    println(" Length of FOIS vector: ", length(sig))
    
    println(" Compute diagonal")
    @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)
    
    if E0 == nothing
        println(" Compute <0|H0|0>")
        E0 = compute_expectation_value(ci_vector, cluster_ops, clustered_ham_0)
        #[@printf(" %4i %12.8f\n", i, E0[i]) for i in 1:length(E0)]
    end

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

    println(" ..................................................................|")
    return e2, sig 
end
#=}}}=#

