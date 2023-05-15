"""
    compute_pt1_wavefunction(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        threaded=true) where {T,N,R}
"""
function compute_pt1_wavefunction(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=false,
        verbose=1,
        threaded=true) where {T,N,R}
    #={{{=#

    println()
    println(" |............................do PT1................................")
    println(" thresh_foi    :", thresh_foi   ) 
    println(" prescreen     :", prescreen   ) 
    println(" H0            :", H0   ) 
    println(" nbody         :", nbody   ) 

    e2 = zeros(T,R)
    
    norms = norm(ci_vector);
    println(" Norms of input states")
    [@printf(" %12.8f\n",i) for i in norms]
    println(" Compute FOIS vector")

    if threaded == true
        sig = open_matvec_thread(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    else
        sig = open_matvec_serial(ci_vector, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    end
    println(" Length of FOIS vector: ", length(sig))

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    project_out!(sig, ci_vector)
    println(" Length of FOIS vector: ", length(sig))
    
    @printf(" %-50s", "Compute diagonal")
    @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)
    
    if E0 === nothing
        @printf(" %-50s", "Compute <0|H0|0>:")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)
        #E0 = diag(E0)
        flush(stdout)
    end

    @printf(" %-50s", "Compute <0|H|0>:")
    @time Evar = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)
    #Evar = diag(Evar)
    flush(stdout)
    

    sig_v = get_vector(sig)
    v_pt  = zeros(T, size(sig_v))

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


