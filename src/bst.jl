using TimerOutputs

"""
    function block_sparse_tucker(input_vec::BSTstate{T,N,R}, cluster_ops, clustered_ham;
        max_iter        = 20,
        max_iter_pt     = 200, # max number of iterations for solving for PT1
        nbody           = 4,
        H0              = "Hcmf",
        thresh_var      = 1e-4,
        thresh_foi      = 1e-6,
        thresh_pt       = 1e-5,
        ci_conv         = 1e-5,
        ci_max_iter     = 50,
        ci_max_ss_vecs  = 12,
	resolve_ss      = false,
        do_pt           = true,
        tol_tucker      = 1e-6 ) where {T,N,R}

# Arguments
- `input_vec::BSTstate`: initial state
- `cluster_ops`: local cluster operators
- `clustered_ham::ClusteredOperator`: hamiltonian
- `max_iter = 20`: max number of iterations
- `max_iter_pt = 200`: max number of iterations to solve for the 1st order wavefunction
- `nbody = 4`: include up to `nbody` terms when forming the FOIS to search
- `H0 = "Hcmf"`: zeroth-order hamiltonian for computing 1st order wavefunction ["Hcmf", "H"]
- `nbody`: max number of nbody terms in the Hamiltonian used for creating FOIS
- `thresh_var`: Compression threshold for the variational solution
- `thresh_foi`: Compression threshold for the FOIS
- `thresh_pt`: Compression threshold for the first-order wavefunction (if used)
- `ci_conv`:     Convergence threshold for the CI (norm of residual)
- `ci_max_iter`:    max iterations for CI problem
- `ci_max_ss_vecs`: max number of subspace vectors for lanczos/davidson
- `resolve_ss`:  After compressing previous variational state, should we resolve in new subspace?
- `do_pt = true`: Compute pt1 wavefunction for finding updated compression basis?
- `tol_tucker`: Convergence threshold for Tucker iterations (energy change)
- `solver`: 
# Returns
- `e_var::Float64`: the final variational energy
- `v_var::BSTstate`: the final variational state

See also: [`BSTstate`](@ref), [`Tucker`](@ref)
"""
function block_sparse_tucker(input_vec::BSTstate{T,N,R}, cluster_ops, clustered_ham;
        max_iter        = 20,
        max_iter_pt     = 200, # max number of iterations for solving for PT1
        nbody           = 4,
        H0              = "Hcmf",
        thresh_var      = 1e-4,
        thresh_foi      = 1e-6,
        thresh_pt       = 1e-5,
        ci_conv         = 1e-5,
        ci_max_iter     = 50,
        ci_max_ss_vecs  = 12,
        resolve_ss      = false,
        do_pt           = true,
        tol_tucker      = 1e-6,
        solver          = "davidson") where {T,N,R}
    #={{{=#
    e_last = 0.0
    e0     = 0.0
    e_var  = 0.0
    e_pt2  = 0.0
    ref_vec = deepcopy(input_vec)
    clustered_S2 = extract_S2(input_vec.clusters)

    to = TimerOutput()
    println(" max_iter       : ", max_iter       ) 
    println(" max_iter_pt    : ", max_iter_pt    ) 
    println(" nbody          : ", nbody          ) 
    println(" H0             : ", H0             ) 
    println(" thresh_var     : ", thresh_var     ) 
    println(" thresh_foi     : ", thresh_foi     ) 
    println(" thresh_pt      : ", thresh_pt      ) 
    println(" ci_conv        : ", ci_conv        ) 
    println(" ci_max_iter    : ", ci_max_iter    ) 
    println(" ci_max_ss_vecs : ", ci_max_ss_vecs ) 
    println(" resolve_ss     : ", resolve_ss     ) 
    println(" do_pt          : ", do_pt          ) 
    println(" tol_tucker     : ", tol_tucker     ) 

    for iter in 1:max_iter
        println()
        println()
        println()
        println(" ===================================================================")
        @printf("     BST Iteration: %4i epsilon: %12.8f\n", iter, thresh_var)
        println(" ===================================================================")

        #
        # Compress Variational Wavefunction
        dim1 = length(ref_vec)
        norm1 = orth_dot(ref_vec, ref_vec)
        ref_vec = compress(ref_vec, thresh=thresh_var)
        orthonormalize!(ref_vec)
        dim2 = length(ref_vec)
        norm2 = orth_dot(ref_vec, ref_vec)
    
        @printf(" %-50s", "Ref state compressed from: ")
        @printf("%10i → %-10i (thresh = %8.1e)\n", dim1, dim2, thresh_var)
        #@printf(" %-50s", "Norm of compressed state: ")
        #@printf("%10.6f\n", norm2)

        # 
        # Solve variationally in reference space
        if resolve_ss
    
            @printf(" %-50s\n", "Get eigenstate for compressed reference space: ")
            flush(stdout)
            @timeit to "CI small" e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, 
                                                                conv_thresh = ci_conv,
                                                                max_iter    = ci_max_iter,
                                                                max_ss_vecs = ci_max_ss_vecs,
                                                                solver = solver)
        else
            tmp = deepcopy(ref_vec)
            zero!(tmp)
            @printf(" %-50s", "Compute zeroth-order energy: ")
            flush(stdout)
            @time build_sigma!(tmp, ref_vec, cluster_ops, clustered_ham)
            e0 = orth_dot(tmp,ref_vec)
        end
        #       sig = deepcopy(ref_vec)
        #       zero!(sig)
        #       build_sigma!(sig, ref_vec, cluster_ops, clustered_ham)
        #       e0 = orth_dot(ref_vec, sig)
        if iter == 1
            e_last = e0
        end

        @printf(" %-50s", "Compute <S^2>: ")
        @time @timeit to "S2" s2 = compute_expectation_value(ref_vec, cluster_ops, clustered_S2) 
        
        @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, e0[r], abs(s2[r]))
        end
        flush(stdout)

        #
        # Get First order wavefunction
        println()
        @printf(" %-50s%10i\n", "Compute FOIS. Reference space dim: ", length(ref_vec) )
        time = @elapsed @timeit to "FOIS" pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
        @printf(" %-50s%10.6f seconds\n", "Total time spent building FOIS: ", time)

        # 
        # Compress FOIS
        norm1 = orth_dot(pt1_vec, pt1_vec)
        dim1 = length(pt1_vec)
       
        if length(pt1_vec) == 0
            error("zero length FOIS?")
        end
        @timeit to "compress" pt1_vec = compress(pt1_vec, thresh=thresh_foi)
        norm2 = orth_dot(pt1_vec, pt1_vec)
        dim2 = length(pt1_vec)
        @printf(" %-50s", "FOIS compressed from: ")
        @printf("%10i → %-10i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
        #@printf(" %-50s%10.8f\n", "Norm of |1>: ",norm2)
        
        ovlp = nonorth_dot(pt1_vec, ref_vec, verbose=0)
        for r in 1:R
            @printf(" %5s %12.8f\n",r, ovlp[r])
        end

        if do_pt
            #
            # 
            println()
            @printf(" %-50s%10i\n", "PT vector reference space dim: ",length(ref_vec))
            time = @elapsed begin
                @timeit to "PT1" pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=ci_conv, max_iter=max_iter_pt, H0=H0)
            end
            @printf(" %-50s%10.6f seconds\n", "Time spent compute PT1 vector: ", time)
        
            # 
            # Compress first order wavefunction 
            norm1 = orth_dot(pt1_vec, pt1_vec)
            dim1 = length(pt1_vec)
            @timeit to "compress" pt1_vec = compress(pt1_vec, thresh=thresh_pt)
            norm2 = orth_dot(pt1_vec, pt1_vec)
            dim2 = length(pt1_vec)
            @printf(" %-50s", "PT compressed from: ")
            @printf("%10i → %-10i (thresh = %8.1e)\n", dim1, dim2, thresh_pt)
            @printf(" %-48s", "Overlap between <1|0>: ")
            ovlp = nonorth_dot(pt1_vec, ref_vec, verbose=0)
            [@printf("%12.8f ",ovlp[r]) for r in 1:R]
            println()
        end

        # 
        # Solve variationally in compressed FOIS 
        # CI
        println()
        time = @elapsed begin
            var_vec = deepcopy(ref_vec)
            #zero!(pt1_vec)
            @timeit to "add" nonorth_add!(var_vec, pt1_vec)
            norm1 = orth_dot(var_vec, var_vec)
            dim1 = length(var_vec)
            @timeit to "compress" var_vec = compress(var_vec, thresh=thresh_pt)
            dim2 = length(var_vec)
            @printf(" %-50s", "Variational space compressed from: ")
            @printf("%10i → %-10i (thresh = %8.1e)\n", dim1, dim2, thresh_pt)
            orthonormalize!(var_vec)
        end
        @printf(" %-50s%10.6f seconds\n", "Add new space to variational space: ", time)
            
        @printf(" %-50s\n", "Solve in compressed FOIS: ")
        @timeit to "CI big" e_var, var_vec = ci_solve(var_vec, cluster_ops, clustered_ham, 
                                                             conv_thresh = ci_conv,
                                                             max_iter    = ci_max_iter,
                                                             max_ss_vecs = ci_max_ss_vecs,
                                                             solver      = solver)

        tmp = deepcopy(var_vec)
        zero!(tmp)
        @printf(" %-50s", "Compute <S^2>: ")
        flush(stdout)
        @time begin
            @timeit to "S2" build_sigma!(tmp, var_vec, cluster_ops, clustered_S2)
        end
        s2 = orth_dot(tmp,var_vec)
        #@printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
        #for r in 1:R
        #    @printf(" %5s %12.8f %12.8f\n",r, e_var[r], abs(s2[r]))
        #end
        @printf(" %-20s", "E(Reference): ")
        [@printf("%12.8f ", e0[r]) for r in 1:R]
        println()

        ref_vec = var_vec

        if do_pt
            @printf(" %-20s", "E(PT2): ")
            [@printf("%12.8f ", e_pt2[r]) for r in 1:R]
            println()
        end
        @printf(" %-20s", "E(BST): ")
        [@printf("%12.8f ", e_var[r]) for r in 1:R]
        println()
    	#show(to)
        println("")

        if maximum(abs.(e_last - e_var)) < tol_tucker 
            @printf("*Converged %-20s", "E(Ref): ")
            [@printf("%12.8f ", e0[r]) for r in 1:R]
            println("")
            @printf("*Converged %-20s", "E(BST): ")
            [@printf("%12.8f ", e_var[r]) for r in 1:R]
            println("")
            show(to)
            println()
            @printf(" ==================================================================|\n")
            return e_var, ref_vec
            break
        end
        e_last = e_var

    end
    @printf(" Not converged %-20s", "E(BST): ")
    [@printf("%12.8f ", e0[r]) for r in 1:R]
    println("")
    show(to)
    return e_var,ref_vec 
end
#=}}}=#
   


"""
    function compute_expectation_value(ci_vector::BSTstate{T,N,R}, cluster_ops, clustered_op::FermiCG.ClusteredOperator; nbody) where {T,N,R}
"""
function compute_expectation_value(vector::BSTstate{T,N,R}, cluster_ops, clustered_op::FermiCG.ClusteredOperator; nbody=4) where {T,N,R}
    tmp = deepcopy(vector)
    zero!(tmp)
    build_sigma!(tmp, vector, cluster_ops, clustered_op, nbody=nbody)
    return orth_dot(tmp,vector)
end
