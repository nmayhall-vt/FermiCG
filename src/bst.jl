using TimerOutputs

"""
    block_sparse_tucker(input_vec::CompressedTuckerState, cluster_ops, clustered_ham;
        max_iter    = 20,
        max_iter_pt = 200, 
        nbody       = 4,
        H0          = "Hcmf",
        thresh_var  = 1e-4,
        thresh_foi  = 1e-6,
        thresh_pt   = 1e-5,
        tol_ci      = 1e-5,
	resolve_ss  = true,
        do_pt       = true,
        tol_tucker  = 1e-6)

# Arguments
- `input_vec::CompressedTuckerState`: initial state
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
- `tol_ci`:     Convergence threshold for the CI (norm of residual)
- `resolve_ss`:  After compressing previous variational state, should we resolve in new subspace?
- `do_pt = true`: Compute pt1 wavefunction for finding updated compression basis?
- `tol_tucker`: Convergence threshold for Tucker iterations (energy change)
# Returns
- `e_var::Float64`: the final variational energy
- `v_var::CompressedTuckerState`: the final variational state

See also: [`CompressedTuckerState`](@ref), [`Tucker`](@ref)
"""
function block_sparse_tucker(input_vec::CompressedTuckerState, cluster_ops, clustered_ham;
        max_iter    = 20,
        max_iter_pt = 200, # max number of iterations for solving for PT1
        nbody       = 4,
        H0          = "Hcmf",
        thresh_var  = 1e-4,
        thresh_foi  = 1e-6,
        thresh_pt   = 1e-5,
        tol_ci      = 1e-5,
	resolve_ss  = true,
        do_pt       = true,
        tol_tucker  = 1e-6)
      #={{{=#
    e_last = 0.0
    e0     = 0.0
    e_var  = 0.0
    e_pt2  = 0.0
    ref_vec = deepcopy(input_vec)
    clustered_S2 = extract_S2(input_vec.clusters)

    to = TimerOutput()

    for iter in 1:max_iter
        println(" --------------------------------------------------------------------")
        println(" Iterate PT-Var:       Iteration #: ",iter)
        println(" --------------------------------------------------------------------")

        #
        # Compress Variational Wavefunction
        dim1 = length(ref_vec)
        norm1 = orth_dot(ref_vec, ref_vec)
        ref_vec = compress(ref_vec, thresh=thresh_var)
        normalize!(ref_vec)
        dim2 = length(ref_vec)
        norm2 = orth_dot(ref_vec, ref_vec)
        @printf(" Compressed Ref state from: %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_var)
        @printf(" Norm of compressed state: %12.8f \n", norm2)
        
        # 
        # Solve variationally in reference space
        println()
        @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
	if resolve_ss
            @timeit to "CI small" e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, tol=tol_ci)
	end
#       sig = deepcopy(ref_vec)
#       zero!(sig)
#       build_sigma!(sig, ref_vec, cluster_ops, clustered_ham)
#       e0 = orth_dot(ref_vec, sig)
        if iter == 1
            e_last = e0
        end
        
        tmp = deepcopy(ref_vec)
        zero!(tmp)
        @timeit to "S2" build_sigma!(tmp, ref_vec, cluster_ops, clustered_S2)
        @printf(" <S^2> = %12.8f\n", orth_dot(tmp,ref_vec))
   
        #
        # Get First order wavefunction
        println()
        println(" Compute first order wavefunction. Reference space dim = ", length(ref_vec))
        @timeit to "FOIS" pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)

        # 
        # Compress FOIS
        norm1 = orth_dot(pt1_vec, pt1_vec)
        dim1 = length(pt1_vec)
        @timeit to "compress" pt1_vec = compress(pt1_vec, thresh=thresh_foi)
        norm2 = orth_dot(pt1_vec, pt1_vec)
        dim2 = length(pt1_vec)
        @printf(" FOIS Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
        @printf(" Norm of |1>:              %12.8f \n", norm2)
        @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))

        if do_pt
            #
            # 
            println()
            println(" Compute PT vector. Reference space dim = ", length(ref_vec))
            @timeit to "PT1" pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol_ci, do_pt=do_pt, max_iter=max_iter_pt, H0=H0)
            # 
            # Compress first order wavefunction 
            norm1 = orth_dot(pt1_vec, pt1_vec)
            dim1 = length(pt1_vec)
            @timeit to "compress" pt1_vec = compress(pt1_vec, thresh=thresh_pt)
            norm2 = orth_dot(pt1_vec, pt1_vec)
            dim2 = length(pt1_vec)
            @printf(" PT   Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_pt)
            @printf(" Norm of |1>:              %12.8f \n", norm2)
            @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))
        end

        # 
        # Solve variationally in compressed FOIS 
        # CI
        println()
        var_vec = deepcopy(ref_vec)
        #zero!(pt1_vec)
        @timeit to "add" nonorth_add!(var_vec, pt1_vec)
        norm1 = orth_dot(var_vec, var_vec)
        dim1 = length(var_vec)
        @timeit to "compress" var_vec = compress(var_vec, thresh=thresh_pt)
        dim2 = length(var_vec)
        @printf(" Var  Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_pt)
        normalize!(var_vec)
        @printf(" Solve in compressed FOIS. Dimension =   %10i\n", length(var_vec))
        @timeit to "CI big" e_var, var_vec = tucker_ci_solve(var_vec, cluster_ops, clustered_ham, tol=tol_ci)
        
        tmp = deepcopy(var_vec)
        zero!(tmp)
        @timeit to "S2" build_sigma!(tmp, var_vec, cluster_ops, clustered_S2)
        @printf(" <S^2> = %12.8f\n", orth_dot(tmp,var_vec))

        ref_vec = var_vec

        @printf(" E(Ref)      = %12.8f\n", e0[1])
        do_pt == false || @printf(" E(PT2) tot  = %12.8f\n", e_pt2)
        @printf(" E(var) tot  = %12.8f\n", e_var[1])

        if abs(e_last[1] - e_var[1]) < tol_tucker 
            println("*Converged")
            show(to)
            return e_var, ref_vec
            break
        end
        e_last = e_var
            
    end
    println(" Not converged")
    show(to)
    return e_var,ref_vec 
end
#=}}}=#
    
