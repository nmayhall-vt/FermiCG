using Polynomials



"""
    compute_ac(v_cmf::BSTstate{T,N,R}, 
                    cluster_ops::Vector{ClusterOps{T}}, 
                    clustered_ham::ClusteredOperator{T,N}, 
                    lambda_grid::Vector{T},
                    thresh_var  = 1e-1,
                    thresh_foi  = 1e-4,
                    thresh_pt   = 1e-3;
                    h0="Hcmf") where {N,T,R}

Form a hamiltonian to diagonalize for a give point on an adiabatic connection path
H(λ) = H0 + λ*(H - H0)

H0 = Hcmf + <|H-Hcmf|>
   = Hcmf + <V> 

H(λ) = Hcmf + <V> + λ*(H - Hcmf - <V>)
     = (1-λ)*Hcmf + λ*H + (1-λ)<V> 

dE/dλ  (@λ') = <λ'|H-H0|λ'> 
       = <λ'|H|λ'> - <λ'|Hcmf|λ'> - <V> 
"""
function compute_ac(v_cmf::BSTstate{T,N,R}, 
                    cluster_ops::Vector{ClusterOps{T}}, 
                    clustered_ham::ClusteredOperator{T,N}, 
                    lambda_grid::Vector{T};
                    thresh_var  = 1e-2,
                    thresh_foi  = 1e-6,
                    thresh_pt   = 1e-5,
                    h0="Hcmf") where {N,T,R}

    R == 1 || error(" Not sure how to do multi-state")

    H = deepcopy(clustered_ham) 
    H0 = build_1B_operator(v_cmf.clusters, op_string = "Hcmf", T=T)
    
    E = compute_expectation_value(v_cmf, cluster_ops, H)[1]
    E0 = compute_expectation_value(v_cmf, cluster_ops, H0)[1]
    
    Eshift = E-E0
    
    Eλ = Vector{T}([]) 
    dEλ = Vector{T}([]) 
    dims = Vector{Int}([]) 
    times = Vector{T}([]) 

    v = deepcopy(v_cmf)

    for λ in lambda_grid
        @printf(" λ = %12.8f\n", λ)
        H0_curr = deepcopy(H0) 
        H_curr = deepcopy(H) 
        
        scale!(H_curr, λ)
        scale!(H0_curr, 1-λ)
        
        Hλ = H_curr + H0_curr
     
        time = @elapsed e_var, v = FermiCG.block_sparse_tucker(v, cluster_ops, Hλ,
                                               max_iter    = 20,
                                               max_iter_pt = 200, 
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = thresh_var,
                                               thresh_foi  = thresh_foi,
                                               thresh_pt   = thresh_pt,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false, 
                                               tol_tucker  = 1e-4)
        push!(Eλ, e_var[1] + (1-λ)*Eshift)
       
        E = compute_expectation_value(v, cluster_ops, H)[1]
        E0 = compute_expectation_value(v, cluster_ops, H0)[1]
        push!(dEλ, E-E0-Eshift)
        push!(dims, length(v))
        push!(times, time)
    end

    return Eλ, dEλ, dims, times
end

"""
    quadratic_fits(lambda_grid, dE, E)

dE/dl = ml^2 + nl + b

int_0^1 = m/3 + n/2 + b
"""
function quadratic_fits(lambda_grid, dE, E)
    fits = []
    fits2 = []
    for i in 2:length(lambda_grid)
        quadfit = fit(lambda_grid[1:i], dE[1:i], 2)
        estimate = quadfit.coeffs[3]/3 + quadfit.coeffs[2]/2 + quadfit.coeffs[1] + E[1]
        push!(fits, estimate)
        
        quadfit = fit(lambda_grid[1:i], E[1:i], 2)
        estimate2 = quadfit(1) 
        @printf(" Estimate from λ=%12.8f: %12.8f %12.8f \n", lambda_grid[i], estimate, estimate2)
        push!(fits2, estimate2)
    end
    #for i in 3:length(lambda_grid)
    #    quadfit = fit(lambda_grid[1:i], E[1:i], i)
    #    estimate2 = quadfit(1) 
    #    @printf(" Estimate from %3i points: %12.8f\n", i, estimate2)
    #    push!(fits2, estimate2)
    #end
    return fits, fits2
end
