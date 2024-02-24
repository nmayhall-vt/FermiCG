

"""
function single_excitonic_basis(clusters, fspace::FockConfig{N}; R=1, Nk=2, T=Float64) where {N}
"""
function single_excitonic_basis(clusters, fspace::FockConfig{N}; R=1, Nk=2, T=Float64) where {N}
    ci_vector = TPSCIstate(clusters, fspace, R=R)
    config = ones(Int,N) 
    for ci in 1:N
        config_i = deepcopy(config)
        for k in 1:Nk
            config_i[ci] = k
            ci_vector[fspace][ClusterConfig(config_i)] = zeros(Int,R) 
        end
    end
    set_vector!(ci_vector, Matrix{T}(I,size(ci_vector)...))
    return ci_vector
end






"""
    correlation_functions(v::TPSCIstate{T,N,R}, refspace::TPSCIstate{T,N,R}; verbose=1) where {T,N,R}

Compute 1st and 2nd cumulants for finding excitations outside of a reference space of local cluster states present in `refspace`

Returns: 
    out["Q0"][order][root]: Dictionary for all results

order is either 1 or 2
"""
function correlation_functions(v::TPSCIstate{T,N,R}, refspace::TPSCIstate{T,N,R2}; verbose=1) where {T,N,R,R2}

    c1 = [zeros(N) for i in 1:R]
    c2 = [zeros(N, N) for i in 1:R]

    for root in 1:R
        for (fock, configs) in v.data
            for (config, coeff) in configs
                for ci in 1:N
                    presenti = false
                    for (focki, configsi) in refspace.data
                        for (configi, coeffi) in configsi
                            if (focki[ci] == fock[ci]) && (config[ci] == configi[ci]) 
                                presenti = true
                                break
                            end
                        end
                    end
                    if presenti == false
                        c1[root][ci] += coeff[root] * coeff[root]
                    end
                    
                    
                        
                    for cj in 1:N
                        ci <= cj || continue
                        presentj = false
                        for (fockj, configsj) in refspace.data
                            for (configj, coeffj) in configsj
                                if (fockj[cj] == fock[cj]) && (config[cj] == configj[cj]) 
                                    presentj = true
                                    break
                                end
                            end
                        end
                        
                        if (presenti == false) && (presentj == false)
                            c2[root][ci, cj] += coeff[root] * coeff[root]
                            c2[root][cj, ci] += coeff[root] * coeff[root]
                        end
                    end
                end
            end
        end
    end

    for r in 1:R
        c2[r] = c2[r] - c1[r] * c1[r]'
    end
    
    cf = Dict{String,Tuple{Vector{Vector{T}}, Vector{Matrix{T}}}}()
    cf["Q0"]     = (c1, c2)
    
    if verbose > 0
        @printf(" * κ1(P0) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Q0>")
            [@printf(" %12.8f", cf["Q0"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ2(P0) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(Q0I, Q0J)")
            for j in 1:N
                [@printf(" %12.8f", cf["Q0"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Corr(Q0I, Q0J)")
            for j in 1:N
                [@printf(" %12.8f", cf["Q0"][2][r][i,j]/sqrt(cf["Q0"][2][r][i,i]*cf["Q0"][2][r][j,j])) for i in 1:N]
                println()
            end
            println()
        end

    end
    return cf
end


"""
    correlation_functions(v::TPSCIstate{T,N,R}, cluster_ops; verbose=1) where {T,N,R}

Compute cumulants for S2, H, and Hcmf
"""
function correlation_functions(v::TPSCIstate{T,N,R}, cluster_ops::Vector{ClusterOps{T}}; verbose=1) where {T,N,R}

    N_1, N_2, Sz_1, Sz_2 = correlation_functions(v)

    cf = Dict{String,Tuple{Vector{Vector{T}}, Vector{Matrix{T}}}}()
    N_1 = N_1
    N_2 = N_2
    Sz_1 = Sz_1
    Sz_2 = Sz_2

    H_1     = [zeros(T,N) for i in 1:R]
    H_2     = [zeros(T,N,N) for i in 1:R]

    Hcmf_1  = [zeros(T,N) for i in 1:R]
    Hcmf_2  = [zeros(T,N,N) for i in 1:R]

    S2_1    = [zeros(T,N) for i in 1:R]
    S2_2    = [zeros(T,N,N) for i in 1:R]

    for (fock,configs) in v.data
        prob = 0
        fock_trans = fock - fock
        for (config, coeff) in configs

            for ci in v.clusters

                I = config[ci.idx]
                Hi = cluster_ops[ci.idx]["H"][fock[ci.idx], fock[ci.idx]][I,I] 
                Hcmfi = cluster_ops[ci.idx]["Hcmf"][fock[ci.idx], fock[ci.idx]][I,I] 
                S2i = cluster_ops[ci.idx]["S2"][fock[ci.idx], fock[ci.idx]][I,I]
                for r in 1:R
                    prob = coeff[r]*coeff[r]

                    H_1[r][ci.idx] += prob * Hi 
                    Hcmf_1[r][ci.idx] += prob * Hcmfi  
                    S2_1[r][ci.idx] += prob * S2i
                end
                for cj in v.clusters
                    ci.idx <= cj.idx || continue

                    J = config[cj.idx]
                    Hj = cluster_ops[cj.idx]["H"][fock[cj.idx], fock[cj.idx]][J,J] 
                    Hcmfj = cluster_ops[cj.idx]["Hcmf"][fock[cj.idx], fock[cj.idx]][J,J] 
                    S2j = cluster_ops[cj.idx]["S2"][fock[cj.idx], fock[cj.idx]][J,J] 

                    for r in 1:R
                        prob = coeff[r]*coeff[r]

                        H_2[r][ci.idx, cj.idx]     += prob * Hi * Hj 
                        Hcmf_2[r][ci.idx, cj.idx]  += prob * Hcmfi * Hcmfj 
                        S2_2[r][ci.idx, cj.idx]    += prob * S2i * S2j

                        H_2[r][cj.idx, ci.idx]     = H_2[r][ci.idx, cj.idx]     
                        Hcmf_2[r][cj.idx, ci.idx]  = Hcmf_2[r][ci.idx, cj.idx]   
                        S2_2[r][cj.idx, ci.idx]    = S2_2[r][ci.idx, cj.idx]    
                    end
                end
            end
        end
    end

    for r in 1:R
        H_2[r]      = H_2[r] - H_1[r]*H_1[r]'
        Hcmf_2[r]   = Hcmf_2[r] - Hcmf_1[r]*Hcmf_1[r]'
        S2_2[r]     = S2_2[r] - S2_1[r]*S2_1[r]'
    end
    cf["N"]     = (N_1, N_2)
    cf["Sz"]    = (Sz_1, Sz_2)
    cf["H"]     = (H_1, H_2)
    cf["Hcmf"]  = (Hcmf_1, Hcmf_2)
    cf["S2"]    = (S2_1, S2_2)

    if verbose > 0
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<N>")
            [@printf(" %12.8f", cf["N"][1][r][i]) for i in 1:N]
            println()
        end

        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Sz>")
            [@printf(" %12.8f", cf["Sz"][1][r][i]) for i in 1:N]
            println()
        end
        
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<S2>")
            [@printf(" %12.8f", cf["S2"][1][r][i]) for i in 1:N]
            println()
        end
        
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<H>")
            [@printf(" %12.8f", cf["H"][1][r][i]) for i in 1:N]
            println()
        end
        
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Hcmf>")
            [@printf(" %12.8f", cf["Hcmf"][1][r][i]) for i in 1:N]
            println()
        end
        
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(Ni,Nj)")
            for j in 1:N
                [@printf(" %12.8f", cf["N"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end

        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(Szi,Szj)")
            for j in 1:N
                [@printf(" %12.8f", cf["Sz"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end

        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(S2i,S2j)")
            for j in 1:N
                [@printf(" %12.8f", cf["S2"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end


        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(Hi,Hj)")
            for j in 1:N
                [@printf(" %12.8f", cf["H"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end


        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(Hcmf_i,Hcmf_j)")
            for j in 1:N
                [@printf(" %12.8f", cf["Hcmf"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end


    end
    return cf
end

