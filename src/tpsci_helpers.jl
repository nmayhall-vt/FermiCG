

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
    correlation_functions(v::TPSCIstate{T,N,R}) where {T,N,R}

Compute <N>, <N1N2 - <N1><N2>>, <Sz>, and <Sz1Sz2 - <Sz1><Sz2>>
"""
function correlation_functions(v::TPSCIstate{T,N,R}) where {T,N,R}

    n1 = [zeros(N) for i in 1:R]
    n2 = [zeros(N,N) for i in 1:R]
    sz1 = [zeros(N) for i in 1:R]
    sz2 = [zeros(N,N) for i in 1:R]

    for root in 1:R
        for (fock,configs) in v.data
            prob = 0
            for (config, coeff) in configs 
                prob += coeff[root]*coeff[root] 
            end

            for ci in v.clusters
                n1[root][ci.idx] += prob * (fock[ci.idx][1] + fock[ci.idx][2])
                sz1[root][ci.idx] += prob * (fock[ci.idx][1] - fock[ci.idx][2]) / 2
                for cj in v.clusters
                    ci.idx <= cj.idx || continue
                    n2[root][ci.idx, cj.idx] += prob * (fock[ci.idx][1] + fock[ci.idx][2]) * (fock[cj.idx][1] + fock[cj.idx][2]) 
                    sz2[root][ci.idx, cj.idx] += prob * (fock[ci.idx][1] - fock[ci.idx][2]) * (fock[cj.idx][1] - fock[cj.idx][2]) / 4
                    n2[root][cj.idx, ci.idx] = n2[root][ci.idx, cj.idx]
                    sz2[root][cj.idx, ci.idx] = sz2[root][ci.idx, cj.idx]
                end
            end
        end
    end

    for r in 1:R
        n2[r] = n2[r] - n1[r]*n1[r]'
        sz2[r] = sz2[r] - sz1[r]*sz1[r]'
    end

    return n1, n2, sz1, sz2
end






"""
    correlation_functions(v::TPSCIstate{T,N,R}, refspace::TPSCIstate{T,N,R}; verbose=1) where {T,N,R}

Compute 1st and 2nd cumulants for finding excitations outside of a reference space of local cluster states present in `refspace`

Returns: 
    out["Q0"][order][root]: Dictionary for all results

order is either 1 or 2
"""
function correlation_functions(v::TPSCIstate{T,N,R}, refspace::TPSCIstate{T,N,R2}; verbose=1) where {T,N,R,R2}
   
    N_1, N_2, Sz_1, Sz_2 = correlation_functions(v)

    cf = Dict{String,Tuple{Vector{Vector{T}}, Vector{Matrix{T}}}}()

    c1 = [zeros(N) for i in 1:R]
    c2 = [zeros(N, N) for i in 1:R]

    p_space = [Dict{Tuple{Int,Int,Int}, Bool}() for i in 1:N]

    for (fock, configs) in refspace.data
        for (config, _) in configs
            for ci in 1:N
                p_space[ci][(fock[ci][1], fock[ci][2], config[ci])] = true
            end
        end
    end 

    display(p_space)
    for root in 1:R
        for (fock, configs) in v.data
            for (config, coeff) in configs
                for ci in 1:N
                    presenti = false
                    if haskey(p_space[ci], (fock[ci][1], fock[ci][2], config[ci]))
                        presenti = true
                    end
                    if presenti == false
                        c1[root][ci] += coeff[root] * coeff[root]
                    end
                    
                    
                        
                    for cj in 1:N
                        presentj = false
                        if haskey(p_space[cj], (fock[cj][1], fock[cj][2], config[cj]))
                            presentj = true
                        end
                        
                        if (presenti == false) && (presentj == false)
                            c2[root][ci, cj] += coeff[root] * coeff[root]
                        end
                    end
                end
            end
        end
    end

    for r in 1:R
        c2[r] = c2[r] - c1[r] * c1[r]'
    end
    
    cf["Q"]    = (c1, c2)
    cf["N"]     = (N_1, N_2)
    cf["Sz"]    = (Sz_1, Sz_2)
    
    if verbose > 0
        @printf(" * κ1(Q) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Q>")
            [@printf(" %12.8f", cf["Q"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ1(N) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<N>")
            [@printf(" %12.8f", cf["N"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ1(Sz) *************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Sz>")
            [@printf(" %12.8f", cf["Sz"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ2(N) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(NI, NJ)")
            for j in 1:N
                [@printf(" %12.8f", cf["N"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        # for r in 1:R
        #     @printf(" Root = %2i: %7s:\n", r, "Corr(NI, NJ)")
        #     for j in 1:N
        #         [@printf(" %12.8f", cf["N"][2][r][i,j]/sqrt(cf["N"][2][r][i,i]*cf["N"][2][r][j,j])) for i in 1:N]
        #         println()
        #     end
        #     println()
        # end

        
        @printf(" * κ2(Sz) *************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(SzI, SzJ)")
            for j in 1:N
                [@printf(" %12.8f", cf["Sz"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        # for r in 1:R
        #     @printf(" Root = %2i: %7s:\n", r, "Corr(SzI, SzJ)")
        #     for j in 1:N
        #         [@printf(" %12.8f", cf["Sz"][2][r][i,j]/sqrt(cf["Sz"][2][r][i,i]*cf["Sz"][2][r][j,j])) for i in 1:N]
        #         println()
        #     end
        #     println()
        # end

        
        @printf(" * κ2(Q) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(QI, QJ)")
            for j in 1:N
                [@printf(" %12.8f", cf["Q"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        # for r in 1:R
        #     @printf(" Root = %2i: %7s:\n", r, "Corr(QI, QJ)")
        #     for j in 1:N
        #         [@printf(" %12.8f", cf["Q"][2][r][i,j]/sqrt(cf["Q"][2][r][i,i]*cf["Q"][2][r][j,j])) for i in 1:N]
        #         println()
        #     end
        #     println()
        # end

    end
    return cf
end

function correlation_functions_old(v::TPSCIstate{T,N,R}, refspace::TPSCIstate{T,N,R2}; verbose=1) where {T,N,R,R2}
   
    N_1, N_2, Sz_1, Sz_2 = correlation_functions(v)

    cf = Dict{String,Tuple{Vector{Vector{T}}, Vector{Matrix{T}}}}()

    c1 = [zeros(N) for i in 1:R]
    c2 = [zeros(N, N) for i in 1:R]

    p_space = [Dict{Tuple{Int,Int}, Dict{Int,Int}}() for i in 1:N]

    for (fock, configs) in refspace.data
        for (config, _) in configs
            for ci in 1:N
                if haskey(p_space[ci], fock[ci]) == false
                    p_space[ci][fock[ci]] = Dict{Int,Int}()
                end
                p_space[ci][fock[ci]][config[ci]] = 1
            end
        end
    end 

    for root in 1:R
        for (fock, configs) in v.data
            for (config, coeff) in configs
                for ci in 1:N
                    presenti = false
                    for (focki, configsi) in refspace.data
                        for (configi, coeffi) in configsi
                            if (focki[ci] == fock[ci]) && (config[ci] == configi[ci]) 
                                presenti = true
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
    
    cf["Q"]    = (c1, c2)
    cf["N"]     = (N_1, N_2)
    cf["Sz"]    = (Sz_1, Sz_2)
    
    if verbose > 0
        @printf(" * κ1(Q) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Q>")
            [@printf(" %12.8f", cf["Q"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ1(N) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<N>")
            [@printf(" %12.8f", cf["N"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ1(Sz) *************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s = ", r, "<Sz>")
            [@printf(" %12.8f", cf["Sz"][1][r][i]) for i in 1:N]
            println()
        end

        
        @printf(" * κ2(N) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(NI, NJ)")
            for j in 1:N
                [@printf(" %12.8f", cf["N"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        # for r in 1:R
        #     @printf(" Root = %2i: %7s:\n", r, "Corr(NI, NJ)")
        #     for j in 1:N
        #         [@printf(" %12.8f", cf["N"][2][r][i,j]/sqrt(cf["N"][2][r][i,i]*cf["N"][2][r][j,j])) for i in 1:N]
        #         println()
        #     end
        #     println()
        # end

        
        @printf(" * κ2(Sz) *************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(SzI, SzJ)")
            for j in 1:N
                [@printf(" %12.8f", cf["Sz"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        # for r in 1:R
        #     @printf(" Root = %2i: %7s:\n", r, "Corr(SzI, SzJ)")
        #     for j in 1:N
        #         [@printf(" %12.8f", cf["Sz"][2][r][i,j]/sqrt(cf["Sz"][2][r][i,i]*cf["Sz"][2][r][j,j])) for i in 1:N]
        #         println()
        #     end
        #     println()
        # end

        
        @printf(" * κ2(Q) **************************\n")
        for r in 1:R
            @printf(" Root = %2i: %7s:\n", r, "Cov(QI, QJ)")
            for j in 1:N
                [@printf(" %12.8f", cf["Q"][2][r][i,j]) for i in 1:N]
                println()
            end
            println()
        end
        # for r in 1:R
        #     @printf(" Root = %2i: %7s:\n", r, "Corr(QI, QJ)")
        #     for j in 1:N
        #         [@printf(" %12.8f", cf["Q"][2][r][i,j]/sqrt(cf["Q"][2][r][i,i]*cf["Q"][2][r][j,j])) for i in 1:N]
        #         println()
        #     end
        #     println()
        # end

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



"""
    full_dim(clusters::Vector{MOCluster}, bases::Vector{ClusterBasis{A,T}}, na, nb) where {A,T}

Compute the total dimension of the Hilbert space defined by our current basis.
- `basis::Vector{ClusterBasis}` 
- `na`: Number of alpha electrons total
- `nb`: Number of alpha electrons total
"""
function full_dim(clusters::Vector{MOCluster}, bases::Vector{ClusterBasis{A,T}}, na, nb) where {A,T}
    # {{{
    println("\n Expand to full space")
    ns = []

    dim_tot = 0
    for c in clusters
        nsi = []
        for (fspace,basis) in bases[c.idx]
            push!(nsi,fspace)
        end
        push!(ns,nsi)
    end
    for newfock in Iterators.product(ns...)
        nacurr = 0
        nbcurr = 0
        for c in newfock
            nacurr += c[1]
            nbcurr += c[2]
        end
        if (nacurr == na) && (nbcurr == nb)
            config = FockConfig(collect(newfock))
            dim_tot += dim(config, clusters)
            #add_fockconfig!(s,config) 
        end
    end
    #expand_each_fock_space!(bases)

    return dim_tot
end
# }}}


