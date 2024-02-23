

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

function correlation_functions(v::TPSCIstate{T,N,R}, cluster_ops, root) where {T,N,R}

    N_1, N_2, Sz_1, Sz_2 = correlation_functions(v)

    N_1 = N_1[root]
    N_2 = N_2[root]
    Sz_1 = Sz_1[root]
    Sz_2 = Sz_2[root]

    H_1 = zeros(T,N)
    H_2 = zeros(T,N,N)

    Hcmf_1 = zeros(T,N)
    Hcmf_2 = zeros(T,N,N)

    S2_1 = zeros(T,N)
    S2_2 = zeros(T,N,N)

    for (fock,configs) in v.data
        prob = 0
        fock_trans = fock - fock
        for (config, coeff) in configs
            prob = coeff[root]*coeff[root]

            for ci in v.clusters

                I = config[ci.idx]
                Hi = cluster_ops[ci.idx]["H"][fock[ci.idx], fock[ci.idx]][I,I] 
                Hcmfi = cluster_ops[ci.idx]["Hcmf"][fock[ci.idx], fock[ci.idx]][I,I] 
                S2i = cluster_ops[ci.idx]["S2"][fock[ci.idx], fock[ci.idx]][I,I] 
                H_1[ci.idx] += prob * Hi 
                Hcmf_1[ci.idx] += prob * Hcmfi  
                S2_1[ci.idx] += prob * S2i

                for cj in v.clusters
                    ci.idx <= cj.idx || continue

                    J = config[cj.idx]
                    Hj = cluster_ops[cj.idx]["H"][fock[cj.idx], fock[cj.idx]][J,J] 
                    Hcmfj = cluster_ops[cj.idx]["Hcmf"][fock[cj.idx], fock[cj.idx]][J,J] 
                    S2j = cluster_ops[cj.idx]["S2"][fock[cj.idx], fock[cj.idx]][J,J] 

                    H_2[ci.idx, cj.idx]     += prob * Hi * Hj 
                    Hcmf_2[ci.idx, cj.idx]  += prob * Hcmfi * Hcmfj 
                    S2_2[ci.idx, cj.idx]    += prob * S2i * S2j
                    
                    H_2[cj.idx, ci.idx]     = H_2[ci.idx, cj.idx]     
                    Hcmf_2[cj.idx, ci.idx]  = Hcmf_2[ci.idx, cj.idx]   
                    S2_2[cj.idx, ci.idx]    = S2_2[ci.idx, cj.idx]    
                end
            end
        end
    end
    
    #N_2 = N_2 - N_1*N_1'
    #Sz_2 = Sz_2 - Sz_1*Sz_1'
    H_2 = H_2 - H_1*H_1'
    Hcmf_2 = Hcmf_2 - Hcmf_1*Hcmf_1'
    S2_2 = S2_2 - S2_1*S2_1'

    println(" <Ni>: ");
    display(N_1)

    println(" <Szi>: ");
    display(Sz_1)
    
    println(" <S2i>: ");
    display(S2_1)
    
    println(" <Hi>: ");
    display(H_1)
    
    println(" <Hcmfi>: ");
    display(Hcmf_1)
    
    println(" <N(i,j)>: ");
    display(N_2)
    
    println(" <Sz(i,j)>: ");
    display(Sz_2)
    
    
    println(" <S2(i,j)>: ");
    display(S2_2)
    
    println(" <Hcmf(i,j)>: ");
    display(Hcmf_2)
end

