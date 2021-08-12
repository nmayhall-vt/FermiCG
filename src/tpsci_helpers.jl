

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


