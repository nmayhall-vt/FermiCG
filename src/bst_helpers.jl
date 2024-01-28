

"""
    function bst_single_excitonic_basis(clusters, fspace::FockConfig{N}; R=1, Nk=2, T=Float64) where {N}
"""
function bst_single_excitonic_basis(fspace::FockConfig{N}, ci_vector::BSTstate{T,N,A}; R=1,number_of_single_excitations=4) where {N, T, A}
    max_index = length(fspace.config)
    dims = ones(Int, max_index)
    configs = []
    for i in 2:number_of_single_excitations
        for indices in combinations(1:max_index, 1)
            config = [in(k, indices) ? (i:i) : (1:1) for k in 1:max_index]
            push!(configs, tuple(config...))
        end
    end
    for config in configs
        ci_vector[fspace][FermiCG.TuckerConfig(config)] =
            FermiCG.Tucker(tuple([zeros(Float64, dims...) for _ in 1:R]...))
    end
    return ci_vector
end
"""
        function bst_biexcitonic_basis(clusters, fspace::FockConfig{N}; R=1, Nk=2, T=Float64) where {N}
"""
function bst_biexcitonic_basis(fspace::FockConfig{N},ci_vector::BSTstate{T,N,A}; R=1) where {N, T, A}
    max_index = length(fspace.config)
    configs = []
    for indices in combinations(1:max_index, 2)
        config = [in(k, indices) ? (2:2) : (1:1) for k in 1:max_index]
        push!(configs, tuple(config...))
    end
    dims = ones(Int, max_index)
    for config in configs
        ci_vector[fspace][FermiCG.TuckerConfig(config)] =
            FermiCG.Tucker(tuple([zeros(Float64, dims...) for _ in 1:R]...))
    end

    return ci_vector
end


