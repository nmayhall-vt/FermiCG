"""
    config::Tuple{FockConfig{N}, FockConfig{N}, T, T}
"""
struct OperatorConfig{N,T} <: SparseIndex 
    config::Tuple{FockConfig{N}, FockConfig{N}, T, T}
end



