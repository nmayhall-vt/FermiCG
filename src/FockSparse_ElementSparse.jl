"""
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{ClusterConfig,Float64}}

This represents an arbitrarily sparse state. E.g., used in TPSCI
"""
struct ClusteredState{T,N} <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, T}}
end

"""
    ClusteredState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function ClusteredState(clusters; T=Float64)
    N = length(clusters)
    return ClusteredState{T,N}(clusters,OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, T}}())
end

"""
    add_fockconfig!(s::AbstractState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
function add_fockconfig!(s::AbstractState, fock::Vector{Tuple{T,T}}) where T<:Integer
    add_fockconfig!(s,FockConfig(fock))
end

"""
    add_fockconfig!(s::ClusteredState, fock::FockConfig)
"""
function add_fockconfig!(s::ClusteredState{T,N}, fock::FockConfig{N}) where {T<:Number,N}
    s.data[fock] = OrderedDict{ClusterConfig{N}, T}(ClusterConfig([1 for i in 1:N])=>T(1))
end

"""
    setindex!(s::ClusteredState, a::OrderedDict, b)
"""
function Base.setindex!(s::AbstractState, a, b)
    s.data[b] = a
end
"""
    getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
Base.getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer = s.data[fock]
Base.getindex(s::ClusteredState, fock) = s.data[fock]

function Base.length(s::ClusteredState)
    l = 0
    for (fock,configs) in s.data 
        l += length(keys(configs))
    end
    return l
end
function get_vector(s::ClusteredState)
    v = zeros(length(s))
    idx = 0
    for (fock, configs) in s
        for (config, coeff) in configs
            v[idx] = coeff
            idx += 1
        end
    end
    return v
end
