using StaticArrays

"""
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, Vector{T}}}

This represents an arbitrarily sparse state. E.g., used in TPSCI
"""
struct ClusteredState{T,N,R} <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, SVector{R,T}}}
end

"""
    ClusteredState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function ClusteredState(clusters; T=Float64, nroots=1)
    N = length(clusters)
    return ClusteredState{T,N,nroots}(clusters,OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, SVector{nroots,T}}}())
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
"""
    get_vector(s::ClusteredState)
"""
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
"""
    set_vector!(s::ClusteredState)
"""
function set_vector!(ts::ClusteredState, v)

    #length(size(v)) == 1 || error(" Only takes vectors", size(v))
    nbasis = size(v)[1]

    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck)

            dim1 = prod(dims)
            ts[fock][tconfig].core .= reshape(v[idx:idx+dim1-1], size(tuck.core))
            idx += dim1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end
