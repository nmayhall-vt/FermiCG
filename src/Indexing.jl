using LinearAlgebra
using StaticArrays



abstract type SparseIndex end

struct ClusterConfig{N} <: SparseIndex
    config::NTuple{N,Int16}  
end

Base.length(a::SparseIndex) = length(a.config)
Base.getindex(s::SparseIndex, i) = s.config[i]
Base.hash(a::SparseIndex) = hash(a.config)
Base.isequal(x::SparseIndex, y::SparseIndex) = isequal(x.config, y.config) 
Base.:(==)(x::SparseIndex, y::SparseIndex) = x.config == y.config 


const FockConfig{N}       = NTuple{N,Tuple{Int16,Int16}} 
const TransferConfig{N}   = Tuple{Vararg{Tuple{Int16,Int16}, N}}
const TuckerConfig{N}     = NTuple{N,UnitRange{Int}}
const OperatorConfig{N,T} = Tuple{FockConfig{N}, FockConfig{N}, T, T}



function ClusterConfig(in::Vector{T}) where T
    return convert(ClusterConfig{length(in)}, in)
end

function Base.convert(::Type{ClusterConfig{N}}, in::Vector{T}) where {T,N} 
    return ClusterConfig{N}(ntuple(i -> convert(Int16, in[i]), length(in)))
end

function Base.convert(::Type{TransferConfig}, in::Vector{Tuple{T,T}}) where {T,N} 
    return ntuple(i -> convert(Tuple{T,T}, in[i]), length(in))
end


Base.size(tc::TuckerConfig) = Tuple(length.(tc))


"""
    Base.:-(a::FockConfig, b::TransferConfig)

Add a `FockConfig` to a `TransferConfig` to get a new `FockConfig`
"""
function Base.:(+)(x::FockConfig{N}, y::TransferConfig{N}) where N
    return FockConfig{N}( ( (x[i][1] + y[i][1], x[i][2] + y[i][2]) for i in 1:N) )
end

"""
    Base.:-(a::FockConfig, b::FockConfig)

Subtract two `FockConfig`'s, returning a `TransferConfig`
"""
function Base.:-(a::FockConfig{N}, b::FockConfig{N}) where N
    return TransferConfig(( (a[i][1]-b[i][1], a[i][2]-b[i][2]) for i in 1:N))
end
"""
    dim(fc::FockConfig, no)

Return total dimension of space indexed by `fc` on `no` orbitals
"""
function dim(fc::FockConfig, clusters)
    dim = 1
    for ci in clusters
        dim *= binomial(length(ci), fc[ci.idx][1]) * binomial(length(ci), fc[ci.idx][2])
    end
    return dim
end




"""
Check if `tc1` is a subset of `tc2`
"""
function is_subset(tc1::TuckerConfig, tc2::TuckerConfig)
    length(tc1.config) == length(tc2.config) || return false
    for i in 1:length(tc1)
        if first(tc1[i]) < first(tc2[i]) || last(tc1[i]) > last(tc2[i])
            return false
        end
    end
    return true
end
