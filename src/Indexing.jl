using LinearAlgebra
using StaticArrays



abstract type SparseIndex end

@inline Base.length(a::SparseIndex) = length(a.config)
@inline Base.getindex(s::SparseIndex, i) = s.config[i]
@inline Base.hash(a::SparseIndex) = hash(a.config)
@inline Base.isequal(x::SparseIndex, y::SparseIndex) = isequal(x.config, y.config) 
Base.:(==)(x::SparseIndex, y::SparseIndex) = x.config == y.config 
Base.iterate(conf::SparseIndex, state=1) = iterate(conf.config, state)

struct ClusterConfig{N} <: SparseIndex
    config::NTuple{N,Int16}  
end

struct TransferConfig{N} <: SparseIndex
    config::NTuple{N,Tuple{Int16,Int16}}
end

struct FockConfig{N} <: SparseIndex
    config::NTuple{N,Tuple{Int16,Int16}}
end

struct TuckerConfig{N} <: SparseIndex
    config::NTuple{N,UnitRange{Int}}
end

struct OperatorConfig{N,T} <: SparseIndex 
    config::Tuple{FockConfig{N}, FockConfig{N}, T, T}
end



ClusterConfig(in::Vector{T}) where T = convert(ClusterConfig{length(in)}, in)
TransferConfig(in::Vector{Tuple{T,T}}) where T = convert(TransferConfig{length(in)}, in)
FockConfig(in::Vector{Tuple{T,T}}) where T = convert(FockConfig{length(in)}, in)
TuckerConfig(in::Vector{T}) where T = convert(TuckerConfig{length(in)}, in)

Base.size(tc::TuckerConfig) = length.(tc.config)
Base.:(==)(x::TuckerConfig, y::TuckerConfig) = all(x.config .== y.config)
"""
    dim(tc::TuckerConfig)

Return total dimension of space indexed by `tc`
"""
dim(tc::TuckerConfig) = prod(size(tc)) 

function Base.convert(::Type{ClusterConfig{N}}, in::Vector{T}) where {T,N} 
    return ClusterConfig{length(in)}(ntuple(i -> convert(Int16, in[i]), length(in)))
end

function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
    return FockConfig{length(in)}(ntuple(i -> Tuple(convert.(Int16, in[i])), length(in)))
end

function Base.convert(::Type{TransferConfig{N}}, in::Vector{Tuple{T,T}}) where {T,N}
    return TransferConfig{length(in)}(ntuple(i -> convert(Tuple{T,T}, in[i]), length(in)))
end

function Base.convert(::Type{TransferConfig{N}}, in::NTuple{M,Tuple{T,T}}) where {T,N,M}
    return TransferConfig{length(in)}(ntuple(i -> convert(Tuple{T,T}, in[i]), length(in)))
end

function Base.convert(::Type{TuckerConfig{N}}, in::Vector{UnitRange{T}}) where {T,N}
    return TuckerConfig{length(in)}(ntuple(i -> in[i], length(in)))
end

function replace(tc::TransferConfig, idx, fock)
    new = [tc.config...]
    length(idx) == length(fock) || error("wrong dimensions")
    for i in 1:length(idx)
        new[idx[i]] = (convert(Int16, fock[i][1]), convert(Int16, fock[i][2])) 
    end
    return TransferConfig(new)
end



"""
    Base.:+(a::FockConfig, b::TransferConfig)

Add a `FockConfig` to a `TransferConfig` to get a new `FockConfig`
"""
function Base.:(+)(x::FockConfig{N}, y::TransferConfig{N}) where N
    return FockConfig{N}( Tuple( (x[i][1] + y[i][1], x[i][2] + y[i][2]) for i in 1:N) )
end

"""
    Base.:-(a::FockConfig, b::FockConfig)

Subtract two `FockConfig`'s, returning a `TransferConfig`
"""
function Base.:-(a::FockConfig{N}, b::FockConfig{N}) where N
    return TransferConfig( Tuple( (a[i][1]-b[i][1], a[i][2]-b[i][2]) for i in 1:N))
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
