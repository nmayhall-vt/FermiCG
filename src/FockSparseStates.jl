using LinearAlgebra

abstract type SparseConfig{N} end
"""
"""
struct FockConfig{N} <: SparseConfig{N} 
    config::NTuple{N,Tuple{UInt8,UInt8}}
end
"""
"""
struct ClusterConfig{N} <: SparseConfig{N}
    config::NTuple{N,UInt8}
end
"""
"""
struct TransferConfig{N} <: SparseConfig{N}
    config::NTuple{N,Tuple{Int8,Int8}}
end
"""
"""
struct OperatorConfig{T,N} where T<:SparseConfig{N}
    config::Tuple{FockConfig, FockConfig, T{N}, T{N}}
end
"""
    config::Vector{UnitRange{Int}}
"""
struct TuckerConfig{N} <: SparseConfig{N}
    config::NTuple{N,UnitRange{Int}}
end

Base.hash(a::SparseConfig) = hash(a.config)
Base.hash(a::OperatorConfig) = hash(hash.(a.config))

Base.isequal(x::FockConfig, y::FockConfig) = all([all(isequal.(x[i],y[i])) for i in 1:length(x.config)])
Base.isequal(x::TransferConfig, y::TransferConfig) = all([all(isequal.(x[i],y[i])) for i in 1:length(x.config)])
Base.isequal(x::ClusterConfig, y::ClusterConfig) = all(isequal.(x.config, y.config))
Base.isequal(x::OperatorConfig, y::OperatorConfig) = all(isequal.(x.config, y.config))
Base.isequal(x::TuckerConfig, y::TuckerConfig) = all(isequal.(x.config, y.config))

Base.:(==)(x::FockConfig, y::FockConfig) = all([all(x[i].==y[i]) for i in 1:length(x.config)])
Base.:(==)(x::TransferConfig, y::TransferConfig) = all([all(x[i].==y[i]) for i in 1:length(x.config)])
Base.:(==)(x::ClusterConfig, y::ClusterConfig) = all(x.config .== y.config)
Base.:(==)(x::TuckerConfig, y::TuckerConfig) = all(x.config .== y.config)

Base.size(tc::TuckerConfig) = Tuple(length.(tc.config))
Base.push!(tc::TuckerConfig, range) = push!(tc.config,range)


function Base.:(+)(x::FockConfig, y::TransferConfig)
    out = []
    for ci in 1:length(x)
        push!(out, (x[ci][1] + y[ci][1], x[ci][2] + y[ci][2]))
    end
    return FockConfig(out)
end

        
Base.length(f::SparseConfig) = length(f.config)
Base.getindex(s::SparseConfig, i) = s.config[i]
Base.setindex!(s::SparseConfig, i, j) = s.config[j] = i

#Base.display(f::SparseConfig) = display(f.config)
#Base.display(f::ClusterConfig) = display(Int.(f.config))
#Base.display(f::FockConfig) = display(Tuple{Int,Int}.(f.config))
#
#   Iteration through Configs
#Base.eltype(iter::FockConfig) = Tuple{UInt8,UInt8} 
#Base.eltype(iter::ClusterConfig) = UInt8 
#Base.eltype(iter::TransferConfig) = Tuple{Int8,Int8}
Base.iterate(conf::SparseConfig, state=1) = iterate(conf.config, state)

# Conversions
Base.convert(::Type{TransferConfig}, input::Vector{Tuple{T,T}}) where T<:Integer = TransferConfig(input)
Base.convert(::Type{TransferConfig}, input::Tuple{Tuple{T,T}}) where T<:Integer = TransferConfig(input)
Base.convert(::Type{ClusterConfig}, input::Vector{T}) where T<:Integer = ClusterConfig(input)
Base.convert(::Type{ClusterConfig}, input::Tuple{T}) where T<:Integer = ClusterConfig(input)
Base.convert(::Type{FockConfig}, input::Vector{Tuple{T,T}}) where T<:Integer = FockConfig(input)
Base.convert(::Type{FockConfig}, input::Tuple{Tuple{T,T}}) where T<:Integer = FockConfig(input)
Base.convert(::Type{TuckerConfig}, input::Vector{UnitRange{T}}) where T<:Integer = TuckerConfig(input)

TransferConfig(in::Vector{Tuple{T,T}}) where T<:Union{Int16,Int} = TransferConfig([(Int8(i[1]), Int8(i[2])) for i in in]) 
TransferConfig(in::Tuple{Tuple{T,T}}) where T<:Union{Int16,Int} = TransferConfig([(Int8(i[1]), Int8(i[2])) for i in in]) 
FockConfig(in::Vector{Tuple{T,T}}) where T<:Union{Int16,Int} = FockConfig([(UInt8(i[1]), UInt8(i[2])) for i in in]) 
FockConfig(in::Tuple{Tuple{T,T}}) where T<:Union{Int16,Int} = FockConfig([(UInt8(i[1]), UInt8(i[2])) for i in in]) 
ClusterConfig(in::Vector{T}) where T<:Union{Int16,Int} = ClusterConfig([UInt8(i) for i in in]) 
ClusterConfig(in::Tuple{T}) where T<:Union{Int16,Int} = ClusterConfig([UInt8(i) for i in in]) 
#ClusterConfig(in::Vector{Int}) = ClusterConfig([UInt8(i) for i in in]) 

"""
    Base.:-(a::FockConfig, b::FockConfig)

Subtract two `FockConfig`'s, returning a `TransferConfig`
"""
function Base.:-(a::FockConfig, b::FockConfig)
    return TransferConfig([(Int8(a[i][1])-Int8(b[i][1]), Int8(a[i][2])-Int8(b[i][2])) for i in 1:length(a)])
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




