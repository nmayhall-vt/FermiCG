using LinearAlgebra
using StaticArrays

abstract type SparseConfig{N} end
"""
"""
struct FockConfig{N} <: SparseConfig{N} 
    config::NTuple{N,Tuple{UInt8,UInt8}}
    #config::SVector{N,Tuple{UInt8,UInt8}}
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
    #config::SVector{N,Tuple{Int8,Int8}}
end
"""
"""
struct OperatorConfig{N} 
    config::Tuple{FockConfig, FockConfig, SparseConfig{N}, SparseConfig{N}}
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
    out = Vector{Tuple{UInt8,UInt8}}()
    for ci in 1:length(x)
        push!(out, (convert(UInt8, x[ci][1] + y[ci][1]), convert(UInt8, (x[ci][2] + y[ci][2]))))
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
Base.convert(::Type{TransferConfig}, input::Vector{Tuple{T,T}}) where T = TransferConfig(ntuple(i -> convert(Tuple{Int8, Int8}, input[i]), length(input)))
Base.convert(::Type{TransferConfig}, input::Tuple{Tuple{T,T}}) where T  = TransferConfig(ntuple(i -> convert(Tuple{Int8, Int8}, input[i]), length(input)))

Base.convert(::Type{FockConfig}, input::Vector{Tuple{T,T}}) where T  = FockConfig(ntuple(i -> convert(Tuple{UInt8, UInt8}, input[i]), length(input)))
Base.convert(::Type{FockConfig}, input::Tuple{Tuple{T,T}}) where T  = FockConfig(ntuple(i -> convert(Tuple{UInt8, UInt8}, input[i]), length(input)))

Base.convert(::Type{ClusterConfig}, input::Vector{T}) where T = ClusterConfig(ntuple(i -> convert(UInt8, input[i]), length(input)))
Base.convert(::Type{TuckerConfig}, input::Vector{T}) where T = TuckerConfig(ntuple(i -> input[i], length(input)))

FockConfig(input::Vector{T}) where T = convert(FockConfig, input)
TuckerConfig(input::Vector{T}) where T = convert(TuckerConfig, input)
TransferConfig(input::Vector) = convert(TransferConfig, input)


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




