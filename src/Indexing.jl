using LinearAlgebra
using StaticArrays


Base.convert(::Type{Tuple{T,T}}, in::Tuple{T2,T2}) where {T,T2} = ( convert(T, in[1]), convert(T, in[2]) )

Base.convert(::Type{NTuple{N, Tuple{T,T}}}, in::Vector{Tuple{T2,T2}}) where {T,N,T2} = ntuple(i -> convert(Tuple{T,T}, in[i]), length(in))
#Base.convert(::Type{NTuple{N, Tuple{T,T}}}, in::Vector{Tuple{T,T}}) where {T,N}      = ntuple(i -> in[i], length(in))

"""
"""
const ClusterConfig{N}    = NTuple{N,UInt16}  
const FockConfig{N}       = NTuple{N,Tuple{UInt16,UInt16}} 
const TransferConfig{N}   = NTuple{N,Tuple{Int16,Int16}}
const TuckerConfig{N}     = NTuple{N,UnitRange{UInt16}}
const OperatorConfig{N,T} = Tuple{FockConfig{N}, FockConfig{N}, T, T}

#Base.isequal(x::FockConfig, y::FockConfig) = all([all(isequal.(x[i],y[i])) for i in 1:length(x.config)])
#Base.isequal(x::TransferConfig, y::TransferConfig) = all([all(isequal.(x[i],y[i])) for i in 1:length(x.config)])
##Base.isequal(x::ClusterConfig, y::ClusterConfig) = all(isequal.(x.config, y.config))
#Base.isequal(x::OperatorConfig, y::OperatorConfig) = all(isequal.(x.config, y.config))
#Base.isequal(x::TuckerConfig, y::TuckerConfig) = all(isequal.(x.config, y.config))
#
#Base.:(==)(x::FockConfig, y::FockConfig) = all([all(x[i].==y[i]) for i in 1:length(x.config)])
#Base.:(==)(x::TransferConfig, y::TransferConfig) = all([all(x[i].==y[i]) for i in 1:length(x.config)])
##Base.:(==)(x::ClusterConfig, y::ClusterConfig) = all(x.config .== y.config)
#Base.:(==)(x::TuckerConfig, y::TuckerConfig) = all(x.config .== y.config)
#
#Base.size(tc::TuckerConfig) = Tuple(length.(tc.config))
#Base.push!(tc::TuckerConfig, range) = push!(tc.config,range)


function Base.:(+)(x::FockConfig{N}, y::TransferConfig{N}) where N
#    out = Vector{Tuple{UInt8,UInt8}}()
#    for ci in 1:length(x)
#        push!(out, (convert(UInt8, x[ci][1] + y[ci][1]), convert(UInt8, (x[ci][2] + y[ci][2]))))
#    end
    return FockConfig{N}(((x[ci][1] + y[ci][1]), (x[ci][2] + y[ci][2]) for i in 1:N))
end

        
##
##   Iteration through Configs
##Base.eltype(iter::FockConfig) = Tuple{UInt8,UInt8} 
##Base.eltype(iter::ClusterConfig) = UInt8 
##Base.eltype(iter::TransferConfig) = Tuple{Int8,Int8}
#Base.iterate(conf::SparseConfig, state=1) = iterate(conf.config, state)
#
## Conversions
#Base.convert(::Type{TransferConfig}, input::Vector{Tuple{T,T}}) where T = TransferConfig(ntuple(i -> convert(Tuple{Int8, Int8}, input[i]), length(input)))
#Base.convert(::Type{TransferConfig}, input::Tuple{Tuple{T,T}}) where T  = TransferConfig(ntuple(i -> convert(Tuple{Int8, Int8}, input[i]), length(input)))
#
#Base.convert(::Type{FockConfig}, input::Vector{Tuple{T,T}}) where T  = FockConfig(ntuple(i -> convert(Tuple{UInt8, UInt8}, input[i]), length(input)))
#Base.convert(::Type{FockConfig}, input::Tuple{Tuple{T,T}}) where T  = FockConfig(ntuple(i -> convert(Tuple{UInt8, UInt8}, input[i]), length(input)))
#
#Base.convert(::Type{ClusterConfig}, input::Vector{T}) where T = ClusterConfig(ntuple(i -> convert(UInt8, input[i]), length(input)))
#Base.convert(::Type{TuckerConfig}, input::Vector{T}) where T = TuckerConfig(ntuple(i -> input[i], length(input)))
#
#FockConfig(input::Vector{T}) where T = convert(FockConfig, input)
#TuckerConfig(input::Vector{T}) where T = convert(TuckerConfig, input)
#TransferConfig(input::Vector) = convert(TransferConfig, input)
#ClusterConfig(input::Vector{T}) where T = convert(ClusterConfig, input)


"""
    Base.:-(a::FockConfig, b::FockConfig)

Subtract two `FockConfig`'s, returning a `TransferConfig`
"""
function Base.:-(a::FockConfig, b::FockConfig)
    return TransferConfig([(a[i][1]-b[i][1], a[i][2]-b[i][2]) for i in 1:length(a)])
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




