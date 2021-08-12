"""
Abstract index type
"""
abstract type SparseIndex end

@inline Base.length(a::SparseIndex) = length(a.config)
@inline Base.getindex(s::SparseIndex, i) = s.config[i]
@inline Base.hash(a::SparseIndex) = hash(a.config)
@inline Base.isequal(x::SparseIndex, y::SparseIndex) = isequal(x.config, y.config) 
Base.:(==)(x::SparseIndex, y::SparseIndex) = x.config == y.config 
Base.iterate(conf::SparseIndex, state=1) = iterate(conf.config, state)

