"""
    config::NTuple{N,UnitRange{Int}}

Indexes a particular subspace in the tensor product space. Each `TuckerConfig` instance specifies the entire subspace
defined by a set of cluster states (specified by a range) on each cluster.

E.g., 
- `((1:1), (1:1), (1:1))` is simply the ground state CMF state for a 3 cluster systems. 
- `((2:20), (1:1), (2:20))`  are all states where clusters 1 and 2 are excited out of their repsective ground states.
"""
struct TuckerConfig{N} <: SparseIndex
    config::NTuple{N,UnitRange{Int}}
end


TuckerConfig(in::Vector{T}) where T = convert(TuckerConfig{length(in)}, in)

Base.size(tc::TuckerConfig) = length.(tc.config)
Base.:(==)(x::TuckerConfig, y::TuckerConfig) = all(x.config .== y.config)
"""
    dim(tc::TuckerConfig)

Return total dimension of space indexed by `tc`
"""
dim(tc::TuckerConfig) = prod(size(tc)) 




"""
    function Base.convert(::Type{TuckerConfig{N}}, in::Vector{UnitRange{T}}) where {T,N}
"""
function Base.convert(::Type{TuckerConfig{N}}, in::Vector{UnitRange{T}}) where {T,N}
    return TuckerConfig{length(in)}(ntuple(i -> in[i], length(in)))
end

"""
    function replace(cc::TuckerConfig{N}, idx, conf) where N
"""
function replace(cc::TuckerConfig{N}, idx, conf) where N
    new = [cc.config...]
    #length(idx) == length(conf) || error("wrong dimensions")
    for i in 1:length(idx)
        new[idx[i]] = conf[i]
        #new[idx[i]] = convert(Int16, conf[i])
    end
    return TuckerConfig(new)
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
