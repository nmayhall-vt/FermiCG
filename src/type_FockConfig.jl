
"""
    config::NTuple{N,FockIndex}

Indexes a 'Change in fock space'. For instance, if `F1` and `F2` are distince `FockConfig` instances, 
then `TransferConfig` is `F1-F2`.

e.g., `config = ((2,3), (2,2), (3,2))` is a 3 cluster configuration with 2, 2, 3 α and 3, 2, 2 β electrons, respectively.  
"""
struct FockConfig{N} <: SparseIndex
    config::NTuple{N,FockIndex}
end

FockConfig(in::Vector{Tuple{T,T}}) where T = convert(FockConfig{length(in)}, in)


"""
    function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
"""
function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
    return FockConfig{length(in)}(ntuple(i -> Tuple(convert.(Int16, in[i])), length(in)))
end

"""
"""


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
