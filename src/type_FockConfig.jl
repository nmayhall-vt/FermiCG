
"""
    config::NTuple{N,Tuple{Int16,Int16}}

Indexes a 'Change in fock space'. For instance, if `F1` and `F2` are distince `FockConfig` instances, 
then `TransferConfig` is `F1-F2`.

e.g., `config = ((2,3), (2,2), (3,2))` is a 3 cluster configuration with 2, 2, 3 α and 3, 2, 2 β electrons, respectively.  
"""
struct FockConfig{N} <: SparseIndex
    config::NTuple{N,Tuple{Int16,Int16}}
end

FockConfig(in::Vector{Tuple{T,T}}) where T = convert(FockConfig{length(in)}, in)


"""
    function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
"""
function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
    return FockConfig{length(in)}(ntuple(i -> Tuple(convert.(Int16, in[i])), length(in)))
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
    function replace(tc::FockConfig, idx, fock)
"""
function replace(tc::FockConfig, idx, fock)
    new = [tc.config...]
    length(idx) == length(fock) || error("wrong dimensions")
    for i in 1:length(idx)
        new[idx[i]] = (convert(Int16, fock[i][1]), convert(Int16, fock[i][2]))
    end
    return FockConfig(new)
end
