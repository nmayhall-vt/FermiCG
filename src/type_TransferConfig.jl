
"""
    config::NTuple{N,Tuple{Int16,Int16}}

Indexes a 'Change in fock space'. For instance, if `F1` and `F2` are distince `FockConfig` instances, 
then `TransferConfig` is `F1-F2`.

This is mainly used to label individual parts of an operator (Hamiltonian) as they can only enact certain changes in the 
Fock space configurations.
"""
struct TransferConfig{N} <: SparseIndex
    config::NTuple{N,Tuple{Int16,Int16}}
end


TransferConfig(in::Vector{Tuple{T,T}}) where T = convert(TransferConfig{length(in)}, in)

"""
    function Base.convert(::Type{TransferConfig{N}}, in::Vector{Tuple{T,T}}) where {T,N}
"""
function Base.convert(::Type{TransferConfig{N}}, in::Vector{Tuple{T,T}}) where {T,N}
    return TransferConfig{length(in)}(ntuple(i -> convert(Tuple{T,T}, in[i]), length(in)))
end

"""
    function Base.convert(::Type{TransferConfig{N}}, in::NTuple{M,Tuple{T,T}}) where {T,N,M}
"""
function Base.convert(::Type{TransferConfig{N}}, in::NTuple{M,Tuple{T,T}}) where {T,N,M}
    return TransferConfig{length(in)}(ntuple(i -> convert(Tuple{T,T}, in[i]), length(in)))
end


"""
    function replace(tc::TransferConfig, idx, fock)
"""
function replace(tc::TransferConfig, idx, fock)
    new = [tc.config...]
    length(idx) == length(fock) || error("wrong dimensions")
    for i in 1:length(idx)
        new[idx[i]] = (convert(Int16, fock[i][1]), convert(Int16, fock[i][2])) 
    end
    return TransferConfig(new)
end


