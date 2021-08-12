
"""
    config::NTuple{N,Int16}  

Indexes an `N` dimensional space 
"""
struct ClusterConfig{N} <: SparseIndex
    #config::SVector{N,Int16}  
    config::NTuple{N,Int16}  
end

@inline ClusterConfig(in::AbstractArray{T,1}) where T = ClusterConfig{length(in)}(ntuple(i -> convert(Int16, in[i]), length(in)))
@inline ClusterConfig(in::AbstractArray{Int16,1}) = ClusterConfig{length(in)}(ntuple(i -> in[i], length(in)))

"""
    function replace(cc::ClusterConfig{N}, idx, conf) where N
"""
function replace(cc::ClusterConfig{N}, idx, conf) where N
    new = [cc.config...]
    #length(idx) == length(conf) || error("wrong dimensions")
    for i in 1:length(idx)
        new[idx[i]] = convert(Int16, conf[i])
    end
    return ClusterConfig(new)
end
