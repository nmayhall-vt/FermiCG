"""
"""
struct FockSparseState{T,N}  
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, T }
end



#const ClusteredState2{T,N} = FockSparseState{OrderedDict{ClusterConfig{N}, T}, N}
#const TuckerState2{T,N} = FockSparseState{OrderedDict{TuckerConfig{N}, Array{T}}, N}
#const CompressedState2{T,N} = FockSparseState{OrderedDict{TuckerConfig{N}, Tucker{T}}, N}
#struct ClusteredState2{T,N}
#    clusters::Vector{Cluster}
#    data::FockSparseState{OrderedDict{ClusterConfig{N}, T}}
#end



"""
Glue different Sparse Vector State types together
"""
abstract type AbstractState end
    


"""
    add_fockconfig!(s::AbstractState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
function add_fockconfig!(s::AbstractState, fock::Vector{Tuple{T,T}}) where T<:Integer
    add_fockconfig!(s,FockConfig(fock))
end
"""
    setindex!(s::AbstractState, a::OrderedDict, b)
"""
function Base.setindex!(s::AbstractState, a, b)
    s.data[b] = a
end
