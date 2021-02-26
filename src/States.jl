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
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{ClusterConfig,Float64}}

This represents an arbitrarily sparse state. E.g., used in TPSCI
"""
struct ClusteredState{T,N} <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, T}}
end

"""
    ClusteredState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function ClusteredState(clusters)
    return ClusteredState(clusters,OrderedDict())
end

"""
    add_fockconfig!(s::AbstractState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
function add_fockconfig!(s::AbstractState, fock::Vector{Tuple{T,T}}) where T<:Integer
    add_fockconfig!(s,FockConfig(fock))
end

"""
    add_fockconfig!(s::ClusteredState, fock::FockConfig)
"""
function add_fockconfig!(s::ClusteredState{T,N}, fock::FockConfig{N}) where {T<:Number,N}
    s.data[fock] = OrderedDict{ClusterConfig{N}, T}(ClusterConfig([1 for i in 1:N])=>T(1))
end

"""
    setindex!(s::ClusteredState, a::OrderedDict, b)
"""
function Base.setindex!(s::AbstractState, a, b)
    s.data[b] = a
end
"""
    getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
Base.getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer = s.data[fock]
Base.getindex(s::ClusteredState, fock) = s.data[fock]

function Base.length(s::ClusteredState)
    l = 0
    for (fock,configs) in s.data 
        l += length(keys(configs))
    end
    return l
end
function get_vector(s::ClusteredState)
    v = zeros(length(s))
    idx = 0
    for (fock, configs) in s
        for (config, coeff) in configs
            v[idx] = coeff
            idx += 1
        end
    end
    return v
end

"""
Represents a state in an set of abitrary (yet low-rank) subspaces of a set of FockConfigs.
e.g. v[FockConfig][TuckerConfig] => Tucker Decomposed Tensor

    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
"""
struct CompressedTuckerState{T,N} <: AbstractState
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker{T,N}}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
end
Base.haskey(ts::CompressedTuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::CompressedTuckerState, i) = return ts.data[i]
Base.setindex!(ts::CompressedTuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::CompressedTuckerState, state=1) = iterate(ts.data, state)
normalize!(ts::CompressedTuckerState) = scale!(ts, 1/sqrt(orth_dot(ts,ts)))


"""
    TuckerState(ts::CompressedTuckerState)

Convert a `CompressedTuckerState` to a `TuckerState`
Constructor
- ts::CompressedTuckerState`
"""
function TuckerState(ts::CompressedTuckerState{T,N}) where {T,N}
#={{{=#

    ts2 = TuckerState(ts.clusters,
                      OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array{T,N}} }(),
                      ts.p_spaces,
                      ts.q_spaces)

    for (fock, tconfigs) in ts.data
        add_fockconfig!(ts2, fock)
        for (tconfig, tuck) in tconfigs

            ts2[fock][tconfig] = recompose(tuck)
            ts2[fock][tconfig] = reshape(ts2[fock][tconfig],  size(ts2[fock][tconfig])..., 1)
        end
    end
    return ts2 
end
#=}}}=#


"""
    CompressedTuckerState(ts::TuckerState; thresh=-1, max_number=nothing, verbose=0)

Convert a `TuckerState` to a `CompressedTuckerState`
Constructor
- ts::TuckerState`
"""
function CompressedTuckerState(ts::TuckerState; thresh=-1, max_number=nothing, verbose=0)
#={{{=#
    # make all AbstractState subtypes parametric
    T = Float64
    N = length(ts.clusters)
    nroots = nothing
    for (fock,configs) in ts
        for (config,coeffs) in configs
            if nroots == nothing
                nroots = last(size(coeffs))
            else
                nroots == last(size(coeffs)) || error(" mismatch in number of roots")
            end
        end
    end

    nroots == 1 || error(" Conversion to CompressedTuckerState can only have 1 root")

    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker{T,N}} }()
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs

            #
            # Since TuckerState has extra dimension for state index, remove that
            tuck = Tucker(reshape(coeffs,size(coeffs)[1:end-1]), thresh=thresh, max_number=max_number, verbose=verbose)
            if length(tuck) > 0
                if haskey(data, fock)
                    data[fock][tconfig] = tuck
                else
                    data[fock] = OrderedDict(tconfig => tuck)
                end
            end
        end
    end
    return CompressedTuckerState(ts.clusters, data, ts.p_spaces, ts.q_spaces)
end
#=}}}=#

