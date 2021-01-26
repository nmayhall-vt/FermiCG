"""
This type is used to characterize a given `TuckerBlock` for a given FockConfig

    cluster::Cluster
    na::Int8
    nb::Int8
    start::UInt8
    stop::UInt8
"""
struct ClusterSubspace
    cluster::Cluster
    na::Int8
    nb::Int8
    start::UInt8
    stop::UInt8
    ClusterSubspace(cluster, na, nb, start, stop) = abs(stop-start+1) > binomial(length(cluster), na)*binomial(length(cluster), nb) ? error("Range too large") : new(cluster, na, nb, start, stop)
end

"""
Defines a list of subspaces across all clusters 

    config::Vector{ClusterSubspace}
"""
struct TuckerBlock
    config::Vector{ClusterSubspace}
end




"""
Represents a state in an set of abitrary (yet dense) subspaces of a set of FockConfigs.
E.g., used in n-body Tucker
    
    clusters::Vector{Cluster}
    data::OrderedDict{TuckerBlock,Array}
    ranges::Vector{UnitRange{Int}}
"""
struct TuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{TuckerBlock,Array}
    ranges::Vector{UnitRange{Int}}
end


"""
"""
function Base.display(tb::TuckerBlock)
    println(" TuckerBlock: ")
    display.(tb.config)
end
function Base.display(css::ClusterSubspace)
    @printf("   Cluster: %3i range: %3i:%-3i ",css.cluster.idx, css.start, css.stop)
    println((css.na, css.nb))
end

"""
"""
function Base.length(css::ClusterSubspace)
    return css.stop - css.start + 1
end
function Base.length(tb::TuckerBlock)
    return sum(length.(tb.config)) 
end





