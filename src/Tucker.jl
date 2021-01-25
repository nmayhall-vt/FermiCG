struct ClusterSubspace
    cluster::Cluster
    na::Int8
    nb::Int8
    start::UInt8
    stop::UInt8
end


"""
This type is used to characterize a given ``TuckerBlock'' for a given FockConfig

`config` is a vector containing the following information: 
    (#alpha, #beta, start_index, stop_index)
"""
struct TuckerBlock
    config::Vector{ClusterSubspace}
end
function Base.display(tb::TuckerBlock)
    println(" TuckerBlock: ")
    display.(tb.config)
end


function Base.display(css::ClusterSubspace)
    @printf("   Cluster: %3i range: %3i:%-3i ",css.cluster.idx, css.start, css.stop)
    println((css.na, css.nb))
end
function Base.length(css::ClusterSubspace)
    return css.stop - css.start + 1
end









"""
    clusters::Vector{Cluster}
    data::OrderedDict{TuckerConfig,Array}

Represents a state in an set of abitrary (yet dense) subspaces of a set of FockConfigs.
E.g., used in n-body Tucker
"""
struct TuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{TuckerBlock,Array}
    ranges::Vector{UnitRange{Int}}
end
