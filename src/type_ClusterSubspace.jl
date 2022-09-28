"""
Defines a single cluster's subspace for Tucker. Each focksector is allowed to have a distinct cluster state range 
for the subspace.

    cluster::MOCluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
"""
struct ClusterSubspace
    cluster::MOCluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
end
function ClusterSubspace(cluster::MOCluster)
    return ClusterSubspace(cluster,OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}())
end
Base.haskey(css::ClusterSubspace, i) = return haskey(css.data,i)
Base.setindex!(tss::ClusterSubspace, i, j) = tss.data[j] = i
Base.getindex(tss::ClusterSubspace, i) = return tss.data[i] 
function Base.display(tss::ClusterSubspace)
    @printf(" Subspace for Cluster: %4i : ", tss.cluster.idx)
    display(tss.cluster)
    for (fock,range) in tss.data
        @printf("  %10s   Range: %4i → %-4i Dim %4i\n",Int.(fock), first(range), last(range), length(range))
    end
end


