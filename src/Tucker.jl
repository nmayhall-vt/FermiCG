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
    ClusterSubspace(cluster, na, nb, start, stop) = abs(stop-start+1) > dim_tot(cluster, na, nb) ? error("Range too large") : new(cluster, na, nb, start, stop)
end

"""
Defines a list of subspaces across all clusters 

    config::Vector{ClusterSubspace}
"""
struct TuckerBlock
    config::Vector{ClusterSubspace}
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





"""
"""
struct TuckerConfig <: SparseConfig
    config::Vector{UnitRange{Int}}
end
"""
    dim(tc::TuckerConfig)

Return total dimension of space indexed by `tc`
"""
function dim(tc::TuckerConfig)
    dim = 1
    for range in tc.config
        dim *= length(range)
    end
    return dim
end


"""
Represents a state in an set of abitrary (yet dense) subspaces of a set of FockConfigs.
E.g., used in n-body Tucker
    
    clusters::Vector{Cluster}
    data::OrderedDict{TuckerBlock,Array}
    #ranges::Vector{UnitRange{Int}}
"""
struct TuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}
    #data::OrderedDict{FockConfig,OrderedDict{TuckerBlock,Array}}
    #data::OrderedDict{TuckerBlock,Array}
    #ranges::Vector{UnitRange{Int}}
end
"""
    TuckerState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function TuckerState(clusters)
    return TuckerState(clusters,OrderedDict())
end

"""
    add_fockconfig!(s::ClusteredState, fock::FockConfig)
"""
function add_fockconfig!(s::TuckerState, fock::FockConfig)
    s.data[fock] = OrderedDict{TuckerConfig, Array}(TuckerConfig([1:1 for i in 1:length(s.clusters)])=>[0.0])
end

"""
    Base.length(s::TuckerState)
"""
function Base.length(s::TuckerState)
    l = 0
    for (fock,configs) in s.data 
        for (config,vector) in configs
            l += length(vector)
        end
    end
    return l
end
"""
    get_vector(s::TuckerState)
"""
function get_vector(s::TuckerState)
    v = zeros(length(s))
    idx = 0
    for (fock, configs) in s
        for (config, coeffs) in configs
            v[idx] = coeff
            idx += 1
        end
    end
    return v
end
    

    

"""
    Base.display(s::TuckerState; thresh=1e-3)

Pretty print
"""
function Base.display(s::TuckerState; thresh=1e-3)
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- Fockspaces in state ------: Dim = %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "(α,β)...") 
    @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
    for (fock,configs) in s.data
        prob = 0
        for (config, coeffs) in configs 
            prob += sum(coeffs.*coeffs)
        end
        if prob > thresh
            @printf(" %-20.3f%-20i", prob,length(s.data[fock]))
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()

            @printf("     %-16s%-20s%-20s\n", "Weight", "", "Ranges") 
            @printf("     %-16s%-20s%-20s\n", "-------", "", "----------")
            for (config, coeffs) in configs 
                prob = sum(coeffs.*coeffs)
                if prob > thresh
                    @printf("     %-16.3f%-20i", prob,length(coeffs))
                    for range in config 
                        @printf("%4s", range)
                    end
                end
            end
            println()
        end
    end
    print(" --------------------------------------------------\n")
end


