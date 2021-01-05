"""
Abstract type
"""
abstract type AbstractState end
    
"""
    clusters::Vector{Cluster}
    data::OrderedDict{Vector{Tuple{Int16,Int16}},OrderedDict{Vector{Int16},Float64}}
"""
struct ClusteredState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{Vector{Tuple{Int16,Int16}},OrderedDict{Vector{Int16},Float64}}
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
    add_fockconfig!(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
function add_fockconfig!(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
    # initialize ground state of fock
    s.data[fock] = OrderedDict{Vector{Int}, Float64}()
    config = zeros(Int16,length(s.clusters))
    config .+= 1
    s.data[fock][config] = 0
end

"""
    setindex!(s::ClusteredState, a::OrderedDict, b)
"""
function Base.setindex!(s::ClusteredState, a::OrderedDict, b)
    s.data[b] = a
end
"""
    getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
function Base.getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
    return s.data[fock]
end
function Base.getindex(s::ClusteredState, fock)
    return s.data[fock]
end

function Base.length(s::ClusteredState)
    l = 0
    for f in keys(s.data) 
        l += length(s.data[f])
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
    Base.display(s::ClusteredState; thresh=1e-3)

Pretty print
"""
function Base.display(s::ClusteredState; thresh=1e-3)
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- Fockspaces in state ------: Dim = %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "Fock space(α,β)...") 
    @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
    for (fock,configs) in s.data
        prob = 0
        for (config, coeff) in configs 
            prob += coeff*coeff 
        end
        if prob > thresh
            @printf(" %-20.3f%-20i", prob,length(s.data[fock]))
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()
        end
    end
    print(" --------------------------------------------------\n")
end

"""
    print_configs(s::ClusterState; thresh=1e-3)

Pretty print
"""
function print_configs(s::ClusteredState; thresh=1e-3)
    for (fock,configs) in s.data
        length(s.data[fock]) > 0 || continue
        @printf(" Dim %4i fock_space: ",length(s.data[fock]))
        [@printf(" Cluster %-2i(%i:%i) ",fii,fi[1],fi[2]) for (fii,fi) in enumerate(fock)] 
        println()
        for (config, value) in s.data[fock]
            for c in config
                @printf("%3i",c)
            end
            @printf(" %12.8f\n",value)
        end
    end
end

"""
    norm(s::ClusteredState)
"""
function norm(s::ClusteredState)
    norm = 0
    for (fock,configs) in s.data
        for (config,coeff) in configs
            norm += coeff*coeff
        end
    end
    return sqrt(norm)
end

"""
    normalize!(s::ClusteredState)
"""
function normalize!(s::ClusteredState)
    scale!(s,1/norm(s)) 
end

"""
    scale!(s::ClusteredState,c)
"""
function scale!(s::ClusteredState,c)
    for (fock,configs) in s.data
        for (config,coeff) in configs
            s[fock][config] = coeff*c
        end
    end
end
    
"""
    prune_empty_fock_spaces!(s::ClusteredState)
        
remove fock_spaces that don't have any configurations 
"""
function prune_empty_fock_spaces!(s::ClusteredState)
    keylist = keys(s.data)
    for fock in keylist
        if length(s[fock]) == 0
            delete!(s.data, fock)
        end
    end
end

"""
    zero!(s::ClusteredState)

set all elements to zero
"""
function zero!(s::ClusteredState)
    for (fock,configs) in s.data
        for (config,coeffs) in configs                
            s.data[fock][config] = 0
        end
    end
end

"""
    clip!(s::ClusteredState; thresh=1e-5)
"""
function clip!(s::ClusteredState; thresh=1e-5)
    for (fock,configs) in s.data
        for (config,coeff) in configs      
            if abs(coeff) < thresh
                delete!(s.data[fock], config)
            end
        end
    end
    prune_empty_fock_spaces!(s)
end
