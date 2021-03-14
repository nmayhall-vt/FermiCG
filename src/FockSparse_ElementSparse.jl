using StaticArrays

"""
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, Vector{T}}}

This represents an arbitrarily sparse state. E.g., used in TPSCI
"""
struct ClusteredState{T,N,R} <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, SVector{R,T}}}
end
Base.haskey(ts::ClusteredState, i) = return haskey(ts.data,i)
Base.iterate(ts::ClusteredState, state=1) = iterate(ts.data, state)

"""
    ClusteredState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function ClusteredState(clusters; T=Float64, nroots=1)
    N = length(clusters)
    return ClusteredState{T,N,nroots}(clusters,OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, SVector{nroots,T}}}())
end

"""
    add_fockconfig!(s::ClusteredState, fock::FockConfig)
"""
function add_fockconfig!(s::ClusteredState{T,N,R}, fock::FockConfig{N}) where {T<:Number,N,R}
    s.data[fock] = OrderedDict{ClusterConfig{N}, SVector{R,T}}(ClusterConfig([1 for i in 1:N]) => zeros(SVector{R,T}))
end

"""
    getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
Base.getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer = s.data[fock]
Base.getindex(s::ClusteredState, fock) = s.data[fock]
Base.setindex!(s::ClusteredState, a, b) = s.data[b] = a

function Base.size(s::ClusteredState{T,N,R}) where {T,N,R}
    return length(s),R
end
function Base.length(s::ClusteredState)
    l = 0
    for (fock,configs) in s.data 
        l += length(keys(configs))
    end
    return l
end
"""
    get_vector(s::ClusteredState)
"""
function get_vector(s::ClusteredState; root=1)
    v = zeros(length(s))
    idx = 1
    for (fock, configs) in s
        for (config, coeff) in configs
            v[idx] = coeff[1]
            idx += 1
        end
    end
    return v
end
"""
    set_vector!(s::ClusteredState)
"""
function set_vector!(ts::ClusteredState{T,N,R}, v::Matrix{T}) where {T,N,R}

    nbasis = size(v,1)
    nroots = size(v,2)

    length(ts) == nbasis || throw(DimensionMismatch)
    R == nroots || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            ts[fock][tconfig] = SVector{R}(v[idx,:])
            idx += 1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end


"""
    Base.display(s::ClusteredState; thresh=1e-3)

Pretty print
"""
function Base.display(s::ClusteredState; thresh=1e-3, root=1)
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
            prob += coeff[root]*coeff[root] 
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
function print_configs(s::ClusteredState; thresh=1e-3, root=1)
    #display(keys(s.data))
    idx = 1
    for (fock,configs) in s.data
        length(s.clusters) == length(fock) || throw(Exception)
        length(s.data[fock]) > 0 || continue
        @printf(" Dim %4i fock_space: ",length(s.data[fock]))
        [@printf(" %-2i(%i:%i) ",fii,fi[1],fi[2]) for (fii,fi) in enumerate(fock)] 
        println()
        for (config, value) in s.data[fock]
            @printf(" %5i",idx)
            for c in config
                @printf("%3i",c)
            end
            @printf(":%12.8f\n",value[1])
            idx += 1
        end
    end
end

"""
    norm(s::ClusteredState, root)
"""
function LinearAlgebra.norm(s::ClusteredState, root)
    norm = 0
    for (fock,configs) in s.data
        for (config,coeff) in configs
            norm += coeff[root]*coeff[root]
        end
    end
    return sqrt(norm)
end

"""
    norm(s::ClusteredState{T,N,R}, root) where {T,N,R}
"""
function norm(s::ClusteredState{T,N,R}, root) where {T,N,R}
    norms = zeros(T,R)
    for (fock,configs) in s.data
        for (config,coeff) in configs
            for r in 1:R
                norms[r] += coeff[r]*coeff[r]
            end
        end
    end
    for r in 1:R
        norms[r] = sqrt(norms[r])
    end
    return norms
end

"""
    normalize!(s::AbstractState)
"""
function normalize!(s::AbstractState)
    scale!(s,1/sqrt(dot(s,s))) 
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
    dot(v1::ClusteredState,v2::ClusteredState; r1=1, r2=1)
"""
function dot(v1::ClusteredState{T,N,R},v2::ClusteredState{T,N,R}; r1=1, r2=1) where {T,N,R}
    d = T(0)
    for (fock,configs) in v1.data
        haskey(v2.data, fock) || continue
        for (config,coeff) in configs
            haskey(v2.data[fock], config) || continue
            d += coeff[r1] * v2.data[fock][config][r2]
        end
    end
    return d
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
function zero!(s::ClusteredState{T,N,R}) where {T,N,R}
    for (fock,configs) in s.data
        for (config,coeffs) in configs                
            s.data[fock][config] = zeros(SVector{R,T})
        end
    end
end

"""
    clip!(s::ClusteredState; thresh=1e-5)
"""
function clip!(s::ClusteredState; thresh=1e-5)
#={{{=#
    for (fock,configs) in s.data
        for (config,coeff) in configs      
            if abs(coeff) < thresh
                delete!(s.data[fock], config)
            end
        end
    end
    prune_empty_fock_spaces!(s)
end
#=}}}=#