using Profile
using LinearMaps
using BenchmarkTools
#using KrylovKit
using IterativeSolvers




"""
Represents a state in an set of abitrary (yet dense) subspaces of a set of FockConfigs.

v[FockConfig][TuckerConfig] => Dense matrix: C(n1*n2*...nN, R)
E.g., used in n-body Tucker
    
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
"""
struct TuckerState{T,N,R} <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Array{T} }}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
end
Base.haskey(ts::TuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::TuckerState, i) = return ts.data[i]
Base.setindex!(ts::TuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::TuckerState, state=1) = iterate(ts.data, state)



# Conversions
#Base.convert(::Type{TuckerConfig}, input::Vector{UnitRange{T}}) where T<:Integer = TuckerConfig(input)



"""
    TuckerState(clusters::Vector{Cluster}, 
        fconfig::FockConfig{N}, 
        cluster_bases::Vector{ClusterBasis}) where {N} 

Constructor using only a single FockConfig. This allows us to turn the CMF state into a TuckerState.
As such, it chooses the ground state of each cluster in the Fock sector specified by `FockConfig` to be the 
P space, and then the Q space is defined as the orthogonal complement of this state within the available basis, 
specified by `cluster_bases`.
# Arguments
- `clusters`: vector of clusters types
- `fconfig`: starting FockConfig 
- `cluster_basis`: list of ClusterBasis types - needed to know the dimensions of the q-spaces
# Returns
- `TuckerState`
"""
function TuckerState(clusters::Vector{Cluster}, 
        fconfig::FockConfig{N}, 
        cluster_bases::Vector{ClusterBasis}; T=Float64, R=1) where {N} 
    #={{{=#

    # 
    # start by building the P and Q spaces needed
    p_spaces = Vector{ClusterSubspace}()
    q_spaces = Vector{ClusterSubspace}()
    # define p spaces
    for ci in clusters
        tss = ClusterSubspace(ci)
        tss[fconfig[ci.idx]] = 1:1
        push!(p_spaces, tss)
    end

    # define q spaces
    for tssp in p_spaces 
        tss = get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Array{T,2}} }()
    state = TuckerState{T,N,R}(clusters, data, p_spaces, q_spaces) 
    add_fockconfig!(state, fconfig)
    
    tconfig = TuckerConfig([p_spaces[ci.idx].data[fconfig[ci.idx]] for ci in clusters])
    
    state[fconfig][tconfig] = zeros(T,dim(tconfig),R) 
    return state
end
#=}}}=#



"""
    function TuckerState(clusters::Vector{Cluster}, 
        p_spaces::Vector{FermiCG.ClusterSubspace}, 
        q_spaces::Vector{FermiCG.ClusterSubspace}) 

Constructor - specify input p and q spaces
# Arguments
- `clusters`: vector of clusters types
- `p_spaces`: list of p space ranges for each cluster
- `q_spaces`: list of q space ranges for each cluster
# Returns
- `TuckerState`
"""
function TuckerState(clusters::Vector{Cluster}, 
        p_spaces::Vector{FermiCG.ClusterSubspace}, 
        q_spaces::Vector{FermiCG.ClusterSubspace}; T=Float64, R=1) 
    #={{{=#

    N = length(clusters)
    length(p_spaces) == N || error(DimensionMismatch)
    length(q_spaces) == N || error(DimensionMismatch)

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Array{T,2}} }()
     
    return TuckerState{T,N,R}(clusters, data, p_spaces, q_spaces) 
end
#=}}}=#



"""
    function TuckerState(v::TuckerState{T,N,R}; T=T, R=R) where {T,N,R}

Constructor creating a `TuckerState` with the same basis as `v`, but with potentially different `R` and `T`. 
Coefficients of new vector are 0.0

# Arguments
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TuckerState`
"""
function TuckerState(v::TuckerState{TT,NN,RR}; T=TT, R=RR) where {TT,NN,RR}
    out = TuckerState(v.clusters, v.p_spaces, v.q_spaces, T=T, R=R)
    for (fock, configs) in v.data
        add_fockconfig!(out,fock)
        for (config, coeffs) in configs
            out[fock][config] = zeros(T,dim(config),R)
        end
    end
    return out
end


"""
    TuckerState(clusters; T=Float64, R=1)

Constructor creating an empty vector
# Arguments
- `clusters::Vector{Cluster}`
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TuckerState`
"""
function TuckerState(clusters, p_spaces, q_spaces; T=Float64, R=1)
    N = length(clusters)
    return TuckerState{T,N,R}(clusters,
                              p_spaces, q_spaces, 
                              OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, MVector{R,T}}}())
end

#"""
#    TuckerState(clusters, p_spaces, q_spaces, foi; nroots=1)
#
#Constructor to build state directly from a definition of a first order interacting space (or more generic even i suppose)
#- `clusters::Vector{Cluster}`
#- `p_spaces::Vector{ClusterSubspace}`
#- `q_spaces::Vector{ClusterSubspace}`
#- `foi::OrderedDict{FockConfig,Vector{TuckerConfig}}` 
#- `nroots`
#"""
#function TuckerState(clusters::Vector{Cluster}, p_spaces::Vector{ClusterSubspace}, q_spaces::Vector{ClusterSubspace}, 
#        foi::OrderedDict{FockConfig,Vector{TuckerConfig}}; nroots=1)
#   #={{{=#
#    length(p_spaces) == length(clusters) || error("# of clusters don't match # of subspaces")
#    length(p_spaces) == length(q_spaces) || error(" p_spaces/q_spaces don't have same length", length(p_space), length(q_space))
#    for ci in clusters
#        for fock in p_spaces[ci.idx].data
#            if haskey(q_spaces[ci.idx].data, fock)
#                all(collect(p_spaces[ci.idx][fock]) .!= collect(q_spaces[ci.idx][fock])) || error(" Not orthogonal")
#            end
#        end
#    end
#
#    #s = TuckerState(clusters)
#    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}()
#    for (fock,tconfig_list) in foi
#        data2 = OrderedDict{TuckerConfig,Array}()
#        for tconfig in tconfig_list
#            data2[tconfig] = zeros((size(tconfig)...,nroots))
#        end
#
#        data[fock] = data2 
#    end
#    return TuckerState(clusters, data, p_spaces, q_spaces) 
##=}}}=#
#end

"""
    +(ts1::FermiCG.TuckerState, ts2::FermiCG.TuckerState)
"""
function Base.:+(ts0::TuckerState, ts2::TuckerState)
#={{{=#
    ts1 = deepcopy(ts0)
    add!(ts1,ts2)
    return ts1
#=}}}=#
end

"""
    add!(ts1::FermiCG.TuckerState, ts2::FermiCG.TuckerState)

Add coeffs in `ts2` to `ts1`
"""
function add!(ts1::TuckerState, ts2::TuckerState)
#={{{=#
    for (fock,configs) in ts2
        if haskey(ts1, fock)
            for (config,coeffs) in configs 
                if haskey(ts1[fock], config)
                    ts1[fock][config] .+= ts2[fock][config]
                else
                    ts1[fock][config] = deepcopy(ts2[fock][config])
                end
            end
        else
            ts1[fock] = deepcopy(ts2[fock])
        end
    end
#=}}}=#
end
"""
    dot(ts1::FermiCG.TuckerState, ts2::FermiCG.TuckerState)

Dot product between in `ts2` to `ts1`
"""
function dot(ts1::TuckerState{T,N,R1}, ts2::TuckerState{T,N,R2}) where {T,N,R1,R2}
#={{{=#
   
    unfold!(ts1)
    unfold!(ts2)
   
    overlap = zeros(R1,R2) 
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        for (config,coeffs) in configs 
            haskey(ts1[fock], config) || continue
            overlap .+= ts1[fock][config]' * coeffs
        end
    end
    fold!(ts1)
    fold!(ts2)
    return overlap
#=}}}=#
end

"""
Scale `ts` by a constant `a`
"""
function scale!(ts::TuckerState{T,N,R}, a::Vector{T}) where {T,N,R}
    #={{{=#
    unfold!(ts)
    for r in 1:R
        for (fock,configs) in ts
            for (config,tuck) in configs 
                ts[fock][config][:,r] .*= a[r] 
            end
        end
    end
    #=}}}=#
end


"""
    unfold!(ts::TuckerState)
"""
function unfold!(ts::TuckerState{T,N,R}) where {T,N,R}
#={{{=#
    for (fock,configs) in ts.data
        #display(fock)
        for (config,coeffs) in configs 
            #length(size(coeffs)) == length(ts.clusters)+1 || error(" Can only unfold a folded vector")
            #if length(size(coeffs)) != length(ts.clusters)+1 
                ts[fock][config] = reshape(coeffs, (prod(size(config)),R))
            #end
        end
    end
#=}}}=#
end
"""
    fold!(ts::TuckerState)
"""
function fold!(ts::TuckerState{T,N,R}) where {T,N,R}
#={{{=#
    for (fock,configs) in ts.data
        for (config,coeffs) in configs 
            #if length(size(coeffs)) != 2
                ts[fock][config] = reshape(coeffs, (size(config)...,R))
            #end
        end
    end
#=}}}=#
end

"""
    randomize!(ts::TuckerState; scale=1)

Add some random noise to the vector
"""
function randomize!(ts::TuckerState; scale=1)
    #={{{=#
    for (fock,configs) in ts
        for (config,coeffs) in configs 
            ts[fock][config] += scale .* (rand(size(coeffs)...) .- .5)
        end
    end
    #=}}}=#
end

"""
    mult!(ts::TuckerState, A)

Multiple `ts` by a matrix A. This is a multiplication over global state index
"""
function mult!(ts::TuckerState, A)
    #={{{=#
    unfold!(ts)
    for (fock,configs) in ts
        for (config,coeffs) in configs 
            ts[fock][config] = coeffs * A
        end
    end
    fold!(ts)
    #=}}}=#
end

"""
    orthogonalize!(ts::TuckerState)

Symmetric Orthogonalization of vectors in ts
"""
function orthogonalize!(ts::TuckerState)
    #={{{=#
    S = dot(ts,ts)
    F = eigen(S)
    l = sqrt.(inv.(F.values))
    U = F.vectors
    for li in 1:length(l)
        U[:,li] .*= l[li]
    end
    mult!(ts,U)
    #=}}}=#
end


"""
"""
function add_config!(s::TuckerState{T,N,R}, fock::FockConfig{N}, config::TuckerConfig{N}) where {T,N,R}
    s.data[fock][tconf] = zero(T,dim(config),R)
end


"""
    function add_fockconfig!(s::TuckerState{T,N}, fock::FockConfig) where {T,N}
"""
function add_fockconfig!(s::TuckerState{T,N}, fock::FockConfig) where {T,N}
    s.data[fock] = OrderedDict{TuckerConfig, Array{T,N}}()
end

"""
    Base.length(s::TuckerState)
"""
function Base.length(s::TuckerState{T,N,R}) where {T,N,R}
    l = 0
    for (fock,configs) in s.data 
        for (config,vector) in configs
            l += dim(config) 
        end
    end
    return l
end

"""
    function Base.size(s::TuckerState{T,N,R}) where {T,N,R}
"""
function Base.size(s::TuckerState{T,N,R}) where {T,N,R}
    return (length(s),R)
end

"""
    prune_empty_fock_spaces!(s::TuckerState)
        
remove fock_spaces that don't have any configurations 
"""
function prune_empty_fock_spaces!(s::TuckerState)
    focklist = keys(s.data)
    for fock in focklist 
        if length(s.data[fock]) == 0
            delete!(s.data, fock)
        end
    end
    focklist = keys(s.data)
    #println(" Here's what's left")
    #for fock in keys(s.data)
    #end
    #    display(fock)
    #    display(s[fock])
    #end
end
"""
"""
function get_vector(ts::TuckerState{T,N,R}) where {T,N,R}
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

    nroots != nothing || error(" Couldn't find nroots? length:", length(ts))

    v = zeros(length(ts), nroots)
    #println(length(ts), nroots)
    idx = 1
    for (fock, configs) in ts
        for (config, coeffs) in configs
            dims = size(coeffs)
            
            dim1 = prod(dims[1:end-1])
            v[idx:idx+dim1-1,:] = copy(reshape(coeffs,dim1,nroots))
            idx += dim1 
        end
    end
    return v
end

"""
    function set_vector!(ts::TuckerState{T,N,R}, v) where {T,N,R}
"""
function set_vector!(ts::TuckerState{T,N,R}, v::Matrix{T}) where {T,N,R}

    nbasis = size(v)[1]
  
    unfold!(ts)
    #println(length(ts), nroots)
    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, coeffs) in tconfigs
            dim1 = dim(tconfig)
            ts[fock][tconfig] = copy(v[idx:idx+dim1-1,:])
            idx += dim1 
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    fold!(ts)
    return 
end

"""
    zero!(s::TuckerState)
"""
function zero!(s::TuckerState)
    for (fock, configs) in s
        for (config, coeffs) in configs
            fill!(s[fock][config], 0.0)
        end
    end
end
"""
    eye!(s::TuckerState)
"""
function eye!(s::TuckerState{T,N,R}) where {T,N,R}
    set_vector!(s, Matrix{T}(I,size(s)))
end
"""
    rand!(s::TuckerState)
"""
function rand!(s::TuckerState{T,N,R}) where {T,N,R}
    set_vector!(s, rand(T,size(s)))
end
    
    

"""
    Base.display(s::TuckerState; thresh=1e-3)

Pretty print
"""
function Base.display(s::TuckerState; root=1, thresh=1e-3)
#={{{=#
    println()
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- State        -------------------: %5i  \n",root)
    @printf(" ---------- # Fockspaces -------------------: %5i  \n",length(keys(s.data)))
    @printf(" ---------- # Configs    -------------------: %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "(α,β)...") 
    @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
    unfold!(s)
    for (fock,configs) in s.data
        prob = 0
        len = 0
        for (config, coeffs) in configs 
            prob += coeffs[:,root]' * coeffs[:,root]
            len += length(coeffs[:,root])
        end
        if prob > thresh
            @printf(" %-20.3f%-20i", prob,len)
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()

            @printf("     %-16s%-20s%-20s\n", "Weight", "", "Subspaces") 
            @printf("     %-16s%-20s%-20s\n", "-------", "", "----------")
            for (config, coeffs) in configs 
                probi = coeffs[:,root]' * coeffs[:,root]
                if probi > thresh
                    @printf("     %-16.3f%-20i", probi,length(coeffs))
                    for range in config 
                        @printf("%7s", range)
                    end
                    println()
                end
            end
            println()
        end
    end
    fold!(s)
    print(" --------------------------------------------------\n")
    println()
#=}}}=#
end
"""
    print_fock_occupations(s::TuckerState; root=1, thresh=1e-3)

Pretty print
"""
function print_fock_occupations(s::TuckerState; root=1, thresh=1e-3)
#={{{=#

    println()
    unfold!(s)
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- State        -------------------: %5i  \n",root)
    @printf(" ---------- # Fockspaces -------------------: %5i  \n",length(keys(s.data)))
    @printf(" ---------- # Configs    -------------------: %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "(α,β)...") 
    @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
    sum = 0
    for (fock,configs) in s.data
        prob = 0
        len = 0
        for (config, coeffs) in configs 
            prob += coeffs[:,root]' * coeffs[:,root]
            len += length(coeffs[:,root])
        end
        if prob > thresh
            @printf(" %-20.3f%-20i", prob,len)
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()
        end
        sum += prob
    end
    print(" --------------------------------------------------\n")
    fold!(s)
    println()
#=}}}=#
end



"""
    expand_each_fock_space!(s::TuckerState, bases)

For each fock space sector defined, add all possible basis states 
- `bases::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::TuckerState, bases::Vector{ClusterBasis}; nroots=1)
    # {{{
    println("\n Make each Fock-Block the full space")
    # create full space for each fock block defined
    for (fblock,configs) in s.data
        #println(fblock)
        dims = TuckerConfig()
        #display(fblock)
        for c in s.clusters
            # get number of vectors for current fock space
            dim = size(bases[c.idx][fblock[c.idx]], 2)
            push!(dims, 1:dim)
        end
       
        s.data[fblock][dims] = zeros((length.(dims)...,nroots))

    end
end
# }}}



"""
    function orthonormalize!(s::TuckerState{T,N,R}) where {T,N,R}

orthonormalize
"""
function orthonormalize!(s::TuckerState{T,N,R}) where {T,N,R}
    #={{{=#
    v0 = get_vector(s) 
    v0[:,1] .= v0[:,1]./norm(v0[:,1])
    for r in 2:R
        #|vr> = |vr> - |v1><v1|vr> - |v2><v2|vr> - ... 
        for r0 in 1:r-1 
            v0[:,r] .-= v0[:,r0] .* (v0[:,r0]'*v0[:,r])
        end
        v0[:,r] .= v0[:,r]./norm(v0[:,r])
    end
    isapprox(det(v0'*v0), 1.0, atol=1e-14) || @warn "initial guess det(v0'v0) = ", det(v0'v0) 
    set_vector!(s,v0)
end
#=}}}=#


