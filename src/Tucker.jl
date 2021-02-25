using Profile
using LinearMaps
using BenchmarkTools
#using KrylovKit
using IterativeSolvers


"""
Defines a single cluster's subspace for Tucker. Each focksector is allowed to have a distinct cluster state range 
for the subspace.

    cluster::Cluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
"""
struct ClusterSubspace
    cluster::Cluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
end
function ClusterSubspace(cluster::Cluster)
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

"""
    get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)

For a given `ClusterSubspace`, `tss`, return the subspace remaining
"""
function get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)
#={{{=#
    data = OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}()
    for (fock,basis) in cb
    
        if haskey(tss.data,fock)
            first(tss.data[fock]) == 1 || error(" p-space doesn't include ground state?")
            newrange = last(tss[fock])+1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        else
            newrange = 1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        end
    end

    return ClusterSubspace(tss.cluster, data)
#=}}}=#
end


"""
    TuckerConfig()
"""
TuckerConfig() = TuckerConfig([])

"""
Check if `tc1` is a subset of `tc2`
"""
function is_subset(tc1::TuckerConfig, tc2::TuckerConfig)
    length(tc1.config) == length(tc2.config) || return false
    for i in 1:length(tc1)
        if first(tc1[i]) < first(tc2[i]) || last(tc1[i]) > last(tc2[i])
            return false
        end
    end
    return true
end

"""
    dim(tc::TuckerConfig)

Return total dimension of space indexed by `tc`
"""
dim(tc::TuckerConfig) = prod(size(tc)) 


"""
Represents a state in an set of abitrary (yet dense) subspaces of a set of FockConfigs.

v[FockConfig][TuckerConfig] => Dense Tensor
E.g., used in n-body Tucker
    
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
"""
struct TuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
end
Base.haskey(ts::TuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::TuckerState, i) = return ts.data[i]
Base.setindex!(ts::TuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::TuckerState, state=1) = iterate(ts.data, state)


"""
    TuckerState(clusters, p_spaces, q_spaces, foi; nroots=1)

Constructor
- `clusters::Vector{Cluster}`
- `p_spaces::Vector{ClusterSubspace}`
- `q_spaces::Vector{ClusterSubspace}`
- `na::Int` Number of alpha
- `nb::Int` Number of beta
"""
function TuckerState(clusters::Vector{Cluster}, p_spaces::Vector{ClusterSubspace}, q_spaces::Vector{ClusterSubspace}, na, nb; nroots=1)
   #={{{=#
    length(p_spaces) == length(clusters) || error("# of clusters don't match # of subspaces")
    length(p_spaces) == length(q_spaces) || error(" p_spaces/q_spaces don't have same length", length(p_space), length(q_space))
    for ci in clusters
        for fock in p_spaces[ci.idx].data
            if haskey(q_spaces[ci.idx].data, fock)
                all(collect(p_spaces[ci.idx][fock]) .!= collect(q_spaces[ci.idx][fock])) || error(" Not orthogonal")
            end
        end
    end

    #s = TuckerState(clusters)
    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}()
    ns = []
    for cssi in p_spaces 
        nsi = []
        for (fock,range) in cssi.data
            push!(nsi,fock)
        end
        push!(ns,nsi)
    end

    for newfock in product(ns...)
        nacurr = 0
        nbcurr = 0
        for c in newfock
            nacurr += c[1]
            nbcurr += c[2]
        end
        if (nacurr == na) && (nbcurr == nb)

            fockconfig = FockConfig(collect(newfock))

            tmp = []
            for ci in clusters
                cssi = p_spaces[ci.idx]
                push!(tmp, cssi.data[newfock[ci.idx]])
            end
            tuckconfig = TuckerConfig(tmp)

            haskey(data, fockconfig) == false || error(" here:", fockconfig)
            data[fockconfig] = OrderedDict(tuckconfig => zeros((size(tuckconfig)...,nroots)))
            #add_fockconfig!(data,fockconfig) 
            #data[fockconfig][tuckconfig] = zeros((size(tuckconfig)...,nroots))  # todo - finish this
        end
    end
    return TuckerState(clusters, data, p_spaces, q_spaces) 
#=}}}=#
end

"""
    TuckerState(clusters, p_spaces, q_spaces, foi; nroots=1)

Constructor to build state directly from a definition of a first order interacting space (or more generic even i suppose)
- `clusters::Vector{Cluster}`
- `p_spaces::Vector{ClusterSubspace}`
- `q_spaces::Vector{ClusterSubspace}`
- `foi::OrderedDict{FockConfig,Vector{TuckerConfig}}` 
- `nroots`
"""
function TuckerState(clusters::Vector{Cluster}, p_spaces::Vector{ClusterSubspace}, q_spaces::Vector{ClusterSubspace}, 
        foi::OrderedDict{FockConfig,Vector{TuckerConfig}}; nroots=1)
   #={{{=#
    length(p_spaces) == length(clusters) || error("# of clusters don't match # of subspaces")
    length(p_spaces) == length(q_spaces) || error(" p_spaces/q_spaces don't have same length", length(p_space), length(q_space))
    for ci in clusters
        for fock in p_spaces[ci.idx].data
            if haskey(q_spaces[ci.idx].data, fock)
                all(collect(p_spaces[ci.idx][fock]) .!= collect(q_spaces[ci.idx][fock])) || error(" Not orthogonal")
            end
        end
    end

    #s = TuckerState(clusters)
    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}()
    for (fock,tconfig_list) in foi
        data2 = OrderedDict{TuckerConfig,Array}()
        for tconfig in tconfig_list
            data2[tconfig] = zeros((size(tconfig)...,nroots))
        end

        data[fock] = data2 
    end
    return TuckerState(clusters, data, p_spaces, q_spaces) 
#=}}}=#
end
"""
    +(ts1::FermiCG.TuckerState, ts2::FermiCG.TuckerState)
"""
function Base.:+(ts0::TuckerState, ts2::TuckerState)
#={{{=#
    ts1 = deepcopy(ts0)
    for (fock,configs) in ts2
        if haskey(ts1, fock)
            for (config,coeffs) in configs 
                if haskey(ts1[fock], config)
                    ts1[fock][config] .+= ts2[fock][config]
                else
                    ts1[fock][config] = ts2[fock][config]
                end
            end
        else
            ts1[fock] = ts2[fock]
        end
    end
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
                    ts1[fock][config] = ts2[fock][config]
                end
            end
        else
            ts1[fock] = ts2[fock]
        end
    end
#=}}}=#
end
"""
    dot(ts1::FermiCG.TuckerState, ts2::FermiCG.TuckerState)

Dot product between in `ts2` to `ts1`
"""
function dot(ts1::TuckerState, ts2::TuckerState)
#={{{=#
   
    unfold!(ts1)
    unfold!(ts2)
    nroots1 = nothing 
    for (fock,configs) in ts1
        for (config,coeffs) in configs
            if nroots1 == nothing
                nroots1 = last(size(coeffs))
            else
                nroots1 == last(size(coeffs)) || error(" mismatch in number of roots")
            end
        end
    end
    nroots2 = nothing 
    for (fock,configs) in ts2
        for (config,coeffs) in configs
            if nroots2 == nothing
                nroots2 = last(size(coeffs))
            else
                nroots2 == last(size(coeffs)) || error(" mismatch in number of roots")
            end
        end
    end
   
    overlap = zeros(nroots1,nroots2) 
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
    scale!(ts::FermiCG.TuckerState, a)

Scale `ts` by a constant `a`
"""
function scale!(ts::TuckerState, a)
    #={{{=#
    for (fock,configs) in ts
        for (config,tuck) in configs 
            ts[fock][config] .*= a 
        end
    end
    #=}}}=#
end


"""
    unfold!(ts::TuckerState)
"""
function unfold!(ts::TuckerState)
#={{{=#
    for (fock,configs) in ts.data
        #display(fock)
        for (config,coeffs) in configs 
            #length(size(coeffs)) == length(ts.clusters)+1 || error(" Can only unfold a folded vector")
            nroots = last(size(coeffs)) 
           
            #display(config)
            #display(size(coeffs))
            #display(size(config))
            #display(nroots)
        
            #display(fock)
            #display(config)
            ts[fock][config] = reshape(coeffs, (prod(size(config)),nroots))
        end
    end
#=}}}=#
end
"""
    fold!(ts::TuckerState)
"""
function fold!(ts::TuckerState)
#={{{=#
    for (fock,configs) in ts.data
        for (config,coeffs) in configs 
            #length(size(coeffs)) == 2 || error(" Can only fold an unfolded vector")
            nroots = last(size(coeffs)) 
            
            ts[fock][config] = reshape(coeffs, (size(config)...,nroots))
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
    add_fockconfig!(s::ClusteredState, fock::FockConfig)
"""
function add_fockconfig!(s::TuckerState, fock::FockConfig)
    s.data[fock] = OrderedDict{TuckerConfig, Array}()
    #s.data[fock] = OrderedDict{TuckerConfig, Array}(TuckerConfig([1:1 for i in 1:length(s.clusters)])=>[0.0])
end

"""
    Base.length(s::TuckerState)
"""
function Base.length(s::TuckerState)
    l = 0
    for (fock,configs) in s.data 
        for (config,vector) in configs
            l += dim(config) 
        end
    end
    return l
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
    get_vector(s::TuckerState)
"""
function get_vector(ts::TuckerState)
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
    set_vector!(s::TuckerState)
"""
function set_vector!(ts::TuckerState, v)

    length(size(v)) <= 2 || error(" Only takes matrices", size(v))
    nbasis = size(v)[1]
    nroots = 0
    if length(size(v)) == 1
        nroots = 1
    elseif length(size(v)) == 2
        nroots = size(v)[2] 
    end

    #println(length(ts), nroots)
    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, coeffs) in tconfigs
            dims = size(tconfig)
            
            dim1 = prod(dims)
            #display(size(v[idx:idx+dim1-1,:]))
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
function eye!(s::TuckerState)
    idx1 = 1
    idx2 = 1
    for (fock, configs) in s
        for (config, coeffs) in configs
            for config in product(config.config)
                coeffs[config,idx] = 1
                idx += 1
            end
        end
    end
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
                @printf("     %-16.3f%-20i", probi,length(coeffs))
                for range in config 
                    @printf("%4s", range)
                end
                println()
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



