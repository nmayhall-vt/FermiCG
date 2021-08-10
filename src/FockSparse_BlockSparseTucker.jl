"""
Represents a state in an set of abitrary (yet low-rank) subspaces of a set of FockConfigs.
e.g. 
    
    v[FockConfig][TuckerConfig] => Tucker Decomposed Tensor

# Data
- `clusters::Vector{Cluster}`
- `data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}`
- `p_spaces::Vector{ClusterSubspace}`
- `q_spaces::Vector{ClusterSubspace}`
"""
struct CompressedTuckerState{T,N} 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N}}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
end
Base.haskey(ts::CompressedTuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::CompressedTuckerState, i) = return ts.data[i]
Base.setindex!(ts::CompressedTuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::CompressedTuckerState, state=1) = iterate(ts.data, state)
normalize!(ts::CompressedTuckerState) = scale!(ts, 1/sqrt(orth_dot(ts,ts)))


"""
    CompressedTuckerState(clusters::Vector{Cluster}, 
        fconfig::FockConfig{N}, 
        cluster_bases::Vector{ClusterBasis}) where {N} 

Constructor using only a single FockConfig. This allows us to turn the CMF state into a CompressedTuckerState.
As such, it chooses the ground state of each cluster in the Fock sector specified by `FockConfig` to be the 
P space, and then the Q space is defined as the orthogonal complement of this state within the available basis, 
specified by `cluster_bases`.
# Arguments
- `clusters`: vector of clusters types
- `fconfig`: starting FockConfig 
- `cluster_basis`: list of ClusterBasis types - needed to know the dimensions of the q-spaces
# Returns
- `CompressedTuckerState`
"""
function CompressedTuckerState(clusters::Vector{Cluster}, 
        fconfig::FockConfig{N}, 
        cluster_bases::Vector{ClusterBasis}; T=Float64) where {N} 
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

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N}} }()
    state = CompressedTuckerState(clusters, data, p_spaces, q_spaces) 
    add_fockconfig!(state, fconfig)

    factors = []
    for ci in clusters
        dim = length(p_spaces[ci.idx][fconfig[ci.idx]])
        push!(factors, 1.0Matrix(I, dim, 1))
    end
    factors = tuple(factors...) 
    
    tconfig = TuckerConfig([p_spaces[ci.idx].data[fconfig[ci.idx]] for ci in clusters])
    tdata = Tucker(reshape([1.0], tuple(ones(Int64, N)...)), factors)
    
    state[fconfig][tconfig] = tdata
    return state
#=}}}=#
end


"""
    CompressedTuckerState(clusters::Vector{Cluster}, 
        p_spaces::Vector{FermiCG.ClusterSubspace}, 
        q_spaces::Vector{FermiCG.ClusterSubspace}) 

Constructor - specify input p and q spaces
# Arguments
- `clusters`: vector of clusters types
- `p_spaces`: list of p space ranges for each cluster
- `q_spaces`: list of q space ranges for each cluster
# Returns
- `CompressedTuckerState`
"""
function CompressedTuckerState(clusters::Vector{Cluster}, 
        p_spaces::Vector{FermiCG.ClusterSubspace}, 
        q_spaces::Vector{FermiCG.ClusterSubspace}) 
    #={{{=#

    N = length(clusters)
    T = Float64

    #factors = []
    #for ci in clusters
    #    dim = length(p_spaces[ci.idx][init_fspace[ci.idx]])
    #    push!(factors, 1.0Matrix(I, dim, 1))
    #end
    #factors = tuple(factors...) 
    
    #tconfig = TuckerConfig([p_spaces[ci.idx].data[init_fspace[ci.idx]] for ci in clusters])
    #fconfig = FockConfig(init_fspace)
    #tdata = Tucker(reshape([1.0], tuple(ones(Int64, N)...)), factors)
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N}} }()
     
    #data[fconfig][tconfig] = tdata
    return CompressedTuckerState(clusters, data, p_spaces, q_spaces) 
#=}}}=#
end


"""
    CompressedTuckerState(ts::TuckerState; thresh=-1, max_number=nothing, verbose=0)

Create a `CompressedTuckerState` from a `TuckerState` 
# Arguments
- `ts::TuckerState`
- `thresh=-1`: discard singular values smaller than `thresh`
- `max_number=nothing`: if != `nothing`, only keep up to `max_number` singular vectors per SVD
- `verbose=0`: print level
# Returns 
- `CompressedTuckerState`
"""
function CompressedTuckerState(ts::TuckerState{T,N}; thresh=-1, max_number=nothing, verbose=0) where {T,N}
#={{{=#
    # make all AbstractState subtypes parametric
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

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N} }}()
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



"""
    compress(ts::CompressedTuckerState{T,N}; thresh=-1, max_number=nothing, verbose=0) where {T,N}

Compress state via HOSVD
# Arguments
- `ts::CompressedTuckerState`
- `thresh = -1`: threshold for compression
- `max_number`: only keep certain number of vectors per TuckerConfig
- `verbose=0`: print level
# Returns
- `CompressedTuckerState`
"""
function compress(ts::CompressedTuckerState{T,N}; thresh=-1, max_number=nothing, verbose=0) where {T,N}
    d = OrderedDict{FockConfig{N}, OrderedDict{TuckerConfig{N}, Tucker{T,N}}}() 
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            tmp = compress(ts.data[fock][tconfig], thresh=thresh, max_number=max_number)
            if length(tmp) == 0
                continue
            end
            if haskey(d, fock)
                d[fock][tconfig] = tmp
            else
                d[fock] = OrderedDict(tconfig => tmp)
            end
        end
    end
    return CompressedTuckerState(ts.clusters, d, ts.p_spaces, ts.q_spaces)
end


"""
    orth_add!(ts1::CompressedTuckerState, ts2::CompressedTuckerState)

Add coeffs in `ts2` to `ts1`

Note: this assumes `t1` and `t2` have the same compression vectors
"""
function orth_add!(ts1::CompressedTuckerState, ts2::CompressedTuckerState)
#={{{=#
    for (fock,configs) in ts2
        if haskey(ts1, fock)
            for (config,coeffs) in configs
                if haskey(ts1[fock], config)
                    ts1[fock][config].core .+= ts2[fock][config].core
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
    nonorth_add!(ts1::CompressedTuckerState, ts2::CompressedTuckerState)

Add coeffs in `ts2` to `ts1`

Note: this does not assume `t1` and `t2` have the same compression vectors
"""
function nonorth_add!(ts1::CompressedTuckerState, ts2::CompressedTuckerState)
#={{{=#
    for (fock,configs) in ts2
        if haskey(ts1, fock)
            for (config,coeffs) in configs
                if haskey(ts1[fock], config)
                    ts1[fock][config] = ts1[fock][config] + ts2[fock][config] # note this is non-trivial work here
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
    add_fockconfig!(s::CompressedTuckerState, fock::FockConfig)
"""
function add_fockconfig!(s::CompressedTuckerState{T,N}, fock::FockConfig) where {T,N}
    s.data[fock] = OrderedDict{TuckerConfig, Tucker{T,N}}()
end

"""
    Base.length(s::CompressedTuckerState)
"""
function Base.length(s::CompressedTuckerState)
    l = 0
    for (fock,tconfigs) in s.data
        for (tconfig, tuck) in tconfigs
            l += length(tuck)
        end
    end
    return l
end
"""
    prune_empty_fock_spaces!(s::AbstractState)

remove fock_spaces that don't have any configurations
"""
function prune_empty_fock_spaces!(s::AbstractState)
    focklist = keys(s.data)
    for fock in focklist
        if length(s.data[fock]) == 0
            delete!(s.data, fock)
        end
    end
    focklist = keys(s.data)
    for (fock,tconfigs) in s.data
        for (tconfig,coeff) in tconfigs
        end
    end
end
"""
    prune_empty_TuckerConfigs!(s::T) where T<:Union{TuckerState, CompressedTuckerState}

remove fock_spaces that don't have any configurations
"""
function prune_empty_TuckerConfigs!(s::T) where T<:Union{TuckerState, CompressedTuckerState}
    focklist = keys(s.data)
    for fock in focklist
        tconflist = keys(s.data[fock])
        for tconf in tconflist
            if length(s.data[fock][tconf]) == 0
                delete!(s.data[fock], tconf)
            end
        end
    end
    for (fock,tconfigs) in s.data
        for (tconfig,coeff) in tconfigs
        end
    end
    prune_empty_fock_spaces!(s)
end


"""
    get_vector(s::CompressedTuckerState)

Return a vector of the variables. Note that this is the core tensors being returned
"""
function get_vector(cts::CompressedTuckerState)

    v = zeros(length(cts), 1)
    idx = 1
    for (fock, tconfigs) in cts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck.core)

            dim1 = prod(dims)
            v[idx:idx+dim1-1,:] = copy(reshape(tuck.core,dim1))
            idx += dim1
        end
    end
    return v
end
"""
    set_vector!(s::CompressedTuckerState)
"""
function set_vector!(ts::CompressedTuckerState, v)

    #length(size(v)) == 1 || error(" Only takes vectors", size(v))
    nbasis = size(v)[1]

    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck)

            dim1 = prod(dims)
            ts[fock][tconfig].core .= reshape(v[idx:idx+dim1-1], size(tuck.core))
            idx += dim1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end
"""
    zero!(s::CompressedTuckerState)
"""
function zero!(s::CompressedTuckerState)
    for (fock, tconfigs) in s
        for (tconfig, tcoeffs) in tconfigs
            fill!(s[fock][tconfig].core, 0.0)
        end
    end
end

"""
    Base.display(s::CompressedTuckerState; thresh=1e-3)

Pretty print
"""
function Base.display(s::CompressedTuckerState; thresh=1e-3)
#={{{=#
    println()
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- # Fockspaces -------------------: %5i  \n",length(keys(s.data)))
    @printf(" ---------- # Configs    -------------------: %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-10s%-10s%-20s\n", "Weight", "# configs", "(full)", "(α,β)...")
    @printf(" %-20s%-10s%-10s%-20s\n", "-------","---------", "---------", "----------")
    for (fock,configs) in s.data
        prob = 0
        len = 0

        lenfull = 0
        for (config, tuck) in configs
            prob += sum(tuck.core .* tuck.core)
            len += length(tuck.core)
            lenfull += prod(dims_large(tuck))
        end
        if prob > thresh
        #if lenfull > 0
            #@printf(" %-20.3f%-10i%-10i", prob,len, lenfull)
            @printf(" %-20.3f%-10s%-10s", prob,"","")
            for sector in fock
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()

            #@printf("     %-16s%-20s%-20s\n", "Weight", "", "Subspaces")
            #@printf("     %-16s%-20s%-20s\n", "-------", "", "----------")
            for (config, tuck) in configs
                probi = sum(tuck.core .* tuck.core)
                @printf("     %-16.3f%-10i%-10i", probi,length(tuck.core),prod(dims_large(tuck)))
                for range in config
                    @printf("%7s", range)
                end
                println()
            end
            #println()
            @printf(" %-20s%-20s%-20s\n", "---------", "", "----------")
        end
    end
    print(" --------------------------------------------------\n")
    println()
#=}}}=#
end
"""
    print_fock_occupations(s::CompressedTuckerState; thresh=1e-3)

Pretty print
"""
function print_fock_occupations(s::CompressedTuckerState; thresh=1e-3)
#={{{=#

    println()
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- # Fockspaces -------------------: %5i  \n",length(keys(s.data)))
    @printf(" ---------- # Configs    -------------------: %5i  \n",length(s))
    @printf(" --------------------------------------------------\n")
    @printf(" Printing contributions greater than: %f", thresh)
    @printf("\n")
    @printf(" %-20s%-10s%-10s%-20s\n", "Weight", "# configs", "(full)", "(α,β)...")
    @printf(" %-20s%-10s%-10s%-20s\n", "-------","---------", "---------", "----------")
    for (fock,configs) in s.data
        prob = 0
        len = 0
        lenfull = 0
        for (config, tuck) in configs
            prob += sum(tuck.core .* tuck.core)
            len += length(tuck.core)
            lenfull += prod(dims_large(tuck))
        end
        if prob > thresh
            @printf(" %-20.3f%-10i%-10i", prob,len,lenfull)
            for sector in fock
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()
        end
    end
    print(" --------------------------------------------------\n")
    println()
#=}}}=#
end


"""
    dot(ts1::FermiCG.CompressedTuckerState, ts2::FermiCG.CompressedTuckerState)

Dot product between `ts2` and `ts1`

Warning: this assumes both `ts1` and `ts2` have the same tucker factors for each `TuckerConfig`
"""
function orth_dot(ts1::CompressedTuckerState, ts2::CompressedTuckerState)
#={{{=#
    overlap = 0.0
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            overlap += sum(ts1[fock][config].core .* ts2[fock][config].core)
        end
    end
    return overlap
#=}}}=#
end



"""
    nonorth_dot(ts1::FermiCG.CompressedTuckerState, ts2::FermiCG.CompressedTuckerState; verbose=0)

Dot product between 1ts2` and `ts1` where each have their own Tucker factors
"""
function nonorth_dot(ts1::CompressedTuckerState, ts2::CompressedTuckerState; verbose=0)
#={{{=#
    overlap = 0.0
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        verbose == 0 || display(fock)
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            verbose == 0 || display(config)
            overlap += dot(ts1[fock][config] , ts2[fock][config])
            verbose == 0 || display(dot(ts1[fock][config] , ts2[fock][config]))
        end
    end
    return overlap
#=}}}=#
end

"""
    scale!(ts::FermiCG.CompressedTuckerState, a::T<:Number)

Scale `ts` by a constant
"""
function scale!(ts::CompressedTuckerState, a::T) where T<:Number
    #={{{=#
    for (fock,configs) in ts
        for (config,tuck) in configs
            ts[fock][config].core .*= a
        end
    end
    #=}}}=#
end
