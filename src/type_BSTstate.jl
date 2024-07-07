"""
Represents a state in an set of abitrary (yet low-rank) subspaces of a set of FockConfigs.
e.g. 
    
    v[FockConfig][TuckerConfig] => Tucker Decomposed Tensor

# Data
- `clusters::Vector{MOCluster}`
- `data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}`
- `p_spaces::Vector{ClusterSubspace}`
- `q_spaces::Vector{ClusterSubspace}`
"""
struct BSTstate{T,N,R} 
    clusters::Vector{MOCluster}
    data::OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
end
Base.haskey(ts::BSTstate, i) = return haskey(ts.data,i)
Base.getindex(ts::BSTstate, i) = return ts.data[i]
Base.setindex!(ts::BSTstate, i, j) = return ts.data[j] = i
Base.iterate(ts::BSTstate, state=1) = iterate(ts.data, state)
Base.size(ts::BSTstate{T,N,R}) where {T,N,R} =  (length(ts), R)
#normalize!(ts::BSTstate) = scale!(ts, 1/sqrt(orth_dot(ts,ts)))


"""
    BSTstate(clusters::Vector{MOCluster}, 
             p_spaces::Vector{ClusterSubspace},
             cluster_bases::Vector{ClusterBasis}; R=1) where {N} 

Constructor needing only prespecified p-spaces. 
The Q space is defined as the orthogonal complement of p-space within the available basis, 
specified by `cluster_bases`.
# Arguments
- `clusters`: vector of clusters types
- `p_spaces`: list of pspaces for each cluster 
- `cluster_basis`: list of ClusterBasis types - needed to know the dimensions of the q-spaces
- `R`: number of roots
# Returns
- `BSTstate`
"""
function BSTstate(clusters::Vector{MOCluster}, 
                  p_spaces::Vector{ClusterSubspace},
                  cluster_bases::Vector{ClusterBasis{A,T}}; R=1) where {T,A} 
    #={{{=#

    N = length(clusters)
    # 
    # start by building the P and Q spaces needed
    q_spaces = Vector{ClusterSubspace}()

    # define q spaces
    for tssp in p_spaces 
        tss = get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}} }()

    return BSTstate{T,N,R}(clusters, data, p_spaces, q_spaces) 
#=}}}=#
end


"""
    BSTstate(clusters::Vector{MOCluster}, 
        fconfig::FockConfig{N}, 
        cluster_bases::Vector{ClusterBasis}) where {N} 

Constructor using only a single FockConfig. This allows us to turn the CMF state into a BSTstate.
As such, it chooses the ground state of each cluster in the Fock sector specified by `FockConfig` to be the 
P space, and then the Q space is defined as the orthogonal complement of this state within the available basis, 
specified by `cluster_bases`.
# Arguments
- `clusters`: vector of clusters types
- `fconfig`: starting FockConfig 
- `cluster_basis`: list of ClusterBasis types - needed to know the dimensions of the q-spaces
# Returns
- `BSTstate`
"""
function BSTstate(clusters::Vector{MOCluster}, 
        fconfig::FockConfig{N}, 
        cluster_bases::Vector{ClusterBasis{A, T}}; R=1) where {T, N, A} 
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

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}} }()
    state = BSTstate{T,N,R}(clusters, data, p_spaces, q_spaces) 
    
    # Replace below with "fill_p_space!"

    add_fockconfig!(state, fconfig)

    factors = []
    for ci in clusters
        dim = length(p_spaces[ci.idx][fconfig[ci.idx]])
        push!(factors, 1.0Matrix(I, dim, 1))
    end
    factors = tuple(factors...) 
    
    tconfig = TuckerConfig([p_spaces[ci.idx].data[fconfig[ci.idx]] for ci in clusters])
    core = tuple([reshape([1.0], tuple(ones(Int64, N)...)) for r in 1:R]...)
    state[fconfig][tconfig] = Tucker{T,N,R}(core, factors)
    return state
#=}}}=#
end


"""

Constructor - create copy, changing T and R optionally 
# Arguments
- `v`: input `BSTstate` object 
- `T`: data type of new state 
- `R`: number of roots in new state 
# Returns
- `BSTstate`
"""
function BSTstate(v::BSTstate{TT,N,RR}; T=TT, R=RR) where {TT,N,RR}
    #={{{=#

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}} }()
   
    w = BSTstate{T,N,R}(v.clusters, data, v.p_spaces, v.q_spaces)
    for (fock, tconfigs) in v.data
        add_fockconfig!(w, fock)
        for (tconfig, tuck) in tconfigs
            w[fock][tconfig] = Tucker(tuck, R=R, T=T)
        end
    end
    return w 
#=}}}=#
end
"""

Constructor - create copy, of a particular root
# Arguments
- `v`: input `BSTstate` object 
- `T`: data type of new state 
- `R`: specific root in new state 
# Returns
- `BSTstate`
"""
function BSTstate(v::BSTstate{T,N,R},  root) where {T,N,R}
    #={{{=#

    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,1}} }()
   
    w = BSTstate{T,N,1}(v.clusters, data, v.p_spaces, v.q_spaces)
    for (fock, tconfigs) in v.data
        add_fockconfig!(w, fock)
        for (tconfig, tuck) in tconfigs
            core=(tuck.core[root],)
            factors = ntuple(i->tuck.factors[i],N)
            w[fock][tconfig] = Tucker{T, N, 1}(core, factors)
        end
    end
    return w 
#=}}}=#
end
# return Tucker{T,NN,R}(cores, ntuple(i->t.factors[i],NN))

"""
    BSTstate(clusters::Vector{MOCluster}, 
        p_spaces::Vector{FermiCG.ClusterSubspace}, 
        q_spaces::Vector{FermiCG.ClusterSubspace}) 

Constructor - specify input p and q spaces
# Arguments
- `clusters`: vector of clusters types
- `p_spaces`: list of p space ranges for each cluster
- `q_spaces`: list of q space ranges for each cluster
# Returns
- `BSTstate`
"""
function BSTstate(clusters::Vector{MOCluster}, 
        p_spaces::Vector{FermiCG.ClusterSubspace}, 
        q_spaces::Vector{FermiCG.ClusterSubspace};
        T=Float64, R=1) 
    #={{{=#

    N = length(clusters)
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}} }()
     
    #data[fconfig][tconfig] = tdata
    return BSTstate(clusters, data, p_spaces, q_spaces) 
#=}}}=#
end


"""
    function BSTstate(ts::BSstate{T,N,R}; thresh=-1, max_number=nothing, verbose=0) where {T,N,R}

Create a `BSTstate` from a `BSstate` 
# Arguments
- `ts::BSstate`
- `thresh=-1`: discard singular values smaller than `thresh`
- `max_number=nothing`: if != `nothing`, only keep up to `max_number` singular vectors per SVD
- `verbose=0`: print level
# Returns 
- `BSTstate`
"""
function BSTstate(ts::BSstate{T,N,R}; thresh=-1, max_number=nothing, verbose=0) where {T,N,R}
#={{{=#

    fold!(ts)
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T, N, R} }}()
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs

            #
            # Since BSstate has extra dimension for state index, remove that
            #tuck = Tucker(reshape(coeffs,size(coeffs)[1:end]), thresh=thresh, max_number=max_number, verbose=verbose)
            #display([selectdim(coeffs, N+1, i) for i in 1:R])
            tuck = Tucker(Tuple([Array(selectdim(coeffs, N+1, i)) for i in 1:R]), thresh=thresh, max_number=max_number, verbose=verbose)
            if length(tuck) > 0
                if haskey(data, fock)
                    data[fock][tconfig] = tuck
                else
                    data[fock] = OrderedDict(tconfig => tuck)
                end
            end
        end
    end
    return BSTstate(ts.clusters, data, ts.p_spaces, ts.q_spaces)
end
#=}}}=#



"""
    compress(ts::BSTstate{T,N,R}; thresh=-1, max_number=nothing, verbose=0) where {T,N,R}

Compress state via HOSVD
# Arguments
- `ts::BSTstate`
- `thresh = -1`: threshold for compression
- `max_number`: only keep certain number of vectors per TuckerConfig
- `verbose=0`: print level
# Returns
- `BSTstate`
"""
function compress(ts::BSTstate{T,N,R}; thresh=-1, max_number=nothing, verbose=0) where {T,N,R}
    d = OrderedDict{FockConfig{N}, OrderedDict{TuckerConfig{N}, Tucker{T,N,R}}}() 
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
    return BSTstate(ts.clusters, d, ts.p_spaces, ts.q_spaces)
end

"""
    compress_iteratively(v::BSTstate, thresh; maxiter=20, verbose=1)

TBW
"""
function compress_iteratively(v::BSTstate, thresh; maxiter=20, verbose=1)
    for i in 1:maxiter
        dim1 = length(v)
        v = compress(v, thresh=thresh)
        dim2 = length(v)
        verbose < 1 || @printf("   Iter: %4i %12i → %12i\n",i, dim1, dim2)
        dim2 < dim1 || break
    end
    return v
end

"""
    orth_add!(ts1::BSTstate, ts2::BSTstate)

Add coeffs in `ts2` to `ts1`

Note: this assumes `t1` and `t2` have the same compression vectors
"""
function orth_add!(ts1::BSTstate, ts2::BSTstate)
#={{{=#
    for (fock,configs) in ts2
        if haskey(ts1, fock)
            for (config,coeffs) in configs
                if haskey(ts1[fock], config)
                    add!(ts1[fock][config].core, ts2[fock][config].core)
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
    nonorth_add!(ts1::BSTstate, ts2::BSTstate; thresh=1e-10)

Add coeffs in `ts2` to `ts1`

Note: this does not assume `t1` and `t2` have the same compression vectors
"""
function nonorth_add!(ts1::BSTstate, ts2::BSTstate; thresh=1e-10)
#={{{=#
    for (fock,configs) in ts2
        if haskey(ts1, fock)
            for (config,coeffs) in configs
                if haskey(ts1[fock], config)
                    #ts1[fock][config] = ts1[fock][config] + ts2[fock][config] # note this is non-trivial work here
                    ts1[fock][config] = nonorth_add(ts1[fock][config], ts2[fock][config])
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
    nonorth_add(ts1::BSTstate, ts2::BSTstate; thresh=1e-10)

TBW
"""
function nonorth_add(ts1::BSTstate, ts2::BSTstate; thresh=1e-10)
    tmp = deepcopy(ts1)
    nonorth_add!(tmp, ts2, thresh=1e-10)
    return tmp
end

"""
    orth_add(ts1::BSTstate, ts2::BSTstate)

TBW
"""
function orth_add(ts1::BSTstate, ts2::BSTstate)
    tmp = deepcopy(ts1)
    orth_add!(tmp, ts2)
    return tmp
end

"""
    add_fockconfig!(s::BSTstate, fock::FockConfig)

Add an uninitialized `FockConfig` to the current basis. Typically, you'll want to 
initialize it right after with `add_tuckerconfig!`
"""
function add_fockconfig!(s::BSTstate{T,N,R}, fock::FockConfig) where {T,N,R}
    s.data[fock] = OrderedDict{TuckerConfig, Tucker{T,N,R}}()
end

function fill_p_space!(s::BSTstate{T,N,R}, na, nb) where {T,N,R}

    sectors = [] 
    for ci in s.clusters
        sectors_i = []
        for (fock, range) in s.p_spaces[ci.idx].data
            push!(sectors_i, fock)
        end
        push!(sectors, sectors_i)
    end
    for fconfig in Iterators.product(sectors...)
        fi = FockConfig([fconfig...])
        if n_elec_a(fi) == na && n_elec_b(fi) == nb 
            add_fockconfig!(s, fi)
            
            factors = []
            dims = []
            for ci in s.clusters
                dim = length(s.p_spaces[ci.idx][fi[ci.idx]])
                push!(factors, Matrix(1.0I, dim, dim))
                push!(dims, dim)
            end
            factors = tuple(factors...) 

            tconfig = TuckerConfig([s.p_spaces[ci.idx].data[fi[ci.idx]] for ci in s.clusters])
            core = tuple([reshape(ones(prod(dims)), tuple(dims...)) for r in 1:R]...)
            s[fi][tconfig] = Tucker{T,N,R}(core, factors)
        end
    end
end


"""
    Base.length(s::BSTstate)
"""
function Base.length(s::BSTstate)
    l = 0
    for (fock,tconfigs) in s.data
        for (tconfig, tuck) in tconfigs
            l += length(tuck)
        end
    end
    return l
end

"""
    eye!(s::BSTstate)
"""
function eye!(s::BSTstate{T,N,R}) where {T,N,R}
    set_vector!(s, Matrix{T}(I,size(s)))
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
    prune_empty_TuckerConfigs!(s::T) where T<:Union{BSstate, BSTstate}

remove fock_spaces that don't have any configurations
"""
function prune_empty_TuckerConfigs!(s::T) where T<:Union{BSstate, BSTstate}
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
    function orthonormalize!(s::BSTstate{T,N,R}) where {T,N,R}

orthonormalize
"""
function orthonormalize!(s::BSTstate{T,N,R}) where {T,N,R}
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



"""
    function get_vector(ts::BSTstate{T,N,R}) where {T,N,R}

Return a matrix of the core tensors.
"""
function get_vector(ts::BSTstate{T,N,R}) where {T,N,R}
#={{{=#
    v = zeros(length(ts), R)
    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck)

            dim1 = prod(dims)
            for r in 1:R
                v[idx:idx+dim1-1,r] .= copy(reshape(tuck.core[r],dim1))
            end
            idx += dim1
        end
    end
    return v
end
#=}}}=#


"""
    function get_vector(ts::BSTstate{T,N,R}, root::Integer) where {T,N,R}

Return a vector of the variables for `root`. Note that this is the core tensors being returned
"""
function get_vector(ts::BSTstate{T,N,R}, root::Integer) where {T,N,R}
#={{{=#
    v = zeros(length(ts), 1)
    root <= R || throw(DimensionMismatch)
    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck.core[root])

            dim1 = prod(dims)
            v[idx:idx+dim1-1,:] = copy(reshape(tuck.core[root],dim1))
            idx += dim1
        end
    end
    return v
end
#=}}}=#


"""
    function set_vector!(ts::BSTstate{T,N,R}, v::Vector{T}, root::Int=1) where {T,N,R}
"""
function set_vector!(ts::BSTstate{T,N,R}, v::Vector{T}; root=1) where {T,N,R}
#={{{=#
    #length(size(v)) == 1 || error(" Only takes vectors", size(v))
    nbasis = size(v)[1]

    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck)

            dim1 = prod(dims)
            ts[fock][tconfig].core[root] .= reshape(v[idx:idx+dim1-1], size(tuck.core[1]))
            idx += dim1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end
#=}}}=#


"""
    function set_vector!(ts::BSTstate{T,N,R}, v::Matrix{T}) where {T,N,R}
"""
function set_vector!(ts::BSTstate{T,N,R}, v::Matrix{T}) where {T,N,R}
#={{{=#
    #length(size(v)) == 1 || error(" Only takes vectors", size(v))
    nbasis = size(v)[1]
    
    R == size(v,2) || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tuck)

            dim1 = prod(dims)
            for r in 1:R
                ts[fock][tconfig].core[r] .= reshape(v[idx:idx+dim1-1,r], size(tuck.core[r]))
            end
            idx += dim1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end
#=}}}=#


"""
"""
function randomize!(s::BSTstate{T,N,R}; seed=nothing) where {T,N,R}
#={{{=#
    for (fock, tconfigs) in s
        for (tconfig, tcoeffs) in tconfigs
            randomize!(s[fock][tconfig], seed=seed)
        end
    end
end
#=}}}=#

"""
    zero!(s::BSTstate)
"""
function zero!(s::BSTstate{T,N,R}) where {T,N,R}
#={{{=#
    for (fock, tconfigs) in s
        for (tconfig, tcoeffs) in tconfigs
            for r in 1:R
                fill!(s[fock][tconfig].core[r], zero(T))
            end
        end
    end
end
#=}}}=#

"""
    Base.display(s::BSTstate; thresh=1e-3)

Pretty print
"""
function Base.display(s::BSTstate; thresh=1e-2, root=1)
#={{{=#
    println()
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- # Fockspaces -------------------: %5i  \n",length(keys(s.data)))
    @printf(" ---------- # Configs    -------------------: %5i  \n",length(s))
    @printf(" ---------- Root ---------------------------: %5i  \n",root)
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
            prob += sum(tuck.core[root] .* tuck.core[root])
            len += length(tuck.core[root])
            lenfull += prod(dims_large(tuck))
        end
        prob = sqrt(prob)
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
                probi = sqrt(sum(tuck.core[root] .* tuck.core[root]))
                if probi > thresh
                    @printf("     %-16.3f%-10i%-10i", probi,length(tuck.core[root]),prod(dims_large(tuck)))
                    for range in config
                        @printf("%7s", range)
                    end
                    println()
                end
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
    print_fock_occupations(s::BSTstate; thresh=1e-3)

Pretty print
"""
function print_fock_occupations(s::BSTstate; thresh=1e-3,root=1)
#={{{=#

    println()
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- # Fockspaces -------------------: %5i  \n",length(keys(s.data)))
    @printf(" ---------- # Configs    -------------------: %5i  \n",length(s))
    @printf(" ---------- Root ---------------------------: %5i  \n",root)
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
            prob += sum(tuck.core[root] .* tuck.core[root])
            len += length(tuck.core[root])
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
    orth_overlap(ts1::FermiCG.BSTstate, ts2::FermiCG.BSTstate)

Overlap between `ts2` and `ts1`

This assumes both `ts1` and `ts2` have the same tucker factors for each `TuckerConfig`
Returns matrix of overlaps
"""
function orth_overlap(ts1::BSTstate{T,N,R}, ts2::BSTstate{T,N,R}) where {T,N,R}
    #={{{=#
    overlap = zeros(T,R,R) 
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            for ri in 1:R
                for rj in ri:R
                    overlap[ri,rj] += sum(ts1[fock][config].core[ri] .* ts2[fock][config].core[rj])
                    overlap[rj,ri] = overlap[ri,rj]
                end
            end
        end
    end
    return overlap
    #=}}}=#
end


"""
    orth_dot(ts1::FermiCG.BSTstate, ts2::FermiCG.BSTstate)

Dot product between `ts2` and `ts1`

Warning: this assumes both `ts1` and `ts2` have the same tucker factors for each `TuckerConfig`
Returns vector of dot products
"""
function orth_dot(ts1::BSTstate{T,N,R}, ts2::BSTstate{T,N,R}) where {T,N,R}
    #={{{=#
    overlap = zeros(T,R) 
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            for r in 1:R
                overlap[r] += sum(ts1[fock][config].core[r] .* ts2[fock][config].core[r])
            end
        end
    end
    return overlap
    #=}}}=#
end



"""
    nonorth_overlap(ts1::FermiCG.BSTstate, ts2::FermiCG.BSTstate; verbose=0)

Dot product between 1ts2` and `ts1` where each have their own Tucker factors
"""
function nonorth_overlap(ts1::BSTstate{T,N,R}, ts2::BSTstate{T,N,R}; verbose=0) where {T,N,R}
    #={{{=#
    overlap = zeros(T,R,R)
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        verbose == 0 || display(fock)
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            verbose == 0 || display(config)
            overlap .+= nonorth_overlap(ts1[fock][config] , ts2[fock][config])
            verbose == 0 || display(dot(ts1[fock][config] , ts2[fock][config]))
        end
    end
    return overlap
    #=}}}=#
end



"""
    nonorth_dot(ts1::FermiCG.BSTstate, ts2::FermiCG.BSTstate; verbose=0)

Dot product between 1ts2` and `ts1` where each have their own Tucker factors
"""
function nonorth_dot(ts1::BSTstate{T,N,R}, ts2::BSTstate{T,N,R}; verbose=0) where {T,N,R}
    #={{{=#
    overlap = zeros(T,R)
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        verbose == 0 || display(fock)
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            verbose == 0 || display(config)
            overlap .+= nonorth_dot(ts1[fock][config] , ts2[fock][config])
            verbose == 0 || display(dot(ts1[fock][config] , ts2[fock][config]))
        end
    end
    return overlap
    #=}}}=#
end

"""
    scale!(ts::FermiCG.BSTstate, a::T<:Number)

Scale `ts` by a constant
"""
function scale!(ts::BSTstate{T,N,R}, a::T) where {T<:Number, N,R}
    #={{{=#
    for (fock,configs) in ts
        for (config,tuck) in configs
            for r in 1:R
                ts[fock][config].core[r] .*= a
            end
        end
    end
    #=}}}=#
end

function scale(ts::BSTstate{T,N,R}, a::T) where {T<:Number, N,R}
    tmp = deepcopy(ts)
    scale!(tmp, a)
    return tmp
end

"""
    function scale!(ts::FermiCG.BSTstate{T,N,R}, a::Vector{T})

Scale `ts` by a constant for each state,`R`
"""
function scale!(ts::BSTstate{T,N,R}, a::Vector{T}) where {T<:Number, N,R}
    #={{{=#
    length(a) == R || throw(DimensionMismatch)   
    for (fock,configs) in ts
        for (config,tuck) in configs
            for r in 1:R
                ts[fock][config].core[r] .*= a[r] 
            end
        end
    end
    #=}}}=#
end

"""
    scale(ts::BSTstate{T,N,R}, a::Vector{T}) where {T<:Number, N,R}

TBW
"""
function scale(ts::BSTstate{T,N,R}, a::Vector{T}) where {T<:Number, N,R}
    tmp = deepcopy(ts)
    scale!(tmp, a)
    return tmp
end

nroots(v::BSTstate{T,N,R}) where {T,N,R} = R
type(v::BSTstate{T,N,R}) where {T,N,R} = T
nclusters(v::BSTstate{T,N,R}) where {T,N,R} = N




"""
    add_spin_focksectors(state::BSTstate{T,N,R}) where {T,N,R}

Add the focksectors needed to spin adapt the given `BSTstate`
"""
function add_spin_focksectors(state::BSTstate{T,N,R}) where {T,N,R}
    out = deepcopy(state)
    for (fock, configs) in state.data
        for f in possible_spin_focksectors(state.clusters, fock)
            add_fockconfig!(out, f)
        end
    end
    #expand_each_fock_space!(state)
    return out
end



"""
    Base.:*(A::BSTstate{T,N,R}, C::T) where {T,N,R}

TBW
"""
function Base.:*(A::BSTstate{T,N,R}, C::T) where {T,N,R}
    B = deepcopy(A)
    scale!(B, C)
    return B
end

"""
    Base.:-(A::BSTstate{T,N,R}, B::BSTstate{T,N,R}) where {T,N,R}

TBW
"""
function Base.:-(A::BSTstate{T,N,R}, B::BSTstate{T,N,R}) where {T,N,R}
    length(A) == length(B) || throw(DimensionMismatch)
    C = deepcopy(A)
    set_vector!(C, get_vector(A) .- get_vector(B))
    return C
end


"""
    Base.:+(A::BSTstate{T,N,R}, B::BSTstate{T,N,R}) where {T,N,R}

TBW
"""
function Base.:+(A::BSTstate{T,N,R}, B::BSTstate{T,N,R}) where {T,N,R}
    length(A) == length(B) || throw(DimensionMismatch)
    C = deepcopy(A)
    set_vector!(C, get_vector(A) .+ get_vector(B))
    return C
end

"""
    Base.:*(A::BSTstate{T,N,R}, C::AbstractArray) where {T,N,R}

TBW
"""
function Base.:*(A::BSTstate{T,N,R}, C::AbstractArray) where {T,N,R}
    B = deepcopy(A)
    zero!(B)
    set_vector!(B, get_vector(A)*C)
    return B
end



"""
    project_into_new_basis(v1::BSTstate{T,N,R}, v2::BSTstate{T,N,R}) where {T,N,R}

Project state `v1`  into the basis defined by `v2`
"""
function project_into_new_basis(v1::BSTstate{T,N,R}, v2::BSTstate{T,N,R}) where {T,N,R}
    #
    flush(stdout)
    out = deepcopy(v2)
    zero!(out)
    for (fock, tconfigs) in v2 
        haskey(v1, fock) || continue
        for (tconfig, tuck) in tconfigs
            haskey(v1[fock], tconfig) || continue
            ref_tuck = v1[fock][tconfig]
            
            # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
            # Ux(Ii') Ux(Jj') ...
            #
            # Cr(i,j,k...) S(ii') S(jj')...
            overlaps = Vector{Matrix{T}}()
            for i in 1:N
                push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
            end
            for r in 1:R
                out[fock][tconfig].core[r] .= transform_basis(ref_tuck.core[r], overlaps)
            end
        end
    end
    return out
end

"""
    ct_table(s::BSTstate; ne_cluster=10, nroots=1)

Prints total weight of charge transfer in each root in table formate
# Arguments
- `s::BSTstate`
- `ne_cluster`:  Int, number of total electrons in each cluster
- `nroots`: Total number of roots
"""
function ct_table(s::BSTstate{T,N,R}; ne_cluster=10, nroots=1) where {T,N,R}
    @printf(" -----------------------\n")
    @printf(" --- CHARGE TRANSFER ---\n")
    @printf(" -----------------------\n")
    @printf(" %-15s%-10s\n", "Root", "Total CT")
    @printf(" %-15s%-10s\n", "-------", "---------")
    for root in 1:nroots
        ct = 0
        for (fock,configs) in s.data
            # println(fock)
            prob = 0
            is_ct = false

            for cluster in 1:length(s.clusters)
                if sum(fock[cluster]) != ne_cluster
                    is_ct = true
                end
            end
            if is_ct 
                prob = 0
                for (tconfig, tucker) in configs 
                    # println(tucker.core)
                    # println(tucker.factors)
                    prob += sum(tucker.core[root] .^ 2)
                end
            end
            ct += prob
        end
        @printf(" %-15i%-10.5f", root, ct)
        println()
    end
end
