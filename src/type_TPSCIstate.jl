using StaticArrays
using LinearAlgebra
# using BenchmarkTools

"""
    clusters::Vector{MOCluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, Vector{T}}}

This represents an arbitrarily sparse state. E.g., used in TPSCI
"""
struct TPSCIstate{T,N,R} <: AbstractState 
    clusters::Vector{MOCluster}
    data::OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, MVector{R,T}}}
end
Base.haskey(ts::TPSCIstate, i) = return haskey(ts.data,i)
#Base.iterate(ts::TPSCIstate, state=1) = iterate(ts.data, state)
#Base.eltype(::Type{TPSCIstate{T,N,R}}) where {T,N,R} = OrderedDict{ClusterConfig{N}, MVector{R,T}} 

"""
    TPSCIstate(clusters; T=Float64, R=1)

Constructor creating an empty vector
# Arguments
- `clusters::Vector{MOCluster}`
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TPSCIstate`
"""
function TPSCIstate(clusters; T=Float64, R=1)
    N = length(clusters)
    return TPSCIstate{T,N,R}(clusters,OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, MVector{R,T}}}())
end

"""
    function TPSCIstate(v::TPSCIstate{T,N,R}; T=T, R=R) where {T,N,R}

Constructor creating a `TPSCIstate` with the same basis as `v`, but with potentially different `R` and `T`. 
Coefficients of new vector are 0.0

# Arguments
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TPSCIstate`
"""
function TPSCIstate(v::TPSCIstate{TT,NN,RR}; T=TT, R=RR) where {TT,NN,RR}
    out = TPSCIstate(v.clusters,T=T,R=R)
    for (fock, configs) in v.data
        add_fockconfig!(out,fock)
        for (config, coeffs) in configs
            out[fock][config] = zeros(T,R)
        end
    end
    return out
end

"""
    TPSCIstate(clusters::Vector{MOCluster}, fconfig::FockConfig{N}; T=Float64, R=1) where {N}

Constructor using only a single FockConfig. This allows us to turn the CMF state into a TPSCIstate.
# Arguments
- `clusters`: vector of clusters types
- `fconfig`: starting FockConfig
- `T`:  Type of data for coefficients
- `R`:  Number of roots
# Returns
- `TPSCIstate`
"""
function TPSCIstate(clusters::Vector{MOCluster}, fconfig::FockConfig{N}; T=Float64, R=1) where {N}
    #={{{=#

    state = TPSCIstate(clusters, T=T, R=R)
    add_fockconfig!(state, fconfig)
    conf = ClusterConfig([1 for i in 1:length(clusters)])
    state[fconfig][conf] = zeros(T,R) 
    return state
#=}}}=#
end









"""
    add_fockconfig!(s::TPSCIstate, fock::FockConfig)
"""
function add_fockconfig!(s::TPSCIstate{T,N,R}, fock::FockConfig{N}) where {T<:Number,N,R}
    s.data[fock] = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    #s.data[fock] = OrderedDict{ClusterConfig{N}, MVector{R,T}}(ClusterConfig([1 for i in 1:N]) => zeros(MVector{R,T}))
end

"""
    getindex(s::TPSCIstate, fock::Vector{Tuple{T,T}}) where T<:Integer
"""
#Base.getindex(s::TPSCIstate, fock::Vector{Tuple{T,T}}) where T<:Integer = s.data[fock]
@inline Base.getindex(s::TPSCIstate, fock) = s.data[fock]
@inline Base.setindex!(s::TPSCIstate, a, b) = s.data[b] = a


function Base.size(s::TPSCIstate{T,N,R}) where {T,N,R}
    return length(s),R
end
function Base.length(s::TPSCIstate)
    l = 0
    for (fock,configs) in s.data 
        l += length(keys(configs))
    end
    return l
end

#remove
"""
    get_vector(s::TPSCIstate, root=1)

Return a vector of the variables for `root`. Note that this is the core tensors being returned
"""
function get_vector(s::TPSCIstate{T,N,R}, root) where {T,N,R}
    root <= R || throw(DimensionMismatch) 
    v = zeros(length(s))
    idx = 1
    for (fock, configs) in s.data
        for (config, coeff) in configs
            v[idx] = coeff[root]
            idx += 1
        end
    end
    return v
end
"""
    get_vector(s::TPSCIstate)

Return a matrix of the variables for `root`. 
"""
function get_vector(s::TPSCIstate{T,N,R}) where {T,N,R}
    v = zeros(T,length(s), R)
    idx = 1
    for (fock, configs) in s.data
        for (config, coeff) in configs
            v[idx,:] .= coeff[:]
            idx += 1
        end
    end
    return v
end

"""
    get_vector!(v, s::TPSCIstate)

Fill a preallocated array with the coefficients
"""
function get_vector!(v, s::TPSCIstate{T,N,R}) where {T,N,R}
    idx = 1
    for (fock, configs) in s.data
        for (config, coeff) in configs
            v[idx,:] .= coeff[:]
            idx += 1
        end
    end
    return
end

"""
    function set_vector!(ts::TPSCIstate{T,N,R}, v::Matrix{T}) where {T,N,R}

Fill the coefficients of `ts` with the values in `v`
"""
function set_vector!(ts::TPSCIstate{T,N,R}, v::Matrix{T}) where {T,N,R}

    nbasis = size(v,1)
    nroots = size(v,2)

    length(ts) == nbasis || throw(DimensionMismatch)
    R == nroots || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            #ts[fock][tconfig] = MVector{R}(v[idx,:])
            @views coeffs .= v[idx,:]
            idx += 1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end

"""
    function set_vector!(ts::TPSCIstate{T,N,R}, v::Vector{T}; root=1) where {T,N,R}

Fill the coefficients of `ts` with the values in `v`
"""
function set_vector!(ts::TPSCIstate{T,N,R}, v::Vector{T}; root=1) where {T,N,R}

    nbasis=length(v)
    length(ts) == length(v) || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            coeffs[root] = v[idx]
            idx += 1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end

function set_vector!(ts::TPSCIstate{T,N,R}, v::Vector{Vector{T}}) where {T,N,R}
    nroots = length(v)
    nbasis = length(v[1])

    length(ts) == nbasis || throw(DimensionMismatch)
    R == nroots || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            @views coeffs .= [vi[idx] for vi in v] 
            idx += 1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end

#"""
#    function set_vector!(ts::TPSCIstate{T,N,R}, v) where {T,N,R}
#
#Fill the coefficients of `ts` with the values in `v`
#"""
#function set_vector!(ts::TPSCIstate{T,N,R}, v) where {T,N,R}
#
#    nbasis = size(v,1)
#    nroots = size(v,2)
#
#    length(ts) == nbasis || throw(DimensionMismatch)
#    R == nroots || throw(DimensionMismatch)
#
#    idx = 1
#    for (fock, tconfigs) in ts.data
#        for (tconfig, coeffs) in tconfigs
#            #ts[fock][tconfig] = MVector{R}(v[idx,:])
#            @views coeffs .= v[idx,:]
#            idx += 1
#        end
#    end
#    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
#    return
#end


"""
    Base.display(s::TPSCIstate; thresh=1e-3, root=1)

Pretty print
"""
function Base.display(s::TPSCIstate; thresh=1e-3, root=1)
    @printf(" --------------------------------------------------\n")
    @printf(" ---------- Fockspaces in state ------: Dim = %5i  \n",length(s))
    @printf(" ----------                root ------:     = %5i  \n",root)
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
function print_configs(s::TPSCIstate; thresh=1e-3, root=1)
    #display(keys(s.data))
    idx = 1
    for (fock,configs) in s.data
        length(s.clusters) == length(fock) || throw(Exception)
        length(s.data[fock]) > 0 || continue
        print_fock = false
        for (config, value) in s.data[fock]
            if value[root]*value[root] > thresh 
                print_fock = true
                break
            end
        end
        print_fock || continue

        @printf(" Dim %4i fock_space: ",length(s.data[fock]))
        [@printf(" %-2i(%i:%i) ",fii,fi[1],fi[2]) for (fii,fi) in enumerate(fock)] 
        println()
        for (config, value) in s.data[fock]
            value[root]*value[root] > thresh || continue
            #@printf(" %5i",idx)
            for c in config
                @printf("%3i",c)
            end
            @printf(":%12.8f\n",value[root])
            idx += 1
        end
    end
end

"""
    norm(s::TPSCIstate, root)
"""
function LinearAlgebra.norm(s::TPSCIstate, root)
    norm = 0
    for (fock,configs) in s.data
        for (config,coeff) in configs
            norm += coeff[root]*coeff[root]
        end
    end
    return sqrt(norm)
end

"""
    norm(s::TPSCIstate{T,N,R}) where {T,N,R}
"""
function LinearAlgebra.norm(s::TPSCIstate{T,N,R}) where {T,N,R}
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

# todo: remove
"""
    normalize!(s::AbstractState)
"""
function normalize!(s::AbstractState)
    scale!(s,1/sqrt(dot(s,s))) 
end

"""
    scale!(s::TPSCIstate,c)
"""
function scale!(s::TPSCIstate{T,N,R},c;root=nothing) where {T,N,R}
    if root == nothing
        for (fock,configs) in s.data
            for (config,coeff) in configs
                s[fock][config] .= coeff.*c
            end
        end
    else
        root <= R || error("root>R")
        for (fock,configs) in s.data
            for (config,coeff) in configs
                s[fock][config][root] = coeff[root]*c
            end
        end
    end
end
    
"""
    dot(v1::TPSCIstate,v2::TPSCIstate; r1=1, r2=1)
"""
function LinearAlgebra.dot(v1::TPSCIstate{T,N,1},v2::TPSCIstate{T,N,1}) where {T,N}
    d = T(0)
    for (fock,configs) in v1.data
        haskey(v2.data, fock) || continue
        for (config,coeff) in configs
            haskey(v2.data[fock], config) || continue
            d += coeff[1] * v2.data[fock][config][1]
        end
    end
    return d
end
    
"""
    dot(v1::TPSCIstate,v2::TPSCIstate; r1=1, r2=1)
"""
function LinearAlgebra.dot(v1::TPSCIstate{T,N,R}, v2::TPSCIstate{T,N,R}, r1, r2) where {T,N,R}
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
    overlap(v1::TPSCIstate{T,N,R}, v2::TPSCIstate{T,N,R}) where {T,N,R}

Compute overlap matrix between `v1` and `v2`
"""
function overlap(v1::TPSCIstate{T,N,R}, v2::TPSCIstate{T,N,R}) where {T,N,R}
    #={{{=#
    overlap = zeros(T,R,R)
    for (fock,configs) in v2.data
        haskey(v1, fock) || continue
        for (config,coeffs) in configs
            haskey(v1[fock], config) || continue
            for ri in 1:R
                #overlap[ri,ri] += v1[fock][config][ri]*v2[fock][config][ri]
                #for rj in ri+1:R
                #    overlap[ri,rj] += v1[fock][config][ri]*v2[fock][config][rj]
                #    overlap[rj,ri] = overlap[ri,rj]
                #end
                for rj in 1:R
                    overlap[ri,rj] += v1[fock][config][ri]*v2[fock][config][rj]
                end
            end
        end
    end
    return overlap
    #=}}}=#
end


"""
    orth!(v1::TPSCIstate{T,N,R}) where {T,N,R}
"""
function orth!(v1::TPSCIstate{T,N,R}) where {T,N,R}
    d = T(0)
    F = svd(get_vector(v1))

    set_vector!(v1, F.U*F.Vt)
    return 
end

"""
    Base.:*(A::TPSCIstate{T,N,R}, C::AbstractArray) where {T,N,R}

TBW
"""
function Base.:*(A::TPSCIstate{T,N,R}, C::AbstractArray) where {T,N,R}
    B = deepcopy(A)
    zero!(B)
    set_vector!(B, get_vector(A)*C)
    return B
end

"""
    mult!(A::TPSCIstate{T,N,R}, C::AbstractArray) where {T,N,R}

TBW
"""
function mult!(A::TPSCIstate{T,N,R}, C::AbstractArray) where {T,N,R}
    for (fock, configs) in A.data
        for (config, coeffs) in configs
            #A[fock][config] .=  C'*A[fock][config]
            mul!(A[fock][config], C', A[fock][config]) 
        end
    end
end


function Base.:-(A::TPSCIstate{T,N,R}, B::TPSCIstate{T,N,R}) where {T,N,R}
    C = deepcopy(B)
    scale!(C,-1.0)
    add!(C, A)
    return C
end
    
function Base.:+(A::TPSCIstate{T,N,R}, B::TPSCIstate{T,N,R}) where {T,N,R}
    C = deepcopy(B)
    add!(C, A)
    return C
end
    
function Base.deepcopy(in::TPSCIstate{T,N,R}) where {T,N,R}
    out = TPSCIstate(in.clusters, T=T, R=R)
    for (fock, configs) in in.data
        # add_fockconfig!(out, fock)
        length(configs) > 0 || continue
        # out[fock] = copy(configs)
        # out[fock] = deepcopy(configs)
        out[fock] = OrderedDict(zip(keys(configs),copy.(values(configs))))
        
        # outf = out[fock]
        for (config, coeffs) in configs
            # outf[config] = copy(coeffs)
            # outf[config] = MVector{R,T}(i for i in coeffs)
        end
        # length(out[fock]) == length(in[fock]) || throw(DimensionMismatch)
    end
    # length(out.data) == length(in.data) || throw(DimensionMismatch)
    # length(out) == length(in) || throw(DimensionMismatch)
    return out
end

    
"""
    prune_empty_fock_spaces!(s::TPSCIstate)
        
remove fock_spaces that don't have any configurations 
"""
function prune_empty_fock_spaces!(s::TPSCIstate)
    keylist = [keys(s.data)...]
    for fock in keylist
        if length(s[fock]) == 0
            delete!(s.data, fock)
        end
    end
#    # I'm not sure why this is necessary
#    idx = 0
#    for (fock,configs) in s.data
#        for (config, coeffs) in s.data[fock]
#            idx += 1
#        end
#    end
    return 
end

"""
    zero!(s::TPSCIstate)

set all elements to zero
"""
function zero!(s::TPSCIstate{T,N,R}) where {T,N,R}
    for (fock,configs) in s.data
        for (config,coeffs) in configs                
            s.data[fock][config] = zeros(T, size(MVector{R,T}))
            #s.data[fock][config] = zeros(MVector{R,T})
        end
    end
end


"""
    function randomize!(s::TPSCIstate{T,N,R}) where {T,N,R}

set all elements to random values, and orthogonalize
"""
function randomize!(s::TPSCIstate{T,N,R}) where {T,N,R}
    #={{{=#
    v0 = rand(T,size(s)) .- .5 
    set_vector!(s,v0)
    orthonormalize!(s)
end
#=}}}=#


"""
    function orthonormalize!(s::TPSCIstate{T,N,R}) where {T,N,R}

orthonormalize
"""
function orthonormalize!(s::TPSCIstate{T,N,R}) where {T,N,R}
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
    clip!(s::TPSCIstate; thresh=1e-5)
"""
function clip!(s::TPSCIstate; thresh=1e-5)
#={{{=#
    for (fock,configs) in s.data
        for (config,coeff) in configs      
            if all(abs(c) < thresh for c in coeff)
                delete!(s.data[fock], config)
            end
        end
    end
    prune_empty_fock_spaces!(s)
end
#=}}}=#


"""
    eye!(s::TPSCIstate)
"""
function eye!(s::TPSCIstate{T,N,R}) where {T,N,R}
    set_vector!(s, Matrix{T}(I,size(s)))
end



"""
    add!(s1::TPSCIstate, s2::TPSCIstate)

Add coeffs in `s2` to `s1`
"""
function add!(s1::TPSCIstate, s2::TPSCIstate)
    for (fock,configs) in s2.data
        if haskey(s1, fock)
            for (config,coeffs) in configs
                if haskey(s1[fock], config)
                    s1[fock][config] .+= s2[fock][config]
                else
                    s1[fock][config] = copy(s2[fock][config])
                end
            end
        else
            s1[fock] = copy(s2[fock])
        end
    end
end


"""
    function extract_roots(v::TPSCIstate{T,N,R}, roots)

Extract roots to give new `TPSCIstate` 
"""
function extract_roots(v::TPSCIstate{T,N,R}, roots) where {T,N,R}
    vecs = get_vector(v)[:,roots]

    out = TPSCIstate(v.clusters, T=T, R=length(roots))
    for (fock,configs) in v.data
        add_fockconfig!(out,fock)
        for (config,coeffs) in configs
            out[fock][config] = deepcopy(v[fock][config][roots])
        end
    end

    return out
end



"""
    add_spin_focksectors(state::TPSCIstate{T,N,R}) where {T,N,R}

Add the focksectors needed to spin adapt the given `TPSCIstate`
"""
function add_spin_focksectors(state::TPSCIstate{T,N,R}) where {T,N,R}
    out = deepcopy(state)
    gs = ClusterConfig([1 for i in 1:N])
    for (fock, configs) in state.data

        for f in possible_spin_focksectors(state.clusters, fock)
            FermiCG.add_fockconfig!(out, f)
            out[f][gs] = zeros(T,R)
        end
    end
    return out
end

"""
    ct_analysis(s::TPSCIstate; ne_cluster=10, thresh=1e-5, nroots=1)

Analyzes charge transfer for each root of the TPSCIstate
Prints total weight of charge transfer in each root
Only works currently if all clusters have same # of electrons!!
# Arguments
- `s::TPSCIstate`
- `ne_cluster`:  Int, number of total electrons in each cluster
- `thresh`:  Threshold for printing but does not effect total ct
- `nroots`: Total number of roots
"""
function ct_analysis(s::TPSCIstate; ne_cluster=10, thresh=1e-5, nroots=1)
    for root in 1:nroots
        println()
        @printf(" --------------------------------------------------\n")
        @printf(" ----------- CHARGE TRANSFER ANALYSIS -------------\n")
        @printf(" --------------------------------------------------\n")
        @printf(" ----------                root ------:     = %5i  \n",root)
        @printf(" --------------------------------------------------\n")
        @printf(" Printing contributions greater than: %f", thresh)
        @printf("\n")
        @printf(" %-20s%-20s%-20s\n", "Weight", "# Configs", "Fock space(α,β)...")
        @printf(" %-20s%-20s%-20s\n", "-------", "---------", "----------")
        ct = 0
        for (fock,configs) in s.data
            prob = 0
            for cluster in 1:length(s.clusters)
                if sum(fock[cluster]) != ne_cluster
                    prob = 0
                    for (config, coeff) in configs 
                        prob += coeff[root]*coeff[root] 
                    end
                    if prob > thresh
                        @printf(" %-20.5f%-20i", prob,length(s.data[fock]))
                        for sector in fock 
                            @printf("(%2i,%-2i)", sector[1],sector[2])
                        end
                        println()
                    end
                end
                break
            end
            ct += prob
        end
        print(" --------------------------------------------------\n")
        @printf(" %-10.5f", ct)
        @printf(" %-10s", "=   Total charge transfer weight")
        println()
        print(" --------------------------------------------------\n")
    end
end

"""
    ct_table(s::TPSCIstate; ne_cluster=10, nroots=1)

Prints total weight of charge transfer in each root in table formate
# Arguments
- `s::TPSCIstate`
- `ne_cluster`:  Int, number of total electrons in each cluster
- `nroots`: Total number of roots
"""
function ct_table(s::TPSCIstate; ne_cluster=10, nroots=1)
    @printf(" -----------------------\n")
    @printf(" --- CHARGE TRANSFER ---\n")
    @printf(" -----------------------\n")
    @printf(" %-15s%-10s\n", "Root", "Total CT")
    @printf(" %-15s%-10s\n", "-------", "---------")
    for root in 1:nroots
        ct = 0
        for (fock,configs) in s.data
            prob = 0
            for cluster in 1:length(s.clusters)
                if sum(fock[cluster]) != ne_cluster
                    prob = 0
                    for (config, coeff) in configs 
                        prob += coeff[root]*coeff[root] 
                    end
                end
                break
            end
            ct += prob
        end
        @printf(" %-15i%-10.5f", root, ct)
        println()
    end
end




"""
    correlation_functions(v::TPSCIstate{T,N,R}) where {T,N,R}

Compute <N>, <N1N2 - <N1><N2>>, <Sz>, and <Sz1Sz2 - <Sz1><Sz2>>
"""
function correlation_functions(v::TPSCIstate{T,N,R}) where {T,N,R}

    n1 = [zeros(N) for i in 1:R]
    n2 = [zeros(N,N) for i in 1:R]
    sz1 = [zeros(N) for i in 1:R]
    sz2 = [zeros(N,N) for i in 1:R]

    for root in 1:R
        for (fock,configs) in v.data
            prob = 0
            for (config, coeff) in configs 
                prob += coeff[root]*coeff[root] 
            end

            for ci in v.clusters
                n1[root][ci.idx] += prob * (fock[ci.idx][1] + fock[ci.idx][2])
                sz1[root][ci.idx] += prob * (fock[ci.idx][1] - fock[ci.idx][2]) / 2
                for cj in v.clusters
                    ci.idx <= cj.idx || continue
                    n2[root][ci.idx, cj.idx] += prob * (fock[ci.idx][1] + fock[ci.idx][2]) * (fock[cj.idx][1] + fock[cj.idx][2]) 
                    sz2[root][ci.idx, cj.idx] += prob * (fock[ci.idx][1] - fock[ci.idx][2]) * (fock[cj.idx][1] - fock[cj.idx][2]) / 4
                    n2[root][cj.idx, ci.idx] = n2[root][ci.idx, cj.idx]
                    sz2[root][cj.idx, ci.idx] = sz2[root][ci.idx, cj.idx]
                end
            end
        end
    end

    for r in 1:R
        n2[r] = n2[r] - n1[r]*n1[r]'
        sz2[r] = sz2[r] - sz1[r]*sz1[r]'
    end

    return n1, n2, sz1, sz2
end




nroots(v::TPSCIstate{T,N,R}) where {T,N,R} = R
type(v::TPSCIstate{T,N,R}) where {T,N,R} = T
nclusters(v::TPSCIstate{T,N,R}) where {T,N,R} = N
