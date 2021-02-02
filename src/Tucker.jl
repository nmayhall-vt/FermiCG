using Profile
using LinearMaps
using BenchmarkTools
#using KrylovKit
using IterativeSolvers

#"""
#This type is used to characterize a given `TuckerBlock` for a given FockConfig
#
#    cluster::Cluster
#    na::Int8
#    nb::Int8
#    start::UInt8
#    stop::UInt8
#"""
#struct ClusterSubspace
#    cluster::Cluster
#    na::Int8
#    nb::Int8
#    start::UInt8
#    stop::UInt8
#    ClusterSubspace(cluster, na, nb, start, stop) = abs(stop-start+1) > dim_tot(cluster, na, nb) ? error("Range too large") : new(cluster, na, nb, start, stop)
#end
#function Base.display(css::ClusterSubspace)
#    @printf("   Cluster: %3i range: %3i:%-3i ",css.cluster.idx, css.start, css.stop)
#    println((css.na, css.nb))
#end
#Base.length(css::ClusterSubspace) = return css.stop - css.start + 1
#
#"""
#Defines a list of subspaces across all clusters 
#
#    config::Vector{ClusterSubspace}
#"""
#struct TuckerBlock
#    config::Vector{ClusterSubspace}
#end
#function Base.display(tb::TuckerBlock)
#    println(" TuckerBlock: ")
#    display.(tb.config)
#end
#Base.length(tb::TuckerBlock) = sum(length.(tb.config)) 


"""
Defines a single cluster's subspace for Tucker. Each focksector is allowed to have a distinct cluster state range 
for the subspace.

    cluster::Cluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
"""
struct TuckerSubspace
    cluster::Cluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
end
function TuckerSubspace(cluster::Cluster)
    return TuckerSubspace(cluster,OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}())
end
Base.setindex!(tss::TuckerSubspace, i, j) = tss.data[j] = i
Base.getindex(tss::TuckerSubspace, i) = return tss.data[i] 
function Base.display(tss::TuckerSubspace)
    @printf(" Subspace for Cluster: %4i : ", tss.cluster.idx)
    display(tss.cluster)
    for (fock,range) in tss.data
        @printf("  %10s   Range: %4i → %-4i Dim %4i\n",Int.(fock), first(range), last(range), length(range))
    end
end

"""
    get_ortho_compliment(tss::TuckerSubspace, cb::ClusterBasis)

For a given `TuckerSubspace`, `tss`, return the subspace remaining
"""
function get_ortho_compliment(tss::TuckerSubspace, cb::ClusterBasis)
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

    return TuckerSubspace(tss.cluster, data)
#=}}}=#
end


"""
    config::Vector{UnitRange{Int}}
"""
struct TuckerConfig <: SparseConfig
    config::Vector{UnitRange{Int}}
end
"""
    TuckerConfig()
"""
function TuckerConfig()
    return TuckerConfig([])
end
Base.size(tc::TuckerConfig) = length.(tc.config)
Base.hash(a::TuckerConfig) = hash(a.config)
Base.isequal(x::TuckerConfig, y::TuckerConfig) = all(isequal.(x.config, y.config))
Base.push!(tc::TuckerConfig, range) = push!(tc.config,range)
Base.:(==)(x::TuckerConfig, y::TuckerConfig) = all(x.config .== y.config)

# Conversions
Base.convert(::Type{TuckerConfig}, input::Vector{UnitRange{T}}) where T<:Integer = TuckerConfig(input)

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
"""
struct TuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Array}}
end
Base.haskey(ts::TuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::TuckerState, i) = return ts.data[i]
Base.setindex!(ts::TuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::TuckerState, state=1) = iterate(ts.data, state)

"""
    TuckerState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function TuckerState(clusters)
    return TuckerState(clusters,OrderedDict())
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
    unfold!(ts::TuckerState)
"""
function unfold!(ts::TuckerState)
#={{{=#
    for (fock,configs) in ts
        #display(fock)
        for (config,coeffs) in configs 
            #length(size(coeffs)) == length(ts.clusters)+1 || error(" Can only unfold a folded vector")
            nroots = last(size(coeffs)) 
           
            #display(config)
            #display(size(coeffs))
            #display(size(config))
            #display(nroots)

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
    for (fock,configs) in ts
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
    TuckerState(clusters, tss::Vector{TuckerSubspace}, na, nb; nroots=1)

Constructor
- `clusters::Vector{Cluster}`
- `tss::Vector{TuckerSubspace}`
- `na::Int` Number of alpha
- `nb::Int` Number of beta
"""
function TuckerState(clusters, tss, na, nb; nroots=1)
   #={{{=#
    length(tss) == length(clusters) || error("# of clusters don't match # of subspaces")

    s = TuckerState(clusters)
    ns = []
    for tssi in tss
        nsi = []
        for (fock,range) in tssi.data
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

            tuckconfig = TuckerConfig()
            for ci in clusters
                tssi = tss[ci.idx]
                push!(tuckconfig, tssi.data[newfock[ci.idx]])
            end

            haskey(s.data, fockconfig) == false || error(" here:", fockconfig)
            add_fockconfig!(s,fockconfig) 
            s[fockconfig][tuckconfig] = zeros((size(tuckconfig)...,nroots))  # todo - finish this
        end
    end
    return s
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
    print(" --------------------------------------------------\n")
    fold!(s)
#=}}}=#
end
"""
    print_fock_occupations(s::TuckerState; root=1, thresh=1e-3)

Pretty print
"""
function print_fock_occupations(s::TuckerState; root=1, thresh=1e-3)
#={{{=#

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




########################################################################################################
########################################################################################################

"""
"""
function form_sigma_block!(term::ClusteredTerm1B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs, ket_coeffs)
#={{{=#
    #display(term)
    #println(bra, ket)

    c1 = term.clusters[1]
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != c1.idx || continue

        fock_bra[ci] == fock_ket[ci] || throw(Exception)
        bra[ci] == ket[ci] || return 0.0 
    end

    # 
    # make sure active clusters are correct transitions 
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)

    op = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][bra[c1.idx],ket[c1.idx]]
        



    # now transpose state vectors and multiply, also, try without transposing to compare
    indices = collect(1:n_clusters+1)
    indices[c1.idx] = 0
    perm,_ = bubble_sort(indices)

    length(size(ket_coeffs)) == n_clusters + 1 || error(" tensors should be folded")
    
    n_roots = last(size(ket_coeffs))
    ket_coeffs2 = permutedims(ket_coeffs,perm)
    bra_coeffs2 = permutedims(bra_coeffs,perm)

    dim1 = size(ket_coeffs2)
    ket_coeffs2 = reshape(ket_coeffs2, dim1[1], prod(dim1[2:end]))

    dim2 = size(bra_coeffs2)
    bra_coeffs2 = reshape(bra_coeffs2, dim2[1], prod(dim2[2:end]))

    bra_coeffs2 .+= op * ket_coeffs2
#    if bra==ket
#        display(op)
#        display(bra_coeffs2)
#        display(ket_coeffs2)
#    end
    

    ket_coeffs2 = reshape(ket_coeffs2, dim1)
    bra_coeffs2 = reshape(bra_coeffs2, dim2)
   
    # now untranspose
    perm,_ = bubble_sort(perm)
    ket_coeffs2 = permutedims(ket_coeffs2,perm)
    bra_coeffs2 = permutedims(bra_coeffs2,perm)
  
    bra_coeffs .= bra_coeffs2
    return  
#=}}}=#
end
"""
"""
function form_sigma_block!(term::ClusteredTerm2B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs, ket_coeffs)
#={{{=#
    #display(term)
    #println(bra, ket)

    c1 = term.clusters[1]
    c2 = term.clusters[2]
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue

        fock_bra[ci] == fock_ket[ci] || throw(Exception)
        bra[ci] == ket[ci] || return 0.0 
    end

    #display(fock_bra)
    #display(fock_ket)
    #display(term.delta)
    #display(term)
    # 
    # make sure active clusters are correct transitions 
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
#    display(term)
#    display(fock_bra)
#    display(fock_ket)
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
   
    op = Array{Float64}[]
    #display(size(term.ints))
    #display(size(gamma1))
    #display(size(gamma2))
    cache_key = (fock_bra[c1.idx], fock_bra[c2.idx], fock_ket[c1.idx], fock_ket[c2.idx], bra[c1.idx], bra[c2.idx], ket[c1.idx], ket[c2.idx])
    if haskey(term.cache, cache_key)
        op = term.cache[cache_key]
    else
        @tensor begin
            op[q,J,I] := term.ints[p,q] * gamma1[p,I,J]
            op[J,L,I,K] := op[q,J,I] * gamma2[q,K,L]
        end
        term.cache[cache_key] = op
    end
    

    # possibly cache some of these integrals

    # now transpose state vectors and multiply, also, try without transposing to compare
    indices = collect(1:n_clusters+1)
    indices[c1.idx] = 0
    indices[c2.idx] = 0
    perm,_ = bubble_sort(indices)

    length(size(ket_coeffs)) == n_clusters + 1 || error(" tensors should be folded")
    
    n_roots = last(size(ket_coeffs))
    ket_coeffs2 = permutedims(ket_coeffs, perm)
    bra_coeffs2 = permutedims(bra_coeffs, perm)

    dim1 = size(ket_coeffs2)
    ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2], prod(dim1[3:end]))

    dim2 = size(bra_coeffs2)
    bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2], prod(dim2[3:end]))

    op = reshape(op, prod(size(op)[1:2]),prod(size(op)[3:4]))
    
#    println()
#    display((c1.idx, c2.idx))
#    display(perm')
#    display(size(op))
#    display(size(ket_coeffs))
#    display(size(permutedims(ket_coeffs,perm)))
#    display(size(ket_coeffs2))
    if state_sign == 1
        bra_coeffs2 .+= op' * ket_coeffs2
    elseif state_sign == -1
        bra_coeffs2 .-= op' * ket_coeffs2
    else
        error()
    end
    

    ket_coeffs2 = reshape(ket_coeffs2, dim1)
    bra_coeffs2 = reshape(bra_coeffs2, dim2)
   
    # now untranspose
    perm,_ = bubble_sort(perm)
    ket_coeffs2 = permutedims(ket_coeffs2,perm)
    bra_coeffs2 = permutedims(bra_coeffs2,perm)
    
    bra_coeffs .= bra_coeffs2
    return  
#=}}}=#
end
"""
"""
function form_sigma_block!(term::ClusteredTerm3B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs, ket_coeffs)
#={{{=#
    #display(term)
    #println(bra, ket)

    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue

        fock_bra[ci] == fock_ket[ci] || throw(Exception)
        bra[ci] == ket[ci] || return 0.0 
    end

    # 
    # make sure active clusters are correct transitions 
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # op[IKM,JLN] = <I|p'|J> h(pqr) <K|q|L> <M|r|N>
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
   
    op = Array{Float64}[]
    #@tensor begin
    #    op[J,L,N,I,K,M] := term.ints[p,q,r] * gamma1[p,I,J] * gamma2[q,K,L] * gamma3[r,M,N]  
    #end
    cache_key = (fock_bra[c1.idx], fock_bra[c2.idx], fock_bra[c3.idx], 
                 fock_ket[c1.idx], fock_ket[c2.idx], fock_ket[c3.idx], 
                 bra[c1.idx], bra[c2.idx], bra[c3.idx], 
                 ket[c1.idx], ket[c2.idx], ket[c3.idx])

    
    if haskey(term.cache, cache_key)
        op = term.cache[cache_key]
    else
        @tensor begin
            op[q,r,I,J] := term.ints[p,q,r] * gamma1[p,I,J]
            op[r,I,J,K,L] := op[q,r,I,J] * gamma2[q,K,L]  
            op[J,L,N,I,K,M] := op[r,I,J,K,L] * gamma3[r,M,N]  
        end
        term.cache[cache_key] = op
   
        # compress this
#        opsize = size(op)
#        op = reshape(op, prod(size(op)[1:3]), prod(size(op)[4:6]))
#        F = svd(op)
#        #display(F.S)
#        cut = 0
#        for si in 1:length(F.S) 
#            if F.S[si] < 1e-5
#                F.S[si] = 0
#                cut += 1
#            end
#        end
#        display((length(F.S), cut))
#        op = F.U * Diagonal(F.S) * F.Vt
#        op = reshape(op,opsize)
    end
   

    # now transpose state vectors and multiply, also, try without transposing to compare
    indices = collect(1:n_clusters+1)
    indices[c1.idx] = 0
    indices[c2.idx] = 0
    indices[c3.idx] = 0
    perm,_ = bubble_sort(indices)

    length(size(ket_coeffs)) == n_clusters + 1 || error(" tensors should be folded")
    
    n_roots = last(size(ket_coeffs))
    ket_coeffs2 = permutedims(ket_coeffs,perm)
    bra_coeffs2 = permutedims(bra_coeffs,perm)

    dim1 = size(ket_coeffs2)
    ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2]*dim1[3], prod(dim1[4:end]))

    dim2 = size(bra_coeffs2)
    bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2]*dim2[3], prod(dim2[4:end]))

    op = reshape(op, prod(size(op)[1:3]),prod(size(op)[4:6]))
    if state_sign == 1
        bra_coeffs2 .+= op' * ket_coeffs2
    elseif state_sign == -1
        bra_coeffs2 .-= op' * ket_coeffs2
    else
        error()
    end
    

    ket_coeffs2 = reshape(ket_coeffs2, dim1)
    bra_coeffs2 = reshape(bra_coeffs2, dim2)
   
    # now untranspose
    perm,_ = bubble_sort(perm)
    ket_coeffs2 = permutedims(ket_coeffs2,perm)
    bra_coeffs2 = permutedims(bra_coeffs2,perm)
    
    bra_coeffs .= bra_coeffs2
    return  
#=}}}=#
end
"""
"""
function form_sigma_block!(term::ClusteredTerm4B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs, ket_coeffs)
#={{{=#
    #display(term)
    #println(bra, ket)

    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue
        ci != c4.idx || continue

        fock_bra[ci] == fock_ket[ci] || throw(Exception)
        bra[ci] == ket[ci] || return 0.0 
    end

    # 
    # make sure active clusters are correct transitions 
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)
    fock_bra[c4.idx] == fock_ket[c4.idx] .+ term.delta[4] || throw(Exception)

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # op[IKMO,JLNP] = <I|p'|J> h(pqrs) <K|q|L> <M|r|N> <O|s|P>
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
   
    op = Array{Float64}[]
    @tensor begin
        op[J,L,N,P,I,K,M,O] := term.ints[p,q,r,s] * gamma1[p,I,J] * gamma2[q,K,L] * gamma3[r,M,N] * gamma4[s,O,P]  
    end
    #@tensor begin
    #    op[q,r,I,J] := term.ints[p,q,r] * gamma1[p,I,J]
    #    op[r,I,J,K,L] := op[q,r,I,J] * gamma2[q,K,L]  
    #    op[J,L,N,I,K,M] := op[r,I,J,K,L] * gamma2[r,M,N]  
    #end
    
    # possibly cache some of these integrals
    # compress this
#    opsize = size(op)
#    op = reshape(op, prod(size(op)[1:4]), prod(size(op)[5:8]))
#    F = svd(op)
#    #display(F.S)
#    for si in 1:length(F.S) 
#        if F.S[si] < 1e-3
#            F.S[si] = 0
#        end
#    end
#    op = F.U * Diagonal(F.S) * F.Vt
#    op = reshape(op,opsize)

    # now transpose state vectors and multiply, also, try without transposing to compare
    indices = collect(1:n_clusters+1)
    indices[c1.idx] = 0
    indices[c2.idx] = 0
    indices[c3.idx] = 0
    indices[c4.idx] = 0
    perm,_ = bubble_sort(indices)

    length(size(ket_coeffs)) == n_clusters + 1 || error(" tensors should be folded")
    
    n_roots = last(size(ket_coeffs))
    ket_coeffs2 = permutedims(ket_coeffs,perm)
    bra_coeffs2 = permutedims(bra_coeffs,perm)

    dim1 = size(ket_coeffs2)
    ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2]*dim1[3]*dim1[4], prod(dim1[5:end]))

    dim2 = size(bra_coeffs2)
    bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2]*dim2[3]*dim2[4], prod(dim2[5:end]))

    op = reshape(op, prod(size(op)[1:4]),prod(size(op)[5:8]))
    if state_sign == 1
        bra_coeffs2 .+= op' * ket_coeffs2
    elseif state_sign == -1
        bra_coeffs2 .-= op' * ket_coeffs2
    else
        error()
    end
    

    ket_coeffs2 = reshape(ket_coeffs2, dim1)
    bra_coeffs2 = reshape(bra_coeffs2, dim2)
   
    # now untranspose
    perm,_ = bubble_sort(perm)
    ket_coeffs2 = permutedims(ket_coeffs2,perm)
    bra_coeffs2 = permutedims(bra_coeffs2,perm)
    
    bra_coeffs .= bra_coeffs2
    return  
#=}}}=#
end


"""
    build_sigma!(sigma_vector, ci_vector, cluster_ops, clustered_ham)
"""
function build_sigma!(sigma_vector, ci_vector, cluster_ops, clustered_ham)
    #={{{=#

    for (fock_bra, configs_bra) in sigma_vector
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket
                

                    for term in clustered_ham[fock_trans]
                    
                        #term isa ClusteredTerm1B || continue
                       
                        FermiCG.form_sigma_block!(term, cluster_ops, fock_bra, config_bra, 
                                                  fock_ket, config_ket,
                                                  coeff_bra, coeff_ket)


                    end
                end
            end
        end
    end
    return 
    #=}}}=#
end



"""
    get_map(ci_vector, cluster_ops, clustered_ham)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ci_vector::TuckerState, cluster_ops, clustered_ham; shift = nothing)
    #={{{=#
    iters = 0
   
    dim = length(ci_vector)
    function mymatvec(v)
        iters += 1
        
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v, length(v), nr)
        elseif length(size(v)) == 2
            nr = size(v)[2]
        else
            error(" is tensor not unfolded?")
        end
    
      
        set_vector!(ci_vector, v)
        
        fold!(ci_vector)
        sig = deepcopy(ci_vector)
        zero!(sig)
        build_sigma!(sig, ci_vector, cluster_ops, clustered_ham)

        unfold!(ci_vector)
        
        sig = get_vector(sig)

        if shift != nothing
            # this is how we do CEPA
            sig += shift * get_vector(ci_vector)
        end

        return sig 
    end
    return LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#

function tucker_ci_solve!(ci_vector, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    unfold!(ci_vector) 
    Hmap = get_map(ci_vector, cluster_ops, clustered_ham)

    v0 = get_vector(ci_vector)
    nr = size(v0)[2] 
    
    davidson = Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=1e-5)
    #Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    @time e,v = FermiCG.solve(davidson)
    set_vector!(ci_vector,v)
    return e,v
end
#=}}}=#


"""
0 = <x|H - E0|x'>v(x') + <x|H - E0|p>v(p) 
0 = <x|H - E0|x'>v(x') + <x|H|p>v(p) 
A(x,x')v(x') = -H(x,p)v(p)

here, x is outside the reference space, and p is inside

Ax=b

works for one root at a time
"""
function tucker_cepa_solve!(ref_vector, ci_vector, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    fold!(ref_vector) 
    fold!(ci_vector) 
    sig = deepcopy(ref_vector) 
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham)
    e0 = dot(ref_vector, sig)
    size(e0) == (1,1) || error("Only one state at a time please")
    e0 = e0[1,1]
    @printf(" Reference Energy: %12.8f\n",e0)
    

    x_vector = deepcopy(ci_vector)
    #
    # now remove reference space from ci_vector
    for (fock,configs) in ref_vector
        if haskey(x_vector, fock)
            for (config,coeffs) in configs
                if haskey(x_vector[fock], config)
                    delete!(x_vector[fock], config)
                end
            end
        end
    end

    b = deepcopy(x_vector) 
    zero!(b)
    build_sigma!(b, ref_vector, cluster_ops, clustered_ham)
    bv = -get_vector(b) 

    function mymatvec(v)
        unfold!(x_vector)
        set_vector!(x_vector, v)
        fold!(x_vector)
        sig = deepcopy(x_vector)
        zero!(sig)
        build_sigma!(sig, x_vector, cluster_ops, clustered_ham)
        unfold!(x_vector)
        unfold!(sig)
        
        sig_out = get_vector(sig)
        sig_out .-= e0*get_vector(x_vector)
        return sig_out
    end
    dim = length(x_vector)
    Axx = LinearMap(mymatvec, dim, dim)
    #Axx = LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
    
    x, solver = cg(Axx,bv,initially_zero=true, log=true)

    set_vector!(x_vector, x)
   
    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = dot(sig,ref_vector)
    size(ecorr) == (1,1) || error(" Dimension Error")
    ecorr = ecorr[1]
  
    zero!(ci_vector)
    add!(ci_vector, ref_vector)
    add!(ci_vector, x_vector)

    @printf(" E(CEPA): corr %12.8f electronic %12.8f\n",ecorr, ecorr+e0)
    #x, info = linsolve(Hmap,zeros(size(v0)))
    return ecorr+e0, x
end#=}}}=#
