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
            overlap += coeffs' * coeffs
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
    randomize!(ts::TuckerState)
"""
function randomize!(ts::TuckerState)
    #={{{=#
    for (fock,configs) in ts
        for (config,coeffs) in configs 
            ts[fock][config] = rand(size(coeffs)...) .- .5
        end
    end
    #=}}}=#
end

"""
    mult!(ts::TuckerState, A)

Multiple `ts` by a matrix A
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
    TuckerState(clusters, tss::Vector{TuckerSubspace})

Constructor
- `clusters::Vector{Cluster}`
- `tss::Vector{TuckerSubspace}`
- `na::Int` Number of alpha
- `nb::Int` Number of beta
"""
function TuckerState(clusters, ss, na, nb; nroots=2)
   #={{{=#
    length(ss) == length(clusters) || error("# of clusters don't match # of subspaces")

    s = TuckerState(clusters)
    ns = []
    for tss in ss
        nsi = []
        for (fock,range) in tss.data
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
                tss = ss[ci.idx]
                push!(tuckconfig, tss.data[newfock[ci.idx]])
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
#={{{=#
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
            @printf(" %-20.3f%-20s", prob,"")
            for sector in fock 
                @printf("(%2i,%-2i)", sector[1],sector[2])
            end
            println()

            @printf("     %-16s%-20s%-20s\n", "Weight", "", "Subspaces") 
            @printf("     %-16s%-20s%-20s\n", "-------", "", "----------")
            for (config, coeffs) in configs 
                probi = sum(coeffs.*coeffs)
                if probi > thresh
                    @printf("     %-16.3f%-20i", probi,length(coeffs))
                    for range in config 
                        @printf("%4s", range)
                    end
                end
                println()
            end
            println()
        end
    end
    print(" --------------------------------------------------\n")
#=}}}=#
end



"""
    expand_each_fock_space!(s::TuckerState, bases)

For each fock space sector defined, add all possible basis states 
- `bases::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::TuckerState, bases::Vector{ClusterBasis})
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
       
        s.data[fblock][dims] = zeros(Tuple(length.(dims)))

    end
end
# }}}


