

abstract type SparseConfig end


"""
"""
struct FockConfig <: SparseConfig
    config::Vector{Tuple{UInt8,UInt8}}
end
"""
"""
struct ClusterConfig <: SparseConfig
    config::Vector{UInt8}
end
"""
"""
struct TransferConfig <: SparseConfig
    config::Vector{Tuple{Int8,Int8}}
end
Base.hash(a::TransferConfig) = hash(a.config)
Base.hash(a::ClusterConfig) = hash(a.config)
Base.hash(a::FockConfig) = hash(a.config)

Base.length(f::SparseConfig) = length(f.config)
Base.getindex(s::SparseConfig, i) = s.config[i]
Base.setindex!(s::SparseConfig, i, j) = s.config[j] = i

#Base.display(f::SparseConfig) = display(f.config)
#Base.display(f::ClusterConfig) = display(Int.(f.config))
#Base.display(f::FockConfig) = display(Tuple{Int,Int}.(f.config))
#
#   Iteration through Configs
Base.eltype(iter::FockConfig) = Tuple{UInt8,UInt8} 
Base.eltype(iter::ClusterConfig) = UInt8 
Base.eltype(iter::TransferConfig) = Tuple{Int8,Int8}

function Base.iterate(conf::SparseConfig, state=1)
    if state >= length(conf)
        return nothing
    end
    return conf.config[state], state + 1
end

# Conversions
Base.convert(::Type{TransferConfig}, input::Vector{Tuple{T,T}}) where T<:Integer = TransferConfig(input)
Base.convert(::Type{TransferConfig}, input::Tuple{Tuple{T,T}}) where T<:Integer = TransferConfig(input)
Base.convert(::Type{ClusterConfig}, input::Vector{T}) where T<:Integer = ClusterConfig(input)
Base.convert(::Type{ClusterConfig}, input::Tuple{T}) where T<:Integer = ClusterConfig(input)
Base.convert(::Type{FockConfig}, input::Vector{Tuple{T,T}}) where T<:Integer = FockConfig(input)
Base.convert(::Type{FockConfig}, input::Tuple{Tuple{T,T}}) where T<:Integer = FockConfig(input)

TransferConfig(in::Vector{Tuple{T,T}}) where T<:Union{Int16,Int} = TransferConfig([(Int8(i[1]), Int8(i[2])) for i in in]) 
TransferConfig(in::Tuple{Tuple{T,T}}) where T<:Union{Int16,Int} = TransferConfig([(Int8(i[1]), Int8(i[2])) for i in in]) 
FockConfig(in::Vector{Tuple{T,T}}) where T<:Union{Int16,Int} = FockConfig([(UInt8(i[1]), UInt8(i[2])) for i in in]) 
FockConfig(in::Tuple{Tuple{T,T}}) where T<:Union{Int16,Int} = FockConfig([(UInt8(i[1]), UInt8(i[2])) for i in in]) 
ClusterConfig(in::Vector{T}) where T<:Union{Int16,Int} = ClusterConfig([UInt8(i) for i in in]) 
ClusterConfig(in::Tuple{T}) where T<:Union{Int16,Int} = ClusterConfig([UInt8(i) for i in in]) 
#ClusterConfig(in::Vector{Int}) = ClusterConfig([UInt8(i) for i in in]) 


function Base.:-(a::FockConfig, b::FockConfig)
    return TransferConfig(a.config .- b.config)
end

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
    data::OrderedDict{FockConfig,OrderedDict{ClusterConfig,Float64}}
    #data::OrderedDict{Vector{Tuple{Int16,Int16}},OrderedDict{Vector{Int16},Float64}}
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
    fockc = FockConfig(fock)
    s.data[fockc] = OrderedDict{ClusterConfig, Float64}()
    config = ClusterConfig(zeros(length(s.clusters).+1))
    #config .+= 1
    s.data[fockc][config] = 0
end

"""
    add_fockconfig!(s::ClusteredState, fock::FockConfig)
"""
function add_fockconfig!(s::ClusteredState, fock::FockConfig)
    # initialize ground state of fock
    s.data[fock] = OrderedDict{ClusterConfig, Float64}()
    config = ClusterConfig(zeros(length(s.clusters).+1))
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
Base.getindex(s::ClusteredState, fock::Vector{Tuple{T,T}}) where T<:Integer = s.data[fock]
Base.getindex(s::ClusteredState, fock) = s.data[fock]

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


"""
    expand_each_fock_space!(s::ClusteredState):
"""
function expand_each_fock_space!(s::ClusteredState, bases)
    # {{{
    println("\n Make each Fock-Block the full space")
    # create full space for each fock block defined
    for (fblock,configs) in s.data
        #println(fblock)
        dims::Vector{UnitRange{Int16}} = []
        #display(fblock)
        for c in s.clusters
            # get number of vectors for current fock space
            dim = size(bases[c.idx][fblock[c.idx]], 2)
            push!(dims, 1:dim)
        end
        for newconfig in product(dims...) 
            #display(newconfig)
            #println(typeof(newconfig))
            #
            # this is not ideal - need to find a way to directly create key
            config = ClusterConfig(collect(newconfig))
            s.data[fblock][config] = 0
            #s.data[fblock][[i for i in newconfig]] = 0
        end
    end
end
# }}}

