using Profile
using LinearMaps
using BenchmarkTools
using IterativeSolvers
#using TensorDecompositions


struct Tucker{T, N} 
    core::Array{T, N}
    factors::NTuple{N, Matrix{T}}
    #props::Dict{Symbol, Any}
end

#Tucker(v; thresh=-1, max_number=nothing) = Tucker(tucker_decompose(v, thresh=thresh, max_number=max_number)..., Dict())
function Tucker(A::Array; thresh=-1, max_number=nothing, verbose=0) 
    core,factors = tucker_decompose(A, thresh=thresh, max_number=max_number, verbose=verbose)
    return Tucker(core, NTuple{length(factors)}(factors))
end
recompose(t::Tucker{T,N}) where {T<:Number, N} = tucker_recompose(t.core, t.factors)
dims_large(t::Tucker{T,N}) where {T<:Number, N} = return [size(f,1) for f in t.factors]
dims_small(t::Tucker{T,N}) where {T<:Number, N} = return [size(f,2) for f in t.factors]
Base.length(t::Tucker) = return prod(dims_small(t))


function dot(t1::Tucker{T,N}, t2::Tucker{T,N}) where {T,N}
    overlaps = [] 
    all(dims_large(t1) .== dims_large(t2)) || error(" t1 and t2 don't have same dimensions")
    for f in 1:N
        push!(overlaps, t1.factors[f]' * t2.factors[f])
    end
    return sum(tucker_recompose(t1.core, overlaps) .* t2.core)
end

"""
Represents a state in an set of abitrary (yet low-rank) subspaces of a set of FockConfigs.
e.g. v[FockConfig][TuckerConfig] => Tucker Decomposed Tensor
    
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}
"""
struct CompressedTuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}
end
Base.haskey(ts::CompressedTuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::CompressedTuckerState, i) = return ts.data[i]
Base.setindex!(ts::CompressedTuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::CompressedTuckerState, state=1) = iterate(ts.data, state)

"""
    CompressedTuckerState(clusters)

Constructor
- `clusters::Vector{Cluster}`
"""
function CompressedTuckerState(clusters)
    return CompressedTuckerState(clusters,OrderedDict())
end



"""
    CompressedTuckerState(ts::TuckerState; thresh=-1, max_number=nothing, verbose=0)

Convert a `TuckerState` to a `CompressedTuckerState`
Constructor
- ts::TuckerState`
"""
function CompressedTuckerState(ts::TuckerState; thresh=-1, max_number=nothing, verbose=0)
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

    cts = CompressedTuckerState(ts.clusters)
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs

            #
            # Since TuckerState has extra dimension for state index, remove that
            tuck = Tucker(reshape(coeffs,size(coeffs)[1:end-1]), thresh=thresh, max_number=max_number, verbose=verbose) 
            if length(tuck) > 0
                if haskey(cts.data, fock)
                    cts[fock][tconfig] = tuck
                else
                    add_fockconfig!(cts, fock)
                    cts[fock][tconfig] = tuck
                end
            end
        end
    end
    return cts
end



"""
    add_fockconfig!(s::CompressedTuckerState, fock::FockConfig)
"""
function add_fockconfig!(s::CompressedTuckerState, fock::FockConfig) 
    s.data[fock] = OrderedDict{TuckerConfig, Tucker}()
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
    prune_empty_fock_spaces!(s::TuckerState)
        
remove fock_spaces that don't have any configurations 
"""
function prune_empty_fock_spaces!(s::CompressedTuckerState)
    focklist = keys(s.data)
    for fock in focklist 
        if length(s.data[fock]) == 0
            delete!(s.data, fock)
        end
    end
    focklist = keys(s.data)
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
function set_vector!(ts::CompressedTuckerState, v::Vector{T}) where T

    length(size(v)) == 1 || error(" Only takes vectors", size(v))
    nbasis = size(v)[1]

    idx = 1
    for (fock, tconfigs) in ts
        for (tconfig, tuck) in tconfigs
            dims = size(tconfig)
            
            dim1 = prod(dims)
            ts[fock][tconfig].core = reshape(v[idx:idx+dim1-1], size(tuck.core))
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
            fill!(s[fock][config].core, 0.0)
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


#"""
#    unfold!(ts::CompressedTuckerState)
#"""
#function unfold!(cts::CompressedTuckerState)
##={{{=#
#    for (fock,tconfigs) in cts.data
#        for (tconfig,tuck) in tconfigs 
#          
#            if length(size(tuck.core)) == length(cts.clusters)
#                display((size(tuck.core), size(reshape(tuck.core, (prod(size(tconfig)))))))
#                cts[fock][tconfig].core = copy(reshape(tuck.core, (prod(size(tconfig)))))
#            else
#                display((length(size(tuck.core)), length(cts.clusters)))
#                @warn("we are already unfolded, what you doing?")
#            end
#
#        end
#    end
##=}}}=#
#end
#"""
#    fold!(ts::CompressedTuckerState)
#"""
#function fold!(cts::CompressedTuckerState)
##={{{=#
#    for (fock,tconfigs) in cts.data
#        for (tconfig,tuck) in tconfigs 
#            try
#                length(size(tuck.core)) == 1 || throw(DimensionMismatch)
#                cts[fock][tconfig].core = reshape(tuck.core, (size(tconfig)...))
#            catch DimensionMismatch
#                @warn("we are already folded, what you doing?")
#            end
#        end
#    end
##=}}}=#
#end

"""
    dot(ts1::FermiCG.CompressedTuckerState, ts2::FermiCG.CompressedTuckerState)

Dot product between `ts2` and `ts1`

Warning: this assumes both `ts1` and `ts2` have the same tucker factors for each `TuckerConfig`
"""
function dot(ts1::CompressedTuckerState, ts2::CompressedTuckerState)
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
    nonorth_dot(ts1::FermiCG.CompressedTuckerState, ts2::FermiCG.CompressedTuckerState)

Dot product between 1ts2` and `ts1` where each have their own Tucker factors
"""
function nonorth_dot(ts1::CompressedTuckerState, ts2::CompressedTuckerState)
#={{{=#
    overlap = 0.0  
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        for (config,coeffs) in configs 
            haskey(ts1[fock], config) || continue
            overlap += dot(ts1[fock][config] , ts2[fock][config])
        end
    end
    return overlap
#=}}}=#
end

"""
    scale!(ts::FermiCG.CompressedTuckerState, a::T<:Number)

Scale `ts` by a constant
"""
function scale!(ts::CompressedTuckerState, a)
    #={{{=#
    for (fock,configs) in ts
        for (config,tuck) in configs 
            ts[fock][config].core .*= a 
        end
    end
    #=}}}=#
end

