using Profile
using LinearMaps
using BenchmarkTools
using IterativeSolvers
#using TensorDecompositions



"""
    core::Array{T, N}
    factors::NTuple{N, Matrix{T}}

Tucker factors are stored as Tall matrices
"""
struct Tucker{T, N} 
    core::Array{T, N}
    factors::NTuple{N, Matrix{T}}
    #props::Dict{Symbol, Any}
end

Tucker{T,N}() where {T,N} = Tucker{T,N}(Array{T}(undef), NTuple{N, Matrix{T}}[]) 
#Tucker{T,N}() where {T,N} = Tucker{T,N}(Array{Float64}[], NTuple{Int, Int}[]) 
#Tucker(v; thresh=-1, max_number=nothing) = Tucker(tucker_decompose(v, thresh=thresh, max_number=max_number)..., Dict())
function Tucker(A::Array{T,N}; thresh=-1, max_number=nothing, verbose=0) where {T,N}
    core,factors = tucker_decompose(A, thresh=thresh, max_number=max_number, verbose=verbose)
    return Tucker{T,N}(core, NTuple{N}(factors))
end
recompose(t::Tucker{T,N}) where {T<:Number, N} = tucker_recompose(t.core, t.factors)
dims_large(t::Tucker{T,N}) where {T<:Number, N} = return [size(f,1) for f in t.factors]
dims_small(t::Tucker{T,N}) where {T<:Number, N} = return [size(f,2) for f in t.factors]
Base.length(t::Tucker) = prod(dims_small(t))
Base.size(t::Tucker) = size(t.core) 
function Base.permutedims(t::Tucker{T,N}, perm) where {T,N}
    #t.core .= permutedims(t.core, perm)
    return Tucker{T,N}(permutedims(t.core, perm), t.factors[perm])
end

"""
    dot(t1::Tucker{T,N}, t2::Tucker{T,N}) where {T,N}

Note: This doesn't assume `t1` and `t2` have the same compression vectors 
"""
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
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
"""
struct CompressedTuckerState <: AbstractState 
    clusters::Vector{Cluster}
    data::OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
end
Base.haskey(ts::CompressedTuckerState, i) = return haskey(ts.data,i)
Base.getindex(ts::CompressedTuckerState, i) = return ts.data[i]
Base.setindex!(ts::CompressedTuckerState, i, j) = return ts.data[j] = i
Base.iterate(ts::CompressedTuckerState, state=1) = iterate(ts.data, state)


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

    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker}}()
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




"""
    add!(ts1::CompressedTuckerState, ts2::CompressedTuckerState)

Add coeffs in `ts2` to `ts1` 

Note: this assumes `t1` and `t2` have the same compression vectors
"""
function add!(ts1::CompressedTuckerState, ts2::CompressedTuckerState)
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

"""
    get_map(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; shift = nothing)
    #={{{=#
    iters = 0
   
    dim = length(ci_vector)
    function mymatvec(v)
        iters += 1
        
        set_vector!(ci_vector, v)
        
        #fold!(ci_vector)
        sig = deepcopy(ci_vector)
        zero!(sig)
        build_sigma!(sig, ci_vector, cluster_ops, clustered_ham)

        #unfold!(ci_vector)
        
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
function tucker_ci_solve!(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    Hmap = get_map(ci_vector, cluster_ops, clustered_ham)

    v0 = get_vector(ci_vector)
    nr = size(v0)[2] 
    
    davidson = Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=1e-5)
    #Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    e,v = FermiCG.solve(davidson)
    set_vector!(ci_vector,v)
    return e,v
end
#=}}}=#




"""
    build_sigma!(sigma_vector::CompressedTuckerState, ci_vector::CompressedTuckerState, cluster_ops, clustered_ham)
"""
function build_sigma!(sigma_vector::CompressedTuckerState, ci_vector::CompressedTuckerState, cluster_ops, clustered_ham)
    #={{{=#

    for (fock_bra, configs_bra) in sigma_vector
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket
                

                    for term in clustered_ham[fock_trans]
                    
                        #term isa ClusteredTerm4B || continue
                       
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
"""
function form_sigma_block!(term::ClusteredTerm1B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N}) where {T,N}
#={{{=#
    #display(term)
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

    op1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]

    #
    # Get 1body operator and compress it using the cluster's Tucker factors
    op = bra_coeffs.factors[c1.idx]' * (op1[bra[c1.idx],ket[c1.idx]] * ket_coeffs.factors[c1.idx])

    # 
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has 
    # distinct Tucker factors
    overlaps = Vector{Array{T}}() 
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue

        # if overlap not just scalar, form and prepare for contraction
        if size(bra_coeffs.factors[ci],2) > 1 || size(ket_coeffs.factors[ci],2) > 1
            push!(overlaps, bra_coeffs.factors[ci]' * ket_coeffs.factors[ci])
            push!(indices, [-ci, ci])
            state_indices[ci] = ci
        else
            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]
            length(S) == 1 || error(" huh?")
            s *= S[1]
        end
    end


    # 
    # Let's try @ncon for dynamically determined contractions, which "hopefully" avoid transposes

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [-c1.idx, c1.idx]
        state_indices[c1.idx] = c1.idx
        push!(overlaps, op)
        push!(indices, op_indices)
    end

    push!(overlaps, ket_coeffs.core)
    push!(indices, state_indices)
    
    length(overlaps) == length(indices) || error(" mismatch between operators and indices")
    if length(overlaps) == 1
        bra_coeffs.core .+= ket_coeffs.core .* s
    else
        #display.(("a", size.(overlaps), indices))
        out = @ncon(overlaps, indices)
        bra_coeffs.core .+= out .* s
    end

    return  
#=}}}=#
end

"""
"""
function form_sigma_block!(term::ClusteredTerm2B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N}) where {T,N}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
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

    # 
    # make sure active clusters are correct transitions 
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    
    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first
    
    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g., 
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ul = bra_coeffs.factors[c1.idx]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #g1 = @ncon([gamma1, U1, U2], [[-1,2,3], [2,-2], [3,-3]])
    
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ul = bra_coeffs.factors[c2.idx]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #g2 = @ncon([gamma2, U1, U2], [[-1,2,3], [2,-2], [3,-3]])
    #display(("g1/2", size(g1), size(g2)))

    # 
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
#    cache_key = (fock_bra[c1.idx], fock_bra[c2.idx], fock_ket[c1.idx], fock_ket[c2.idx], bra[c1.idx], bra[c2.idx], ket[c1.idx], ket[c2.idx])
#    if haskey(term.cache, cache_key)
#        op = term.cache[cache_key]
#    else
#        @tensor begin
#            op[q,J,I] := term.ints[p,q] * g1[p,I,J]
#            op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
#        end
#        term.cache[cache_key] = op
#    end
    @tensor begin
        op[q,J,I] := term.ints[p,q] * g1[p,I,J]
        op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
    end

    # 
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has 
    # distinct Tucker factors
    overlaps = Vector{Array{T}}() 
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue

        # if overlap not just scalar, form and prepare for contraction
        if size(bra_coeffs.factors[ci],2) > 1 || size(ket_coeffs.factors[ci],2) > 1
            push!(overlaps, bra_coeffs.factors[ci]' * ket_coeffs.factors[ci])
            push!(indices, [-ci, ci])
            state_indices[ci] = ci
        else
            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]
            length(S) == 1 || error(" huh?")
            s *= S[1]
        end
    end


    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, -c1.idx, -c2.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        push!(overlaps, op)
        push!(indices, op_indices)
    end

    push!(overlaps, ket_coeffs.core)
    push!(indices, state_indices)
   
    length(overlaps) == length(indices) || error(" mismatch between operators and indices")
    if length(overlaps) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_coeffs.core .+= ket_coeffs.core .* s 
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        out = @ncon(overlaps, indices)
        bra_coeffs.core .+= out .* s
    end

    return  
#=}}}=#
end

"""
"""
function form_sigma_block!(term::ClusteredTerm3B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N}) where {T,N}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
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

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    
    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g., 
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ul = bra_coeffs.factors[c1.idx]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ul = bra_coeffs.factors[c2.idx]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #display(("g1/2", size(g1), size(g2)))

    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    Ul = bra_coeffs.factors[c3.idx]
    Ur = ket_coeffs.factors[c3.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma3[p,I,J]
        g3[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end

    # 
    # Now contract into 3body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
#    cache_key = (fock_bra[c1.idx], fock_bra[c2.idx], fock_ket[c1.idx], fock_ket[c2.idx], bra[c1.idx], bra[c2.idx], ket[c1.idx], ket[c2.idx])
#    if haskey(term.cache, cache_key)
#        op = term.cache[cache_key]
#    else
#        @tensor begin
#            op[q,J,I] := term.ints[p,q] * g1[p,I,J]
#            op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
#        end
#        term.cache[cache_key] = op
#    end
    @tensor begin
        op[q,r,I,J] := term.ints[p,q,r] * g1[p,I,J]
        op[r,I,J,K,L] := op[q,r,I,J] * g2[q,K,L]  
        op[J,L,N,I,K,M] := op[r,I,J,K,L] * g3[r,M,N]  
    end

    # 
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has 
    # distinct Tucker factors
    overlaps = Vector{Array{T}}() 
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue

        # if overlap not just scalar, form and prepare for contraction
        if size(bra_coeffs.factors[ci],2) > 1 || size(ket_coeffs.factors[ci],2) > 1
            push!(overlaps, bra_coeffs.factors[ci]' * ket_coeffs.factors[ci])
            push!(indices, [-ci, ci])
            state_indices[ci] = ci
        else
            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]
            length(S) == 1 || error(" huh?")
            s *= S[1]
        end
    end


    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, c3.idx, -c1.idx, -c2.idx, -c3.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        push!(overlaps, op)
        push!(indices, op_indices)
    end

    push!(overlaps, ket_coeffs.core)
    push!(indices, state_indices)
   
    length(overlaps) == length(indices) || error(" mismatch between operators and indices")
    if length(overlaps) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_coeffs.core .+= ket_coeffs.core .* s 
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        out = @ncon(overlaps, indices)
        bra_coeffs.core .+= out .* s
    end

    return  
#=}}}=#
end

"""
"""
function form_sigma_block!(term::ClusteredTerm4B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N}) where {T,N}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
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

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    
    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g., 
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ul = bra_coeffs.factors[c1.idx]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ul = bra_coeffs.factors[c2.idx]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #display(("g1/2", size(g1), size(g2)))

    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    Ul = bra_coeffs.factors[c3.idx]
    Ur = ket_coeffs.factors[c3.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma3[p,I,J]
        g3[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end

    gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
    Ul = bra_coeffs.factors[c4.idx]
    Ur = ket_coeffs.factors[c4.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma4[p,I,J]
        g4[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end

    # 
    # Now contract into 4body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
#    cache_key = (fock_bra[c1.idx], fock_bra[c2.idx], fock_ket[c1.idx], fock_ket[c2.idx], bra[c1.idx], bra[c2.idx], ket[c1.idx], ket[c2.idx])
#    if haskey(term.cache, cache_key)
#        op = term.cache[cache_key]
#    else
#        @tensor begin
#            op[q,J,I] := term.ints[p,q] * g1[p,I,J]
#            op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
#        end
#        term.cache[cache_key] = op
#    end
    @tensor begin
        op[q,r,s,J,I] := term.ints[p,q,r,s] * g1[p,I,J]  
        op[r,s,J,L,I,K] := op[q,r,s,J,I] * g2[q,K,L]  
        op[s,J,L,N,I,K,M] := op[r,s,J,L,I,K] * g3[r,M,N]  
        op[J,L,N,P,I,K,M,O] := op[s,J,L,N,I,K,M] * g4[s,O,P]  
    end

    # 
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has 
    # distinct Tucker factors
    overlaps = Vector{Array{T}}() 
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue
        ci != c4.idx || continue

        # if overlap not just scalar, form and prepare for contraction
        if size(bra_coeffs.factors[ci],2) > 1 || size(ket_coeffs.factors[ci],2) > 1
            push!(overlaps, bra_coeffs.factors[ci]' * ket_coeffs.factors[ci])
            push!(indices, [-ci, ci])
            state_indices[ci] = ci
        else
            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]
            length(S) == 1 || error(" huh?")
            s *= S[1]
        end
    end


    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, c3.idx, c4.idx, -c1.idx, -c2.idx, -c3.idx, -c4.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        state_indices[c4.idx] = c4.idx
        push!(overlaps, op)
        push!(indices, op_indices)
    end

    push!(overlaps, ket_coeffs.core)
    push!(indices, state_indices)
   
    length(overlaps) == length(indices) || error(" mismatch between operators and indices")
    if length(overlaps) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_coeffs.core .+= ket_coeffs.core .* s 
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        out = @ncon(overlaps, indices)
        bra_coeffs.core .+= out .* s
    end

    return  
#=}}}=#
end



"""
0 = <x|H - E0|x'>v(x') + <x|H - E0|p>v(p) 
0 = <x|H - E0|x'>v(x') + <x|H|p>v(p) 
A(x,x')v(x') = -H(x,p)v(p)

here, x is outside the reference space, and p is inside

Ax=b

works for one root at a time
"""
function tucker_cepa_solve!(ref_vector::CompressedTuckerState, ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    sig = deepcopy(ref_vector) 
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham)
    e0 = dot(ref_vector, sig)
    length(e0) == 1 || error("Only one state at a time please", e0)
    e0 = e0[1]
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
        set_vector!(x_vector, v)
        sig = deepcopy(x_vector)
        zero!(sig)
        build_sigma!(sig, x_vector, cluster_ops, clustered_ham)
        
        sig_out = get_vector(sig)
        sig_out .-= e0*get_vector(x_vector)
        return sig_out
    end
    dim = length(x_vector)
    Axx = LinearMap(mymatvec, dim, dim)
    #Axx = LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
    
    x, solver = cg!(get_vector(x_vector), Axx,bv,log=true)

    set_vector!(x_vector, x)
   
    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = dot(sig,ref_vector)
    length(ecorr) == 1 || error(" Dimension Error", ecorr)
    ecorr = ecorr[1]
  
    zero!(ci_vector)
    add!(ci_vector, ref_vector)
    add!(ci_vector, x_vector)

    #x, info = linsolve(Hmap,zeros(size(v0)))
    return ecorr+e0, x
end#=}}}=#


"""
    define_foi_space(v::CompressedTuckerState, clustered_ham; nbody=2)
Compute the first-order interacting space as defined by clustered_ham

#Arguments
- `v::CompressedTuckerState`: input state
- `clustered_ham`: Hamiltonian
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `foi::OrderedDict{FockConfig,Vector{TuckerConfig}}`

"""
function define_foi_space(cts::T, clustered_ham; nbody=2) where T<:Union{TuckerState, CompressedTuckerState}
    println(" Define the FOI space for CompressedTuckerState. nbody = ", nbody)#={{{=#

    foi_space = OrderedDict{FockConfig,Vector{TuckerConfig}}()

    for (fock, tconfigs) in cts 
            
        for (fock_trans, terms) in clustered_ham

            # 
            # new fock sector configuration
            new_fock = fock + fock_trans


            # 
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in cts.clusters
                if new_fock[ci.idx][1] > length(ci) || new_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if new_fock[ci.idx][1] < 0 || new_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue

            

            #
            # find the cluster state index ranges (TuckerConfigs) reached by Hamiltonian
            for (tconfig, coeffs) in tconfigs 
                for term in terms
                    new_tconfig = deepcopy(tconfig)

                    length(term.clusters) <= nbody || continue
            
                    new_tconfigs = []
                    tmp = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        new_tconfig[ci.idx] = cts.q_spaces[ci.idx][new_fock[ci.idx]]
                        tmp2 = []
                        if haskey(cts.p_spaces[ci.idx], new_fock[ci.idx])
                            push!(tmp2, cts.p_spaces[ci.idx][new_fock[ci.idx]])
                        end
                        if haskey(cts.q_spaces[ci.idx], new_fock[ci.idx])
                            push!(tmp2, cts.q_spaces[ci.idx][new_fock[ci.idx]])
                        end
                        push!(tmp, tmp2)
                    end
                    
                    for prod in product(tmp...)
                        new_tconfig = deepcopy(tconfig)
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            new_tconfig[ci.idx] = prod[cidx]
                        end
                        push!(new_tconfigs, new_tconfig)
                    end
                    
                    if haskey(foi_space, new_fock)
                        foi_space[new_fock] = unique((foi_space[new_fock]..., new_tconfigs...))
                    else
                        foi_space[new_fock] = new_tconfigs
                    end
                    
                    
                end
            end
        end
    end
    return foi_space
#=}}}=#
end



"""
|psi> = sum_f,t c(f,t)|f,t>
"""
function expand_compressed_space(foi_space, cts::CompressedTuckerState, cluster_ops, clustered_ham;
                                thresh=1e-7, max_number=nothing)

    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Vector{Tucker}}}()
    for (fock_bra,tconfigs_bra) in foi_space
        for (fock_ket,tconfigs_ket) in cts 
            fock_trans = fock_bra - fock_ket
            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue
            for tconfig_bra in tconfigs_bra
                for (tconfig_ket, tuck_ket) in tconfigs_ket
                    for term in clustered_ham[fock_trans]
                        
                        term isa ClusteredTerm2B || continue

                        check = FermiCG.check_term(term, fock_bra, tconfig_bra, fock_ket, tconfig_ket)

                        check == true || continue

                        new_tuck = FermiCG.form_sigma_block_expand(term, cluster_ops, 
                                                  fock_bra, tconfig_bra, 
                                                  fock_ket, tconfig_ket, tuck_ket,
                                                  thresh=thresh, max_number=max_number)

                        length(new_tuck) > 0 || continue


                        if haskey(data, fock_bra)
                            if haskey(data[fock_bra], tconfig_bra)
                                push!(data[fock_bra][tconfig_bra], new_tuck)
                            else
                                data[fock_bra][tconfig_bra] = [new_tuck]
                            end
                        else
                            data[fock_bra] = OrderedDict(tconfig_bra => [new_tuck])
                        end
                    end
                end
            end
        end
    end
    return data
end



function check_term(term::ClusteredTerm1B, 
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig)
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    return true
end

function check_term(term::ClusteredTerm2B, 
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig)
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    return true
end

function check_term(term::ClusteredTerm3B, 
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig)
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue
        ci != term.clusters[3].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    return true
end

function check_term(term::ClusteredTerm4B, 
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig)
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue
        ci != term.clusters[3].idx || continue
        ci != term.clusters[4].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    return true
end

"""
"""
function form_sigma_block_expand(term::ClusteredTerm2B, 
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig, 
                            fock_ket::FockConfig, ket::TuckerConfig,
                            ket_coeffs::Tucker{T,N}; 
                            thresh=1e-7, max_number=nothing) where {T,N}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    n_clusters = length(bra)

    # 
    # make sure active clusters are correct transitions 
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    
    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first
    
    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g., 
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end
    
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    # 
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,J,I] := term.ints[p,q] * g1[p,I,J]
        op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
    end

    # 
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has 
    # distinct Tucker factors
    tensors = Vector{Array{T}}() 
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, -c1.idx, -c2.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end

    push!(tensors, ket_coeffs.core)
    push!(indices, state_indices)
   
    length(tensors) == length(indices) || error(" mismatch between operators and indices")

    bra_core = Array{T,N}(undef,(0,0,0))
    if length(tensors) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_core = ket_coeffs.core .* s 
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        out = @ncon(tensors, indices)
        bra_core = out .* s
    end

    # first decompose the already partially decomposed core tensor
    #
    # Vket ~ Aket x U1 x U2 x ...
    #
    # then we modified the compressed coefficients in the ket, Aket
    #
    # to get Abra, which we then compress again. 
    #
    # The active cluster tucker factors become identity matrices
    #
    # Abra ~ Bbra x UU1 x UU2 x .... 
    #
    #
    # e.g, V(IJK) = C(ijk) * U1(Ii) * U2(Jj) * U3(Kk)
    # 
    # then C get's modified and furhter compressed
    #
    # V(IJK) = C(abc) * U1(ia) * U2(jb) * U3(kc) * U1(Ii) * U2(Jj) * U3(Kk)
    # V(IJK) = C(abc) * (U1(ia) * U1(Ii)) * (U2(jb) * U2(Jj)) * (U3(kc) * U3(Kk))
    # V(IJK) = C(abc) * U1(Ia)  * U2(Jb) * U3(Kc) 
    #
    bra_tuck = Tucker(bra_core, thresh=thresh, max_number=max_number)
    
    old_factors = deepcopy(ket_coeffs.factors)

    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue

        bra_tuck.factors[ci] .= old_factors[ci] * bra_tuck.factors[ci]

    end
    return bra_tuck 
#=}}}=#
end

