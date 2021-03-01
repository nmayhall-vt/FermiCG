using Profile
using LinearMaps
using BenchmarkTools
using IterativeSolvers
#using TensorDecompositions





"""
    compress(ts::CompressedTuckerState; thresh=-1, max_number=nothing)

- `ts::TuckerState`
- `thresh`: threshold for compression
- `max_number`: only keep certain number of vectors per TuckerConfig
"""
function compress(ts::CompressedTuckerState{T,N}; thresh=-1, max_number=nothing, verbose=0) where {T,N}
    d = OrderedDict{FockConfig, OrderedDict{TuckerConfig, Tucker{T,N}}}() 
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

"""
    get_map(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; shift = nothing, cache=false)
    #={{{=#
    iters = 0

    dim = length(ci_vector)
    function mymatvec(v)
        iters += 1

        set_vector!(ci_vector, v)

        #fold!(ci_vector)
        sig = deepcopy(ci_vector)
        zero!(sig)
        build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)

        #unfold!(ci_vector)

        sig = get_vector(sig)

        if shift != nothing
            # this is how we do CEPA
            sig += shift * get_vector(ci_vector)
        end
        flush(stdout)

        return sig
    end
    return LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#

function tucker_ci_solve(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
  
    vec = deepcopy(ci_vector)
    normalize!(vec)
    #flush term cache
    flush_cache(clustered_ham)
    
    Hmap = get_map(vec, cluster_ops, clustered_ham, cache=true)

    v0 = get_vector(vec)
    nr = size(v0)[2]
   

    davidson = Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=tol)
    #Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    e,v = FermiCG.solve(davidson)
    set_vector!(vec,v)
    
    println(" Memory used by cache: ", mem_used_by_cache(clustered_ham))

    ##flush term cache
    #flush_cache(clustered_ham)

    return e,vec
end
#=}}}=#




"""
    build_sigma!(sigma_vector::CompressedTuckerState, ci_vector::CompressedTuckerState, cluster_ops, clustered_ham)
"""
function build_sigma!(sigma_vector::CompressedTuckerState, ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; nbody=4, cache=false)
    #={{{=#

    for (fock_bra, configs_bra) in sigma_vector
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket


                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue

                        FermiCG.form_sigma_block!(term, cluster_ops, fock_bra, config_bra,
                                                  fock_ket, config_ket,
                                                  coeff_bra, coeff_ket,
                                                  cache=cache)


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
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N};
                            cache=false ) where {T,N}
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

    op = Array{Float64}[]
    cache_key = (fock_bra, fock_ket, bra, ket)
    #cache_key = (fock_bra[c1.idx], fock_ket[c1.idx], bra[c1.idx], ket[c1.idx])
    if cache && haskey(term.cache, cache_key)
        op = term.cache[cache_key]
    else
        op1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]

        #
        # Get 1body operator and compress it using the cluster's Tucker factors
        op = bra_coeffs.factors[c1.idx]' * (op1[bra[c1.idx],ket[c1.idx]] * ket_coeffs.factors[c1.idx])
        if cache
            term.cache[cache_key] = op
        end
    end

    use_ncon = false
    if use_ncon
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
    else
        #
        # Use blas with transposes

        #
        # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
        # distinct Tucker factors
        overlaps = Dict{Int,Matrix{T}}()
        s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions
        for ci in 1:n_clusters
            ci != c1.idx || continue

            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]

            # if overlap not just scalar, form and prepare for contraction
            # if it is a scalar, then just update the sign with the value
            if length(S) == 1
                s *= S[1]
            else
                overlaps[ci] = S
            end
        end


        indices = collect(1:n_clusters)
        indices[c1.idx] = 0
        perm,_ = bubble_sort(indices)

        bra_coeffs2 = copy(bra_coeffs.core)
        ket_coeffs2 = copy(ket_coeffs.core)

        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(bra_coeffs2) < length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end
        ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        ket_coeffs2 = permutedims(ket_coeffs2, perm)
        bra_coeffs2 = permutedims(bra_coeffs2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(ket_coeffs2)
        ket_coeffs2 = reshape(ket_coeffs2, dim1[1], prod(dim1[2:end]))
        dim2 = size(bra_coeffs2)
        bra_coeffs2 = reshape(bra_coeffs2, dim2[1], prod(dim2[2:end]))

        #
        # Reshape Hamiltonian term operator
        # ... not needed for 1b term

        #
        # Multiply
        bra_coeffs2 .+= s .* (op * ket_coeffs2)

        # now untranspose
        perm,_ = bubble_sort(perm)
        bra_coeffs2 = reshape(bra_coeffs2, dim2)
        bra_coeffs2 = permutedims(bra_coeffs2,perm)

        #
        # multiply by overlaps now if the bra side is larger,
        if length(bra_coeffs2) >= length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end

        bra_coeffs.core .= bra_coeffs2
    end

    return
#=}}}=#
end

function _compress_local_operator(gamma, Ul::Matrix{T}, Ur::Matrix{T}) where T
# this is way slower than @tensor
#={{{=#
    # gamma has 3 indices (orbital indices, cluster indices (left), cluster indices (right)

    #
    # out(i,jp) = gamma(p,I,J) Ul(I,i)
    out = Ul' * reshape(permutedims(gamma, [2,3,1]), size(gamma,2), size(gamma,3)*size(gamma,1))

    #
    # out(j,pi) = out(J,pi) Ur(J,j)
    out = Ur' * reshape(out', size(gamma,3), size(gamma,1)*size(Ul,2)) 
    
    # out(j,pi) -> out(p,i,j)
    return reshape(out', size(gamma,1), size(Ul,2), size(Ur,2))


#    # out(i,pJ) = gamma(I,pJ) U(I,i)
#    out = Ul' * unfold(gamma, 2) 
#    # out(ip,J) 
#    out = reshape(out, size(out,1) * size(gamma,1), size(gamma,3))
#    
#    # out(ip,j) = gamma(ip,J) U(J,j)
#    out = out * Ur
#    # out(i,p,j) 
#    out = reshape(out, size(Ul,2), size(gamma,1), size(Ur,2))
#
#    # out(p,i,j)
#    return permutedims(out, [2,1,3])
end
#=}}}=#


"""
"""
function form_sigma_block!(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N};
                            cache=false) where {T,N}
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

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first
    #
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    cache_key = OperatorConfig((fock_bra, fock_ket, bra, ket))
    if cache && haskey(term.cache, cache_key)
        op = term.cache[cache_key]
    else
        #
        # Compress Gammas using the cluster's Tucker factors
        # e.g.,
        #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
        @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
        Ul = bra_coeffs.factors[c1.idx]
        Ur = ket_coeffs.factors[c1.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
            g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end
        #g1 = _compress_local_operator(gamma1, Ul, Ur)
        #g1 = @ncon([gamma1, U1, U2], [[-1,2,3], [2,-2], [3,-3]])

        @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
        Ul = bra_coeffs.factors[c2.idx]
        Ur = ket_coeffs.factors[c2.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
            g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end
        #g2 = @ncon([gamma2, U1, U2], [[-1,2,3], [2,-2], [3,-3]])
        #display(("g1/2", size(g1), size(g2)))

        @tensor begin
            op[q,J,I] := term.ints[p,q] * g1[p,I,J]
            op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
        end
        if cache
            term.cache[cache_key] = op
        end
    end

    use_ncon = false
    if use_ncon
#        #
#        # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
#        # distinct Tucker factors
#        overlaps = Vector{Array{T}}()
#        indices = Vector{Vector{Int16}}()
#        state_indices = -collect(1:n_clusters)
#        s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
#        for ci in 1:n_clusters
#            ci != c1.idx || continue
#            ci != c2.idx || continue
#
#            # if overlap not just scalar, form and prepare for contraction
#            if size(bra_coeffs.factors[ci],2) > 1 || size(ket_coeffs.factors[ci],2) > 1
#                push!(overlaps, bra_coeffs.factors[ci]' * ket_coeffs.factors[ci])
#                push!(indices, [-ci, ci])
#                state_indices[ci] = ci
#            else
#                S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]
#                length(S) == 1 || error(" huh?")
#                s *= S[1]
#            end
#        end
#
#
#        # if the compressed operator becomes a scalar, treat it as such
#        if length(op) == 1
#            s *= op[1]
#        else
#            op_indices = [c1.idx, c2.idx, -c1.idx, -c2.idx]
#            state_indices[c1.idx] = c1.idx
#            state_indices[c2.idx] = c2.idx
#            push!(overlaps, op)
#            push!(indices, op_indices)
#        end
#
#        push!(overlaps, ket_coeffs.core)
#        push!(indices, state_indices)
#
#        length(overlaps) == length(indices) || error(" mismatch between operators and indices")
#        #
#        # Use ncon
#        if length(overlaps) == 1
#            # this means that all the overlaps and the operator is a scalar
#            bra_coeffs.core .+= ket_coeffs.core .* s
#        else
#            #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
#            #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
#            out = @ncon(overlaps, indices)
#            bra_coeffs.core .+= out .* s
#        end

    else
        #
        # Use blas with transposes

        #
        # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
        # distinct Tucker factors
        overlaps = Dict{Int,Matrix{T}}()
        s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
        for ci in 1:n_clusters
            ci != c1.idx || continue
            ci != c2.idx || continue

            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]

            # if overlap not just scalar, form and prepare for contraction
            if length(S) == 1
                s *= S[1]
            else
                overlaps[ci] = S
            end
        end


        indices = collect(1:n_clusters)
        indices[c1.idx] = 0
        indices[c2.idx] = 0
        perm,_ = bubble_sort(indices)

        bra_coeffs2 = copy(bra_coeffs.core)
        ket_coeffs2 = copy(ket_coeffs.core)

        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(bra_coeffs2) < length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end
            ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        ket_coeffs2 = permutedims(ket_coeffs2, perm)
        bra_coeffs2 = permutedims(bra_coeffs2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(ket_coeffs2)
        ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2], prod(dim1[3:end]))
        dim2 = size(bra_coeffs2)
        bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2], prod(dim2[3:end]))

        #
        # Reshape Hamiltonian term operator
        op = reshape(op, prod(size(op)[1:2]), prod(size(op)[3:4]))

        #display(term)
        ##display(overlaps)
        #display((size(bra_coeffs2), size(bra_coeffs.core)))
        #display(size(op'))
        #display((size(ket_coeffs2), size(ket_coeffs.core)))
        bra_coeffs2 .+= s .* (op' * ket_coeffs2)

        bra_coeffs2 = reshape(bra_coeffs2, dim2)

        # now untranspose
        perm,_ = bubble_sort(perm)
        bra_coeffs2 = permutedims(bra_coeffs2,perm)

        #
        # multiply by overlaps now if the bra side is larger,
        if length(bra_coeffs2) >= length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end

        bra_coeffs.core .= bra_coeffs2
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
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N};
                            cache=false) where {T,N}
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


    op = Array{Float64}[]
    cache_key = OperatorConfig((fock_bra, fock_ket, bra, ket))
    if cache && haskey(term.cache, cache_key)
        op = term.cache[cache_key]
    else
        #
        # Compress Gammas using the cluster's Tucker factors
        # e.g.,
        #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
        @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
        Ul = bra_coeffs.factors[c1.idx]
        Ur = ket_coeffs.factors[c1.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
            g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end

        @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
        Ul = bra_coeffs.factors[c2.idx]
        Ur = ket_coeffs.factors[c2.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
            g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end
        #display(("g1/2", size(g1), size(g2)))

        @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
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
        @tensor begin
            op[q,r,I,J] := term.ints[p,q,r] * g1[p,I,J]
            op[r,I,J,K,L] := op[q,r,I,J] * g2[q,K,L]
            op[J,L,N,I,K,M] := op[r,I,J,K,L] * g3[r,M,N]
        end
        if cache
            term.cache[cache_key] = op
        end
    end

    use_ncon = false
    if use_ncon
#        #
#        # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
#        # distinct Tucker factors
#        overlaps = Vector{Array{T}}()
#        indices = Vector{Vector{Int16}}()
#        state_indices = -collect(1:n_clusters)
#        s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
#        for ci in 1:n_clusters
#            ci != c1.idx || continue
#            ci != c2.idx || continue
#            ci != c3.idx || continue
#
#            # if overlap not just scalar, form and prepare for contraction
#            if size(bra_coeffs.factors[ci],2) > 1 || size(ket_coeffs.factors[ci],2) > 1
#                push!(overlaps, bra_coeffs.factors[ci]' * ket_coeffs.factors[ci])
#                push!(indices, [-ci, ci])
#                state_indices[ci] = ci
#            else
#                S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]
#                length(S) == 1 || error(" huh?")
#                s *= S[1]
#            end
#        end
#
#
#        # if the compressed operator becomes a scalar, treat it as such
#        if length(op) == 1
#            s *= op[1]
#        else
#            op_indices = [c1.idx, c2.idx, c3.idx, -c1.idx, -c2.idx, -c3.idx]
#            state_indices[c1.idx] = c1.idx
#            state_indices[c2.idx] = c2.idx
#            state_indices[c3.idx] = c3.idx
#            push!(overlaps, op)
#            push!(indices, op_indices)
#        end
#
#        push!(overlaps, ket_coeffs.core)
#        push!(indices, state_indices)
#
#        length(overlaps) == length(indices) || error(" mismatch between operators and indices")
#        if length(overlaps) == 1
#            # this means that all the overlaps and the operator is a scalar
#            bra_coeffs.core .+= ket_coeffs.core .* s
#        else
#            #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
#            #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
#            out = @ncon(overlaps, indices)
#            bra_coeffs.core .+= out .* s
#        end
    else
        #
        # Use blas with transposes

        #
        # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
        # distinct Tucker factors
        overlaps = Dict{Int,Matrix{T}}()
        s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
        for ci in 1:n_clusters
            ci != c1.idx || continue
            ci != c2.idx || continue
            ci != c3.idx || continue

            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]

            # if overlap not just scalar, form and prepare for contraction
            if length(S) == 1
                s *= S[1]
            else
                overlaps[ci] = S
            end
        end


        indices = collect(1:n_clusters)
        indices[c1.idx] = 0
        indices[c2.idx] = 0
        indices[c3.idx] = 0
        perm,_ = bubble_sort(indices)

        bra_coeffs2 = copy(bra_coeffs.core)
        ket_coeffs2 = copy(ket_coeffs.core)

        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(bra_coeffs2) < length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end
        ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        ket_coeffs2 = permutedims(ket_coeffs2, perm)
        bra_coeffs2 = permutedims(bra_coeffs2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(ket_coeffs2)
        ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2]*dim1[3], prod(dim1[4:end]))
        dim2 = size(bra_coeffs2)
        bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2]*dim2[3], prod(dim2[4:end]))

        #
        # Reshape Hamiltonian term operator
        op = reshape(op, prod(size(op)[1:3]), prod(size(op)[4:6]))

        #
        # Multiply
        bra_coeffs2 .+= s .* (op' * ket_coeffs2)

        # now untranspose
        perm,_ = bubble_sort(perm)
        bra_coeffs2 = reshape(bra_coeffs2, dim2)
        bra_coeffs2 = permutedims(bra_coeffs2,perm)

        #
        # multiply by overlaps now if the bra side is larger,
        if length(bra_coeffs2) >= length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end

        bra_coeffs.core .= bra_coeffs2
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
                            bra_coeffs::Tucker{T,N}, ket_coeffs::Tucker{T,N};
                            cache=false) where {T,N}
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

    op = Array{Float64}[]
    cache_key = OperatorConfig((fock_bra, fock_ket, bra, ket))
    if cache && haskey(term.cache, cache_key)
        op = term.cache[cache_key]
    else

        #
        # Compress Gammas using the cluster's Tucker factors
        # e.g.,
        #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
        @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
        Ul = bra_coeffs.factors[c1.idx]
        Ur = ket_coeffs.factors[c1.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
            g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end

        @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
        Ul = bra_coeffs.factors[c2.idx]
        Ur = ket_coeffs.factors[c2.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
            g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end
        #display(("g1/2", size(g1), size(g2)))

        @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
        Ul = bra_coeffs.factors[c3.idx]
        Ur = ket_coeffs.factors[c3.idx]
        @tensor begin
            tmp[p,k,J] := Ul[I,k] * gamma3[p,I,J]
            g3[p,k,l] := Ur[J,l] * tmp[p,k,J]
        end

        @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
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
        @tensor begin
            op[q,r,s,J,I] := term.ints[p,q,r,s] * g1[p,I,J]
            op[r,s,J,L,I,K] := op[q,r,s,J,I] * g2[q,K,L]
            op[s,J,L,N,I,K,M] := op[r,s,J,L,I,K] * g3[r,M,N]
            op[J,L,N,P,I,K,M,O] := op[s,J,L,N,I,K,M] * g4[s,O,P]
        end
        if cache
            term.cache[cache_key] = op
        end
    end

    use_ncon = false
    if use_ncon
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
    else
        #
        # Use blas with transposes

        #
        # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
        # distinct Tucker factors
        overlaps = Dict{Int,Matrix{T}}()
        s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
        for ci in 1:n_clusters
            ci != c1.idx || continue
            ci != c2.idx || continue
            ci != c3.idx || continue
            ci != c4.idx || continue

            S = bra_coeffs.factors[ci]' * ket_coeffs.factors[ci]

            # if overlap not just scalar, form and prepare for contraction
            if length(S) == 1
                s *= S[1]
            else
                overlaps[ci] = S
            end
        end


        indices = collect(1:n_clusters)
        indices[c1.idx] = 0
        indices[c2.idx] = 0
        indices[c3.idx] = 0
        indices[c4.idx] = 0
        perm,_ = bubble_sort(indices)

        bra_coeffs2 = copy(bra_coeffs.core)
        ket_coeffs2 = copy(ket_coeffs.core)

        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(bra_coeffs2) < length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end
        ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        ket_coeffs2 = permutedims(ket_coeffs2, perm)
        bra_coeffs2 = permutedims(bra_coeffs2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(ket_coeffs2)
        ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2]*dim1[3]*dim1[4], prod(dim1[5:end]))
        dim2 = size(bra_coeffs2)
        bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2]*dim2[3]*dim2[4], prod(dim2[5:end]))

        #
        # Reshape Hamiltonian term operator
        op = reshape(op, prod(size(op)[1:4]), prod(size(op)[5:8]))

        #
        # Multiply
        bra_coeffs2 .+= s .* (op' * ket_coeffs2)

        # now untranspose
        perm,_ = bubble_sort(perm)
        bra_coeffs2 = reshape(bra_coeffs2, dim2)
        bra_coeffs2 = permutedims(bra_coeffs2,perm)

        #
        # multiply by overlaps now if the bra side is larger,
        if length(bra_coeffs2) >= length(ket_coeffs2)
            #ket_coeffs2 = transform_basis(ket_coeffs2, overlaps, trans=true)
        end

        bra_coeffs.core .= bra_coeffs2
    end

    return
#=}}}=#
end



"""
    tucker_cepa_solve!(ref_vector::CompressedTuckerState, cepa_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5, cache=true)

# Arguments
- `ref_vector`: Input reference state. 
- `cepa_vector`: CompressedTuckerState which defines the configurational space defining {X}. This 
should be the first-order interacting space (or some compressed version of it).
- `cluster_ops`
- `clustered_ham`
- `tol`: haven't yet set this up (NYI)
- `cache`: Should we cache the compressed H operators? Speeds up drastically, but uses lots of memory

Compute compressed CEPA.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |1 | = |I   + Sax| |1 | E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa + Hax*Cx = (1 + Sax*Cx)E
Hxa + HxxCx = SxaE + CxE

The idea for CEPA is to approximate E in the amplitude equation.
CEPA(0): E = Eref

(Hxx-Eref)*Cx = Sxa*Eref - Hxa

Ax=b

After solving, the Energy can be obtained as:
E = (Eref + Hax*Cx) / (1 + Sax*Cx)
"""
function tucker_cepa_solve(ref_vector::CompressedTuckerState, cepa_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5, cache=true, max_iter=30, verbose=false)
#={{{=#
    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham)
    e0 = nonorth_dot(ref_vector, sig)
    length(e0) == 1 || error("Only one state at a time please", e0)
    e0 = e0[1]
    @printf(" Reference Energy: %12.8f\n",e0)


    x_vector = deepcopy(cepa_vector)
    a_vector = deepcopy(ref_vector)

    #
    # Project out reference space
    for (fock,tconfigs) in x_vector 
        for (tconfig, tuck) in tconfigs
            if haskey(ref_vector, fock)
                if haskey(ref_vector[fock], tconfig)
                    ref_tuck = ref_vector[fock][tconfig]

                    ovlp = nonorth_dot(tuck, ref_tuck) / nonorth_dot(ref_tuck, ref_tuck)
                    tmp = scale(ref_tuck, -1.0 * ovlp)
                    x_vector[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
                end
            end
        end
    end
    @printf(" Overlap between <1|0>:          %8.1e\n", nonorth_dot(x_vector, ref_vector, verbose=0))

    b = deepcopy(x_vector)
    zero!(b)
    build_sigma!(b, ref_vector, cluster_ops, clustered_ham, cache=false)
    bv = -get_vector(b)

    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(x_vector)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        for (tconfig, tuck) in tconfigs
            if haskey(ref_vector, fock)
                if haskey(ref_vector[fock], tconfig)
                    ref_tuck = ref_vector[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = []
                    for i in 1:length(Sx.clusters)
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    Sx[fock][tconfig].core .= transform_basis(ref_tuck.core, overlaps)
                end
            end
        end
    end

    bv .= bv .+ get_vector(Sx)*e0

    @printf(" Norm of Sx overlap: %12.8f\n", orth_dot(Sx,Sx))

    function mymatvec(v)
        set_vector!(x_vector, v)
        #@printf(" Overlap between <1|0>:          %8.1e\n", nonorth_dot(x_vector, ref_vector, verbose=0))
        sig = deepcopy(x_vector)
        zero!(sig)
        build_sigma!(sig, x_vector, cluster_ops, clustered_ham, cache=cache)

        tmp = deepcopy(x_vector)
        scale!(tmp, -e0)
        orth_add!(sig, tmp)
        return get_vector(sig)
    end
    dim = length(x_vector)
    Axx = LinearMap(mymatvec, dim, dim)
    #Axx = LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)

    #flush term cache
    println(" Now flushing:")
    flush_cache(clustered_ham)
   
    println(" Start CEPA iterations with dimension = ", length(x_vector))
    x, solver = cg!(get_vector(x_vector), Axx,bv,log=true, maxiter=max_iter, verbose=verbose, abstol=tol)
    
    #flush term cache
    println(" Now flushing:")
    flush_cache(clustered_ham)

    set_vector!(x_vector, x)


    SxC = orth_dot(Sx,x_vector)
    @printf(" <A|X>C(X) = %12.8f\n", SxC)

    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(sig,ref_vector)
    @printf(" Cepa: %12.8f\n", ecorr)
    length(ecorr) == 1 || error(" Dimension Error", ecorr)
    ecorr = ecorr[1]

    @printf(" E(CEPA) = %12.8f\n", (e0 + ecorr)/(1+SxC))

    #x, info = linsolve(Hmap,zeros(size(v0)))
    return (ecorr+e0)/(1+SxC), x_vector 
end#=}}}=#


"""
    open_sigma(cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham; nbody=2) where {T,N}
Apply the Hamiltonian to `v` expanding into the uncompressed space.
This is done only partially, where each term is recompressed after being computed.
Lots of overhead probably from compression, but never completely uncompresses.


#Arguments
- `cts::CompressedTuckerState`: input state
- `cluster_ops`:
- `clustered_ham`: Hamiltonian
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `foi::CompressedTuckerState`

"""
function open_sigma(ket_cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham; thresh=1e-7, max_number=nothing, nbody=2) where {T,N}
    println(" Define the FOI space for CompressedTuckerState. nbody = ", nbody)
#={{{=#
    #
    # Initialize data for our output sigma, which we will convert to a
    sig_cts = CompressedTuckerState(ket_cts.clusters, OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker{T,N}} }(),  ket_cts.p_spaces, ket_cts.q_spaces)

    for (ket_fock, ket_tconfigs) in ket_cts
        for (fock_trans, terms) in clustered_ham

            #
            # new fock sector configuration
            sig_fock = ket_fock + fock_trans

            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in ket_cts.clusters
                if sig_fock[ci.idx][1] > length(ci) || sig_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if sig_fock[ci.idx][1] < 0 || sig_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue

            for (ket_tconfig, ket_tuck) in ket_tconfigs
                for term in terms

                    #
                    # only proceed if current term acts on no more than our requested max number of clusters
                    length(term.clusters) <= nbody || continue

                    #
                    # find the sig TuckerConfigs reached by applying current Hamiltonian term to ket_tconfig.
                    #
                    # For example:
                    #
                    #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                    #
                    #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                    #   have both non-zero P and Q spaces within the current fock sector, "sig_fock".
                    #
                    # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                    # available spaces, this list of which we will keep in "available".
                    #

                    available = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for ci in term.clusters
                        tmp = []
                        if haskey(ket_cts.p_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.p_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        if haskey(ket_cts.q_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.q_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        push!(available, tmp)
                    end


                    #
                    # Now loop over cartesian product of available subspaces (those in X above) and
                    # create the target TuckerConfig and then evaluate the associated terms
                    for prod in product(available...)
                        sig_tconfig = deepcopy(ket_tconfig)
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            sig_tconfig[ci.idx] = prod[cidx]
                        end

                        #
                        # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                        # let's compute the matrix element block, then compress, then add it to any existing compressed
                        # coefficient tensor for that sig TuckerConfig.
                        #
                        # Both the Compression and addition takes a fair amount of work.


                        FermiCG.check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue

                        sig_tuck = FermiCG.form_sigma_block_expand(term, cluster_ops,
                                                                    sig_fock, sig_tconfig,
                                                                    ket_fock, ket_tconfig, ket_tuck,
                                                                    thresh=thresh, max_number=max_number)

                        sig_tuck = compress(sig_tuck)
                        
                        length(sig_tuck) > 0 || continue


                        if haskey(sig_cts, sig_fock)
                            if haskey(sig_cts[sig_fock], sig_tconfig)
                                #
                                # In this case, our sigma vector already has a compressed coefficient tensor.
                                # Consequently, we need to add these two together

                                sig_tuck = add([sig_tuck, sig_cts[sig_fock][sig_tconfig]])
                                sig_tuck = compress(sig_tuck, thresh=thresh, max_number=max_number)
                                sig_cts[sig_fock][sig_tconfig] = sig_tuck

                            else
                                sig_cts[sig_fock][sig_tconfig] = sig_tuck
                            end
                        else
                            sig_cts[sig_fock] = OrderedDict(sig_tconfig => sig_tuck)
                        end

                    end

                end
            end
        end
    end
    prune_empty_TuckerConfigs!(sig_cts)
    return sig_cts
#=}}}=#
end


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


#={{{=#
"""
    expand_compressed_space(foi_space, cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham;
                                    thresh=1e-7, max_number=nothing) where {T,N}

Apply the Hamiltonian to cts, generating a `CompressedTuckerState` without ever fully uncompressing
"""
function expand_compressed_space(foi_space, cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham;
                                    thresh=1e-7, max_number=nothing) where {T,N}

    data = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Vector{Tucker{T,N}} } }()
    for (fock_bra,tconfigs_bra) in foi_space
        for (fock_ket,tconfigs_ket) in cts
            fock_trans = fock_bra - fock_ket
            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue
            for tconfig_bra in tconfigs_bra
                for (tconfig_ket, tuck_ket) in tconfigs_ket
                    for term in clustered_ham[fock_trans]

                        term isa ClusteredTerm1B || continue
                        #term isa ClusteredTerm2B || continue

                        #ok = false
                        #if term isa ClusteredTerm1B || term isa ClusteredTerm2B
                        #    ok = true
                        #end
                        #ok || continue

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

    #   Two ways to proceed.
    #   1) Orthogonalize the tucker factors to get a larger subpace
    #   2) Keep a list of Tucker objects that yeilds fewer variational parameters,
    #       but introduces non-orthogonality and extra terms inside a single TuckerConfig
    #
    #   Try 1) first
    #
    data2 = OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker{T,N}} }()
    for (fock,tconfigs) in data
        #display(fock)
        for (tconfig,tucks) in tconfigs
            #display(tconfig)

            display(typeof(tucks))
            new_tuck = add(tucks)
            display(size(new_tuck))
            new_tuck = compress(new_tuck)
            display(size(new_tuck))
            if length(new_tuck) > 0
                if haskey(data2, fock)
                    data2[fock][tconfig] = new_tuck
                else
                    data2[fock] = OrderedDict(tconfig => new_tuck)
                end
            end

        end
    end

    return CompressedTuckerState(cts.clusters, data2, cts.p_spaces, cts.q_spaces)
end
#=}}}=#


"""
"""
function form_sigma_block_expand(term::ClusteredTerm1B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            ket_coeffs::Tucker{T,N};
                            thresh=1e-7, max_number=nothing, screen=1e-8) where {T,N}
#={{{=#
    #display(term)
    c1 = term.clusters[1]
    n_clusters = length(bra)

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    #
    # Get 1body operator and compress it using the cluster's Tucker factors,
    # but since we are expanding the compression space
    # only compress the right hand side
    op1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    op =  (op1[bra[c1.idx],ket[c1.idx]] * ket_coeffs.factors[c1.idx])

#    println("Nick:")
#    display(term)
#    display(fock_bra)
#    display(fock_ket)
#    display(bra)
#    display(ket)
#    display(size(op))
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [-c1.idx, c1.idx]
        state_indices[c1.idx] = c1.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end

    push!(tensors, ket_coeffs.core)
    push!(indices, state_indices)

    length(tensors) == length(indices) || error(" mismatch between operators and indices")

    bra_core = zeros(1,1)
    if length(tensors) == 1
        bra_core = ket_coeffs.core .* s
    else
        bra_core = @ncon(tensors, indices)
        bra_core .= bra_core .* s
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

    new_factors = [ket_coeffs.factors[i] for i in 1:N]
    new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
#    println(" Output:")
#    display.(size.(new_factors))
    return Tucker(bra_core, NTuple{N}(new_factors))
    
#    bra_tuck = Tucker(bra_core, thresh=thresh, max_number=max_number)
#
#    old_factors = deepcopy(ket_coeffs.factors)
#    new_factors = [bra_tuck.factors[i] for i in 1:N]
#
#    for ci in 1:n_clusters
#        ci != c1.idx || continue
#
#        #bra_tuck.factors[ci] .= old_factors[ci] * bra_tuck.factors[ci]
#        new_factors[ci] = old_factors[ci] * bra_tuck.factors[ci]
#        #push!(new_factors, old_factors[ci] * bra_tuck.factors[ci])
#
#    end
#    return Tucker(bra_tuck.core, NTuple{N}(new_factors))
#=}}}=#
end

"""
"""
function form_sigma_block_expand(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            ket_coeffs::Tucker{T,N};
                            thresh=1e-7, max_number=nothing, screen=1e-8) where {T,N}
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

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
    
    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))

    do_screening = false 
    if do_screening 
        screen = 1e-4
        
   
        D = permutedims(g1, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor1 = F.U[:,1:nkeep]
        g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
        g1 = permutedims(g1, [2,1,3])


        D = permutedims(g2, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor2 = F.U[:,1:nkeep]
        g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
        g2 = permutedims(g2, [2,1,3])

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

    bra_core = zeros(1,1)
    if length(tensors) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_core = ket_coeffs.core .* s
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        bra_core = @ncon(tensors, indices)
        bra_core .= bra_core .* s
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
    
    new_factors = [ket_coeffs.factors[i] for i in 1:N]
    #new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
    #new_factors[c2.idx] = Matrix(1.0I, size(bra_core,c2.idx), size(bra_core,c2.idx))
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    return Tucker(bra_core, NTuple{N}(new_factors))

#    bra_tuck = Tucker(bra_core, thresh=thresh, max_number=max_number)
#
#    old_factors = deepcopy(ket_coeffs.factors)
#    new_factors = [bra_tuck.factors[i] for i in 1:N]
#
#    for ci in 1:n_clusters
#        ci != c1.idx || continue
#        ci != c2.idx || continue
#
#        #bra_tuck.factors[ci] .= old_factors[ci] * bra_tuck.factors[ci]
#        new_factors[ci] = old_factors[ci] * bra_tuck.factors[ci]
#        #push!(new_factors, old_factors[ci] * bra_tuck.factors[ci])
#
#    end
#    return Tucker(bra_tuck.core, NTuple{N}(new_factors))
#=}}}=#
end

"""
"""
function form_sigma_block_expand(term::ClusteredTerm3B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            ket_coeffs::Tucker{T,N};
                            thresh=1e-7, max_number=nothing, screen=1e-8) where {T,N}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    n_clusters = length(bra)

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    Ur = ket_coeffs.factors[c3.idx]
    @tensor begin
        g3[p,I,l] := Ur[J,l] * gamma3[p,I,J]
    end

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
    
    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))
    new_factor3 = Matrix(1.0I, size(g3,2), size(g3,2))

    do_screening = true 
    if do_screening 
        screen = 1e-4
        
   
        D = permutedims(g1, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor1 = F.U[:,1:nkeep]
        g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
        g1 = permutedims(g1, [2,1,3])


        D = permutedims(g2, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor2 = F.U[:,1:nkeep]
        g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
        g2 = permutedims(g2, [2,1,3])


        D = permutedims(g3, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        # 
        # for now, let's just keep the full space, then maybe later start threshing
        new_factor3 = F.U[:,1:nkeep]
        g3 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g3 = reshape(g3, size(g3,1), size(D,2), size(D,3))
        g3 = permutedims(g3, [2,1,3])
    end
    
    #
    # Now contract into 3body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,r,I,J] := term.ints[p,q,r] * g1[p,I,J]
    end
    @tensor begin
        op[r,I,J,K,L] := op[q,r,I,J] * g2[q,K,L]
    end
    @tensor begin
        op[J,L,N,I,K,M] := op[r,I,J,K,L] * g3[r,M,N]
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
        op_indices = [c1.idx, c2.idx, c3.idx, -c1.idx, -c2.idx, -c3.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end

    push!(tensors, ket_coeffs.core)
    push!(indices, state_indices)

    length(tensors) == length(indices) || error(" mismatch between operators and indices")

    bra_core = zeros(1,1)
    if length(tensors) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_core = ket_coeffs.core .* s
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        bra_core = @ncon(tensors, indices)
        bra_core .= bra_core .* s
    end

    new_factors = [ket_coeffs.factors[i] for i in 1:N]
    #new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
    #new_factors[c2.idx] = Matrix(1.0I, size(bra_core,c2.idx), size(bra_core,c2.idx))
    #new_factors[c3.idx] = Matrix(1.0I, size(bra_core,c3.idx), size(bra_core,c3.idx))
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    new_factors[c3.idx] = new_factor3 
    return Tucker(bra_core, NTuple{N}(new_factors))
    
#=}}}=#
end


"""
"""
function form_sigma_block_expand(term::ClusteredTerm4B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            ket_coeffs::Tucker{T,N};
                            thresh=1e-7, max_number=nothing, screen=1e-8) where {T,N}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]
    n_clusters = length(bra)

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
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ur = ket_coeffs.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = ket_coeffs.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    Ur = ket_coeffs.factors[c3.idx]
    @tensor begin
        g3[p,I,l] := Ur[J,l] * gamma3[p,I,J]
    end

    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
    Ur = ket_coeffs.factors[c4.idx]
    @tensor begin
        g4[p,I,l] := Ur[J,l] * gamma4[p,I,J]
    end
    
    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
    
    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))
    new_factor3 = Matrix(1.0I, size(g3,2), size(g3,2))
    new_factor4 = Matrix(1.0I, size(g4,2), size(g4,2))

    do_screening = true 
    if do_screening 
        
   
        D = permutedims(g1, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor1 = F.U[:,1:nkeep]
        g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
        g1 = permutedims(g1, [2,1,3])


        D = permutedims(g2, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor2 = F.U[:,1:nkeep]
        g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
        g2 = permutedims(g2, [2,1,3])


        D = permutedims(g3, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor3 = F.U[:,1:nkeep]
        g3 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g3 = reshape(g3, size(g3,1), size(D,2), size(D,3))
        g3 = permutedims(g3, [2,1,3])


        D = permutedims(g4, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si > screen
                nkeep += 1
            end
        end
        new_factor4 = F.U[:,1:nkeep]
        g4 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g4 = reshape(g4, size(g4,1), size(D,2), size(D,3))
        g4 = permutedims(g4, [2,1,3])
    end

    #
    # Now contract into 4body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,r,s,J,I] := term.ints[p,q,r,s] * g1[p,I,J]
        op[r,s,J,L,I,K] := op[q,r,s,J,I] * g2[q,K,L]
        op[s,J,L,N,I,K,M] := op[r,s,J,L,I,K] * g3[r,M,N]
        op[J,L,N,P,I,K,M,O] := op[s,J,L,N,I,K,M] * g4[s,O,P]
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
        op_indices = [c1.idx, c2.idx, c3.idx, c4.idx, -c1.idx, -c2.idx, -c3.idx, -c4.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        state_indices[c4.idx] = c4.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end

    push!(tensors, ket_coeffs.core)
    push!(indices, state_indices)

    length(tensors) == length(indices) || error(" mismatch between operators and indices")

    bra_core = zeros(1,1)
    if length(tensors) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_core = ket_coeffs.core .* s
    else
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(bra_coeffs), size(ket_coeffs), "sizes: ", overlaps, indices))
        bra_core = @ncon(tensors, indices)
        bra_core .= bra_core .* s
    end

    new_factors = [ket_coeffs.factors[i] for i in 1:N]
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    new_factors[c3.idx] = new_factor3 
    new_factors[c4.idx] = new_factor4 
    return Tucker(bra_core, NTuple{N}(new_factors))
    
#=}}}=#
end


"""
H0 = either 1 body H or sum of CMF hamiltonians
H1 = H - H0
X  = projection onto orthogonal space from |0>

In the compressed basis, our H0 is not diagonal,
so we solve for the |1> iteratively

(H0 - E0) |1> = - X * H1|0>
              = - X * H|0> + X*H0|0>
              = - H|0> + |0><0|H|0> + H0|0> - |0><0|H0|0>
              = - H|0> + |0>E_ref + H0|0> - |0>E0
              = - H|0> +  H0|0> + |0>(E_ref - E0)
              = b

E0 = <0|H0|0>
E_ref = <0|H|0>

"""
function hylleraas_compressed_mp2(sig::CompressedTuckerState, ref::CompressedTuckerState,
            cluster_ops, clustered_ham; tol=1e-6, nbody=4, max_iter=40, verbose=1, do_pt = true)
#={{{=#
    
#
    # (H0 - E0) |1> = X H |0>

    e2 = 0.0

    #
    # get |sig0> = H0|0>
    #
    # todo: currently only working for 1body hamiltonian, replace with CMF hamiltonias
    #
    sig0 = deepcopy(ref)
    zero!(sig0)
    FermiCG.build_sigma!(sig0, ref, cluster_ops, clustered_ham, nbody=1)
    e0 = orth_dot(ref,sig0)
    @printf(" <0|sig>  : %12.8f\n",nonorth_dot(ref,sig))
    @printf(" <0|H0|0>  : %12.8f\n",e0)
#
#    tmp = deepcopy(sig)
#    FermiCG.scale!(tmp, -e0)
#
#    nonorth_add!(sig, sig0)
#    nonorth_add!(sig, tmp)
    
    b = get_vector(sig)

    #
    # now we should have
    #
    #  (H0 - E0) |1> = b
    #
    #  solve with CG

    function mymatvec(x)

        xr = deepcopy(sig)
        xl = deepcopy(sig)
        set_vector!(xr,x)
        zero!(xl)
        build_sigma!(xl, xr, cluster_ops, clustered_ham, nbody=1)

        # subtract off -E0|1>
        #
        scale!(xr,-e0)
        orth_add!(xl,xr)

        return get_vector(xl)
    end

    dim = length(b)
    Axx = LinearMap(mymatvec, dim, dim)


    x_vector = zeros(dim)
    #x, solver = cg!(x_vector, Axx, b)
    x, solver = cg!(x_vector, Axx, b, log=true, maxiter=max_iter, verbose=true, abstol=tol)

    psi1 = deepcopy(sig)
    set_vector!(psi1,x_vector)
   
    if do_pt
        #
        # Compute PT2 Energy 
        tmp = deepcopy(ref)
        zero!(tmp)
        @time build_sigma!(tmp, ref, cluster_ops, clustered_ham)
        e_ref = nonorth_dot(ref, tmp)
        zero!(tmp)

        @time build_sigma!(tmp, psi1, cluster_ops, clustered_ham)
        
        # 
        # project out the overlap 
        # E2 = <0|HX|1> = <0|H|1> - <0|H|0><0|1>
        e_2 = nonorth_dot(ref, tmp) - e_ref * nonorth_dot(ref,psi1) 
        e_2 = -e_2
        @printf(" E(REF)       =                  %12.8f\n", e_ref)
        @printf(" E(PT2) corr  =                  %12.8f\n", e_2)
        @printf(" E(PT2) total =                  %12.8f\n", e_ref + e_2)
        e2 = e_ref + e_2
    end
    @printf(" <0|sig>  : %12.8f\n",nonorth_dot(ref,psi1))
        


    return psi1, e2
     
end#=}}}=#




"""
    build_compressed_1st_order_state(cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham; thresh=1e-7, max_number=nothing, nbody=2) where {T,N}
Apply the Hamiltonian to `v` expanding into the uncompressed space.
This is done only partially, where each term is recompressed after being computed.
Lots of overhead probably from compression, but never completely uncompresses.

#Arguments
- `cts::CompressedTuckerState`: input state
- `cluster_ops`:
- `clustered_ham`: Hamiltonian
- `thresh`: Threshold for Tucker decomosition
- `max_number`: max number of tucker factors kept
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `v1::CompressedTuckerState`

"""
function build_compressed_1st_order_state(ket_cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham; 
        H0_string="H", 
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4, 
        do_pt=false) where {T,N}
    println(" Compute the 1st order wavefunction for CompressedTuckerState. nbody = ", nbody)
#={{{=#
    #
    # Initialize data for our output sigma, which we will convert to a
    sig_cts = CompressedTuckerState(ket_cts.clusters, OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker{T,N}} }(),  ket_cts.p_spaces, ket_cts.q_spaces)

    data = OrderedDict{FockConfig, OrderedDict{TuckerConfig, Vector{Tucker{T,N}} } }()

    for (ket_fock, ket_tconfigs) in ket_cts
        for (fock_trans, terms) in clustered_ham

            #
            # new fock sector configuration
            sig_fock = ket_fock + fock_trans

            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in ket_cts.clusters
                if sig_fock[ci.idx][1] > length(ci) || sig_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if sig_fock[ci.idx][1] < 0 || sig_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue

            for term in terms

                #
                # only proceed if current term acts on no more than our requested max number of clusters
                length(term.clusters) <= nbody || continue
                for (ket_tconfig, ket_tuck) in ket_tconfigs

                    #
                    # find the sig TuckerConfigs reached by applying current Hamiltonian term to ket_tconfig.
                    #
                    # For example:
                    #
                    #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                    #
                    #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                    #   have both non-zero P and Q spaces within the current fock sector, "sig_fock".
                    #
                    # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                    # available spaces, this list of which we will keep in "available".
                    #

                    available = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for ci in term.clusters
                        tmp = []
                        if haskey(ket_cts.p_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.p_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        if haskey(ket_cts.q_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.q_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        push!(available, tmp)
                    end


                    #
                    # Now loop over cartesian product of available subspaces (those in X above) and
                    # create the target TuckerConfig and then evaluate the associated terms
                    for prod in product(available...)
                        sig_tconfig = [ket_tconfig.config...]
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            sig_tconfig[ci.idx] = prod[cidx]
                        end
                        sig_tconfig = TuckerConfig(sig_tconfig)

                        #
                        # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                        # let's compute the matrix element block, then compress, then add it to any existing compressed
                        # coefficient tensor for that sig TuckerConfig.
                        #
                        # Both the Compression and addition takes a fair amount of work.


                        check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue


                        sig_tuck = form_sigma_block_expand(term, cluster_ops,
                                                                sig_fock, sig_tconfig,
                                                                ket_fock, ket_tconfig, ket_tuck,
                                                                thresh=thresh, max_number=max_number)
                       
                        sig_tuck = compress(sig_tuck, thresh=thresh)

    
                        #sig_tuck = compress(sig_tuck, thresh=1e-16, max_number=max_number)

                        length(sig_tuck) > 0 || continue


                        if haskey(data, sig_fock)
                            if haskey(data[sig_fock], sig_tconfig)
                                #
                                # In this case, our sigma vector already has a compressed coefficient tensor.
                                # Consequently, we need to add these two together
                                
                                push!(data[sig_fock][sig_tconfig], sig_tuck)
                                #sig_tuck = add([sig_tuck, sig_cts[sig_fock][sig_tconfig]])
                                ##sig_tuck = compress(sig_tuck, thresh=thresh, max_number=max_number)
                                #sig_cts[sig_fock][sig_tconfig] = sig_tuck

                            else
                                data[sig_fock][sig_tconfig] = [sig_tuck]
                                #sig_cts[sig_fock][sig_tconfig] = sig_tuck
                            end
                        else
                            #sig_cts[sig_fock] = OrderedDict(sig_tconfig => sig_tuck)
                            data[sig_fock] = OrderedDict(sig_tconfig => [sig_tuck])
                        end

                    end

                end
            end
        end
    end

    for (fock,tconfigs) in data
        #display(fock)
        for (tconfig, tuck) in tconfigs
            #display(tconfig)
            #display(length(tuck))
            
            #sig_cts[fock][tconfig] = nonorth_add(tuck)
            if haskey(sig_cts, fock)
                sig_cts[fock][tconfig] = compress(nonorth_add(tuck), thresh=thresh)
                #sig_cts[fock][tconfig] = nonorth_add(tuck)
            else
                sig_cts[fock] = OrderedDict(tconfig => nonorth_add(tuck))
            end
            #sig_cts[fock][tconfig] = compress(nonorth_add(tuck), thresh=thresh)
        end
    end

    # 
    # project out A space
    for (fock,tconfigs) in sig_cts 
        for (tconfig, tuck) in tconfigs
            if haskey(ket_cts, fock)
                if haskey(ket_cts[fock], tconfig)
                    ket_tuck_A = ket_cts[fock][tconfig]

                    ovlp = nonorth_dot(tuck, ket_tuck_A) / nonorth_dot(ket_tuck_A, ket_tuck_A)
                    tmp = scale(ket_tuck_A, -1.0 * ovlp)
                    sig_cts[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
                end
            end
        end
    end
   
   
    # now combine Tuckers, project out reference space and multiply by resolvents
    #prune_empty_TuckerConfigs!(sig_cts)
    return sig_cts
#=}}}=#
end
    
    
    
"""
    solve_for_compressed_space(ref_vector::CompressedTuckerState, cluster_ops, clustered_ham;
        max_iter    = 20,
        H0          = "cmf",
        nbody       = 4,
        thresh_foi  = 1e-6,
        thresh_var  = 1e-4,
        tol_ci      = 1e-5,
        tol_tucker  = 1e-6)

# Arguments
- `ref_vector`: initial state
- `nbody`: max number of nbody terms in the Hamiltonian used for creating FOIS
- `thresh_foi`: Compression threshold for the FOIS, or first order wavefunction
- `thresh_var`: Compression threshold for the variational solution
- `tol_ci`:     Convergence threshold for the CI (norm of residual)
- `tol_tucker`: Convergence threshold for Tucker iterations (energy change)
"""
function solve_for_compressed_space(ref_vec::CompressedTuckerState, cluster_ops, clustered_ham;
        max_iter    = 20,
        nbody       = 4,
        H0          = "cmf",
        thresh_var  = 1e-4,
        thresh_foi  = 1e-6,
        thresh_pt   = 1e-5,
        tol_ci      = 1e-5,
        do_pt       = true,
        tol_tucker  = 1e-6)
      #={{{=#
    e_last = 0.0
    e_var  = 0.0
    e_pt2  = 0.0
    for iter in 1:max_iter
        println(" --------------------------------------------------------------------")
        println(" Iterate PT-Var:       Iteration #: ",iter)
        println(" --------------------------------------------------------------------")

        #
        # Compress Variational Wavefunction
        dim1 = length(ref_vec)
        norm1 = orth_dot(ref_vec, ref_vec)
        ref_vec = compress(ref_vec, thresh=thresh_var)
        normalize!(ref_vec)
        dim2 = length(ref_vec)
        norm2 = orth_dot(ref_vec, ref_vec)
        @printf(" Compressed Ref state from: %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_var)
        @printf(" Norm of compressed state: %12.8f \n", norm2)
        
        # 
        # Solve variationally in reference space
        println()
        @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
        @time e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, tol=tol_ci)
#       sig = deepcopy(ref_vec)
#       zero!(sig)
#       build_sigma!(sig, ref_vec, cluster_ops, clustered_ham)
#       e0 = orth_dot(ref_vec, sig)
        if iter == 1
            e_last = e0
        end
   
        #
        # Get First order wavefunction
        println()
        println(" Compute first order wavefunction. Reference space dim = ", length(ref_vec))
        hstr = "Hcmf"
        if H0 == "H"
            hstr = "H"
        end
        @time pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, 
                                                          nbody=nbody, thresh=thresh_foi, H0_string=hstr)

        # 
        # Compress FOIS
        norm1 = orth_dot(pt1_vec, pt1_vec)
        dim1 = length(pt1_vec)
        pt1_vec = compress(pt1_vec, thresh=thresh_foi)
        norm2 = orth_dot(pt1_vec, pt1_vec)
        dim2 = length(pt1_vec)
        @printf(" FOIS Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
        @printf(" Norm of |1>:              %12.8f \n", norm2)
        @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))

        if do_pt
            #
            # 
            println()
            println(" Compute PT vector. Reference space dim = ", length(ref_vec))
            pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol_ci, do_pt=do_pt)
            # 
            # Compress first order wavefunction 
            norm1 = orth_dot(pt1_vec, pt1_vec)
            dim1 = length(pt1_vec)
            pt1_vec = compress(pt1_vec, thresh=thresh_pt)
            norm2 = orth_dot(pt1_vec, pt1_vec)
            dim2 = length(pt1_vec)
            @printf(" PT   Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_pt)
            @printf(" Norm of |1>:              %12.8f \n", norm2)
            @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))
        end

        # 
        # Solve variationally in compressed FOIS 
        # CI
        println()
        var_vec = deepcopy(ref_vec)
        nonorth_add!(var_vec, pt1_vec)
        normalize!(var_vec)
        @printf(" Solve in compressed FOIS. Dimension =   %10i\n", length(var_vec))
        @time e_var, var_vec = tucker_ci_solve(var_vec, cluster_ops, clustered_ham, tol=tol_ci)

#        #
#        # Compress Variational Wavefunction
#        dim1 = length(var_vec)
#        norm1 = orth_dot(var_vec, var_vec)
#        var_vec = compress(var_vec, thresh=thresh_var)
#        normalize!(var_vec)
#        dim2 = length(var_vec)
#        norm2 = orth_dot(var_vec, var_vec)
#        @printf(" Compressed CI state from: %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_var)
#        @printf(" Norm of compressed state: %12.8f \n", norm2)

        ref_vec = var_vec

        @printf(" E(Ref)      = %12.8f\n", e0[1])
        do_pt == false || @printf(" E(PT2) tot  = %12.8f\n", e_pt2)
        @printf(" E(var) tot  = %12.8f\n", e_var[1])

        if abs(e_last[1] - e_var[1]) < tol_tucker 
            println("*Converged")
            return e_var, var_vec
            break
        end
        e_last = e_var
            
    end
    println(" Not converged")
    return e_var, var_vec
end
    #=}}}=#
    


function do_fois_cepa(ref::CompressedTuckerState, cluster_ops, clustered_ham;
            max_iter    = 20,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            verbose     = true)

    # 
    # Solve variationally in reference space
    println()
    ref_vec = deepcopy(ref)
    @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
    @time e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, tol=tol)

    #
    # Get First order wavefunction
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    @time pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)

#    #
#    # 
#    println()
#    println(" Compute PT vector. Reference space dim = ", length(ref_vec))
#    pt1_vec = hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, do_pt=do_pt)

    # 
    # Compress FOIS
    norm1 = orth_dot(pt1_vec, pt1_vec)
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    norm2 = orth_dot(pt1_vec, pt1_vec)
    dim2 = length(pt1_vec)
    @printf(" FOIS Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
    @printf(" Norm of |1>:              %12.8f \n", norm2)
    @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))

    #
    # Compute PT2 Energy
    sig = deepcopy(pt1_vec)
    zero!(sig)
    build_sigma!(sig, ref_vec, cluster_ops, clustered_ham)
    e_2 = nonorth_dot(pt1_vec, sig)
    @printf(" E(PT2) corr =                  %12.8f\n", e_2)

    # 
    # Solve CEPA 
    println()
    cepa_vec = deepcopy(pt1_vec)
    zero!(cepa_vec)
    println(" Do CEPA: Dim = ", length(cepa_vec))
    @time e_cepa, x_cepa = tucker_cepa_solve(ref_vec, cepa_vec, cluster_ops, clustered_ham, tol=tol, max_iter=max_iter, verbose=verbose)

    @printf(" E(cepa) corr =                 %12.8f\n", e_cepa)
    @printf(" X(cepa) norm =                 %12.8f\n", sqrt(orth_dot(x_cepa, x_cepa)))
    nonorth_add!(x_cepa, ref_vec)
    normalize!(x_cepa)
    return e_cepa, x_cepa
end


