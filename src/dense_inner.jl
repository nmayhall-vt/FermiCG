using BlockDavidson

########################################################################################################
########################################################################################################

"""
"""
function form_sigma_block!(term::ClusteredTerm1B, 
        cluster_ops::Vector{ClusterOps{T}},
        fock_bra::FockConfig, bra::TuckerConfig, 
        fock_ket::FockConfig, ket::TuckerConfig,
        bra_coeffs::Array, ket_coeffs::Array) where {T}
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
        cluster_ops::Vector{ClusterOps{T}},
        fock_bra::FockConfig, bra::TuckerConfig, 
        fock_ket::FockConfig, ket::TuckerConfig,
        bra_coeffs::Array, ket_coeffs::Array) where {T}
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
        cluster_ops::Vector{ClusterOps{T}},
        fock_bra::FockConfig, bra::TuckerConfig, 
        fock_ket::FockConfig, ket::TuckerConfig,
        bra_coeffs::Array, ket_coeffs::Array) where {T}
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


    #    if haskey(term.cache, cache_key)
    #        op = term.cache[cache_key]
    #    else
    #        @tensor begin
    #            op[q,r,I,J] := term.ints[p,q,r] * gamma1[p,I,J]
    #            op[r,I,J,K,L] := op[q,r,I,J] * gamma2[q,K,L]  
    #            op[J,L,N,I,K,M] := op[r,I,J,K,L] * gamma3[r,M,N]  
    #        end
    #        term.cache[cache_key] = op
    #  
    #        #tucker_decompose(op)
    #        # compress this
    #        opsize = size(op)
    #        op = reshape(op, prod(size(op)[1:3]), prod(size(op)[4:6]))
    #        F = svd(op)
    #        #display(F.S)
    #        cut = 0
    #        for si in 1:length(F.S) 
    #            if F.S[si] < 1e-4
    #                F.S[si] = 0
    #                cut += 1
    #            end
    #        end
    #        #if cut > 0
    #        #    display((length(F.S), cut))
    #        #end
    #        #op = F.U * Diagonal(F.S) * F.Vt
    #        op = reshape(op,opsize)
    #        core, factors = tucker_decompose(op, thresh=-1, verbose=0)
    #        op = tucker_recompose(core, factors)
    #    end

    @tensor begin
        op[q,r,I,J] := term.ints[p,q,r] * gamma1[p,I,J]
        op[r,I,J,K,L] := op[q,r,I,J] * gamma2[q,K,L]  
        op[J,L,N,I,K,M] := op[r,I,J,K,L] * gamma3[r,M,N]  
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
        cluster_ops::Vector{ClusterOps{T}},
        fock_bra::FockConfig, bra::TuckerConfig, 
        fock_ket::FockConfig, ket::TuckerConfig,
        bra_coeffs, ket_coeffs) where {T}
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
    build_sigma!(sigma_vector::BSstate, ci_vector::BSstate, cluster_ops, clustered_ham, nbody=4)
"""
function build_sigma!(sigma_vector::BSstate, ci_vector::BSstate, cluster_ops, clustered_ham; nbody=4)
    #={{{=#

    fold!(sigma_vector)
    fold!(ci_vector)
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
function get_map(ci_vector::BSstate, cluster_ops, clustered_ham; shift = nothing)
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

function tucker_ci_solve!(ci_vector::BSstate, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    unfold!(ci_vector) 
    Hmap = get_map(ci_vector, cluster_ops, clustered_ham)

    v0 = get_vector(ci_vector)
    nr = size(v0)[2] 
    
    davidson = Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=1e-5)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    e,v = FermiCG.solve(davidson)
    set_vector!(ci_vector,v)
    return e,v
end
#=}}}=#




function compress_blocks(ts::BSstate; thresh=1e-7)
    for (fock, tconfigs) in ts.data
        display(fock)
        for (tconfig, coeffs) in tconfigs
            display(tconfig)
            tucker_decompose(coeffs)
        end
    end
end

