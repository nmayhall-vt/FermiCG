#
# form_sigma_block_expand computes the action of the term on a Tucker compressed state, 
# NOT projected into the space defined by bra, but rather prescreen each operator defined by prescreen. 
# This is used to find the first order interactiong space (FOIS) from a Tucker compressed state.  
#

function form_sigma_block_expand(term::ClusteredTerm1B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            precontract=false,
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
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
    op =  (op1[bra[c1.idx],ket[c1.idx]] * coeffs_ket.factors[c1.idx])

    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [-c1.idx, c1.idx]
        state_indices[c1.idx] = c1.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
    push!(indices, state_indices)

    output_size = [size(coeffs_ket.core[1])...]
    output_size[c1.idx] = size(op,1)
    
    # loop over global states
    bra_cores = ntuple(r->zeros(T,output_size...), R)
    for r in 1:R

        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        if length(tensors) == 1
            # this means that all the overlaps and the operator is a scalar
            bra_cores[r] .= coeffs_ket.core[r] .* s
        else
            bra_cores[r] .= @ncon(tensors, indices) .* s
            #bra_cores[r] .= bra_cores[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    new_factors[c1.idx] = Matrix(1.0I, size(bra_cores[1],c1.idx), size(bra_cores[1],c1.idx)) 
    return Tucker(bra_cores, NTuple{N}(new_factors))
end
#=}}}=#

function form_sigma_block_expand(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            precontract=false,
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
    #={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    n_clusters = length(bra)
    #bound1 = norm(term.ints)*norm(coeffs_ket.core)
    ##bound1 *= norm(gamma1)*norm(gamma2)
    ##bound1 *= norm(coeffs_ket.factors[c1.idx])*norm(coeffs_ket.factors[c2.idx])
    #if bound1 < sqrt(prescreen)
    #    return FermiCG.Tucker(Array{Float64}(undef,[0 for i in 1:N]...), ntuple(i->Array{Float64}(undef,0,0), N))
    #end

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
    g1tmp::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2tmp::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    @views gamma1 = g1tmp[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = g2tmp[:,bra[c2.idx],ket[c2.idx]]
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    #@views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
  


    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end
    #@printf("Norm: %12.8f %12.8f %12.8f\n",norm(gamma1),norm(coeffs_ket.core),norm(gamma1)*norm(coeffs_ket.core))

    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))



    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
    if precontract 
        scale0 = 1
        for r in 1:R
            scale0 = max(scale0,norm(coeffs_ket.core[r])) 
        end
        scale1 = scale0*norm(term.ints)*prescreen
       
        if size(g1,2) > size(g1,1)*size(g1,3)
            D = permutedims(g1, [2,1,3])
            F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
            nkeep = 0
            for si in F.S
                if scale1*si > prescreen 
                    nkeep += 1
                end
            end
            #println(size(F.U), size(F.Vt))
            new_factor1 = F.U[:,1:nkeep]
            g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
            g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
            g1 = permutedims(g1, [2,1,3])

        end

        if size(g2,2) > size(g2,1)*size(g2,3)
            D = permutedims(g2, [2,1,3])
            F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
            nkeep = 0
            for si in F.S
                if scale1*si > prescreen 
                    nkeep += 1
                end
            end
            new_factor2 = F.U[:,1:nkeep]
            g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
            g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
            g2 = permutedims(g2, [2,1,3])
        end
    end

    #
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{T}[]
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

    if false 
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
        push!(indices, state_indices)

        output_size = [size(coeffs_ket.core[1])...]
        output_size[c1.idx] = size(op,3)
        output_size[c2.idx] = size(op,4)

        # loop over global states
        bra_cores = ntuple(r->zeros(T,output_size...), R)
        for r in 1:R

            push!(tensors, coeffs_ket.core[r])

            length(tensors) == length(indices) || error(" mismatch between operators and indices")
            #bra_core = zeros(1,1)
            #bra_core = zeros(1,1)
            if length(tensors) == 1
                # this means that all the overlaps and the operator is a scalar
                bra_cores[r] .= coeffs_ket.core[r] .* s
            else
                bra_cores[r] .= @ncon(tensors, indices)
                bra_cores[r] .= bra_cores[r] .* s
            end
            deleteat!(tensors,length(tensors))
        end

        new_factors = [coeffs_ket.factors[i] for i in 1:N]
        #new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
        #new_factors[c2.idx] = Matrix(1.0I, size(bra_core,c2.idx), size(bra_core,c2.idx))
        new_factors[c1.idx] = new_factor1
        new_factors[c2.idx] = new_factor2 
        return Tucker(bra_cores, NTuple{N}(new_factors))

    else
        new_factors = [coeffs_ket.factors[i] for i in 1:N]
        new_factors[c1.idx] = new_factor1
        new_factors[c2.idx] = new_factor2 
      
        dims = ntuple(i->size(new_factors[i],2), N)
        out_tucker = Tucker(ntuple(i->zeros(T,dims), R), NTuple{N}(new_factors))
        #out_tucker = Tucker(zeros(T,dims), NTuple{N}(new_factors))
        bra_cores = contract_dense_H_with_state(term, op, state_sign, out_tucker, coeffs_ket)

        for r in 1:R
            out_tucker.core[r] .= bra_cores[r]
        end
        return out_tucker
    end
end
#=}}}=#

function form_sigma_block_expand(term::ClusteredTerm3B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            precontract=true,
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
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
    g1tmp::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2tmp::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    g3tmp::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    @views gamma1 = g1tmp[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = g2tmp[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = g3tmp[:,bra[c3.idx],ket[c3.idx]]
    
    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    Ur = coeffs_ket.factors[c3.idx]
    @tensor begin
        g3[p,I,l] := Ur[J,l] * gamma3[p,I,J]
    end

    new_factor1 = Matrix(1.0I, 1,1)
    new_factor2 = Matrix(1.0I, 1,1)
    new_factor3 = Matrix(1.0I, 1,1)
    

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
    if precontract 
        scale0 = 1
        for r in 1:R
            scale0 = max(scale0,norm(coeffs_ket.core[r])) 
        end
        scale1 = scale0*norm(term.ints)
        #scale1 = 1.0

        D = permutedims(g1, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si*scale1 > prescreen
                nkeep += 1
            end
        end
        new_factor1 = zeros(T,size(F.U,1), nkeep)
        n1 = size(F.U,1)
        n2 = nkeep
        @inbounds @simd for ij in 1:n1*n2
            new_factor1[ij] = F.U[ij]
        end
        g1 = F.S[1:nkeep] .* F.Vt[1:nkeep,:] 
        #g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
        g1 = permutedims(g1, [2,1,3])


        D = permutedims(g2, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si*scale1 > prescreen
                nkeep += 1
            end
        end
        new_factor2 = zeros(T,size(F.U,1), nkeep)
        n1 = size(F.U,1)
        n2 = nkeep
        @inbounds @simd for ij in 1:n1*n2
            new_factor2[ij] = F.U[ij]
        end
        g2 = F.S[1:nkeep] .* F.Vt[1:nkeep,:] 
        #g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
        g2 = permutedims(g2, [2,1,3])


        D = permutedims(g3, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si*scale1 > prescreen
                nkeep += 1
            end
        end
        new_factor3 = zeros(T,size(F.U,1), nkeep)
        n1 = size(F.U,1)
        n2 = nkeep
        @inbounds @simd for ij in 1:n1*n2
            new_factor3[ij] = F.U[ij]
        end
        #new_factor3 = F.U[:,1:nkeep]
        #g3 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g3 = F.S[1:nkeep] .* F.Vt[1:nkeep,:] 
        g3 = reshape(g3, size(g3,1), size(D,2), size(D,3))
        g3 = permutedims(g3, [2,1,3])
    
    else
        new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
        new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))
        new_factor3 = Matrix(1.0I, size(g3,2), size(g3,2))
    end

    #
    # Now contract into 3body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{T}[]
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
    #tensors = Vector{NTuple{R,Array{T}} }()
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    if false 
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
        push!(indices, state_indices)

        output_size = [size(coeffs_ket.core[1])...]
        output_size[c1.idx] = size(op,4)
        output_size[c2.idx] = size(op,5)
        output_size[c3.idx] = size(op,6)

        # loop over global states
        bra_cores = ntuple(r->zeros(T,output_size...), R)
        for r in 1:R

            push!(tensors, coeffs_ket.core[r])

            length(tensors) == length(indices) || error(" mismatch between operators and indices")
            #bra_core = zeros(1,1)
            #bra_core = zeros(1,1)
            if length(tensors) == 1
                # this means that all the overlaps and the operator is a scalar
                bra_cores[r] .= coeffs_ket.core[r] .* s
            else
                #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", size.(overlaps), indices))
                #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", overlaps, indices))
                bra_cores[r] .= @ncon(tensors, indices)
                bra_cores[r] .= bra_cores[r] .* s
            end
            deleteat!(tensors,length(tensors))
        end

        new_factors = [coeffs_ket.factors[i] for i in 1:N]
        new_factors[c1.idx] = new_factor1
        new_factors[c2.idx] = new_factor2 
        new_factors[c3.idx] = new_factor3 
        out_tucker = Tucker(bra_cores, tuple(new_factors...))
    else
        new_factors = [coeffs_ket.factors[i] for i in 1:N]
        new_factors[c1.idx] = new_factor1
        new_factors[c2.idx] = new_factor2 
        new_factors[c3.idx] = new_factor3 
      
        dims = ntuple(i->size(new_factors[i],2), N)
        out_tucker = Tucker(ntuple(i->zeros(T,dims), R), NTuple{N}(new_factors))
        #out_tucker = Tucker(zeros(T,dims), tuple(new_factors...))
        #out_tucker = Tucker(zeros(T,dims), NTuple{N}(new_factors))
        bra_cores = contract_dense_H_with_state(term, op, state_sign, out_tucker, coeffs_ket)

        for r in 1:R
            out_tucker.core[r] .= bra_cores[r]
        end
        return out_tucker
    end

end
#=}}}=#

function form_sigma_block_expand(term::ClusteredTerm4B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            precontract=true,
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
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
    g1tmp::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2tmp::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    g3tmp::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    g4tmp::Array{Float64,3} = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    @views gamma1 = g1tmp[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = g2tmp[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = g3tmp[:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = g4tmp[:,bra[c4.idx],ket[c4.idx]]

    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    Ur = coeffs_ket.factors[c3.idx]
    @tensor begin
        g3[p,I,l] := Ur[J,l] * gamma3[p,I,J]
    end

    Ur = coeffs_ket.factors[c4.idx]
    @tensor begin
        g4[p,I,l] := Ur[J,l] * gamma4[p,I,J]
    end
    
    new_factor1 = Matrix(1.0I, 1,1)
    new_factor2 = Matrix(1.0I, 1,1)
    new_factor3 = Matrix(1.0I, 1,1)
    new_factor4 = Matrix(1.0I, 1,1)
    
    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
   
    if precontract 

        scale0 = 1
        for r in 1:R
            scale0 = max(scale0,norm(coeffs_ket.core[r])) 
        end
        scale1 = scale0*norm(term.ints)
        #scale1 = 1.0
        
        D = permutedims(g1, [2,1,3])
        F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
        nkeep = 0
        for si in F.S
            if si*scale1 > prescreen
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
            if si*scale1 > prescreen
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
            if si*scale1 > prescreen
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
            if si*scale1 > prescreen
                nkeep += 1
            end
        end
        new_factor4 = F.U[:,1:nkeep]
        g4 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
        g4 = reshape(g4, size(g4,1), size(D,2), size(D,3))
        g4 = permutedims(g4, [2,1,3])

    else
        new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
        new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))
        new_factor3 = Matrix(1.0I, size(g3,2), size(g3,2))
        new_factor4 = Matrix(1.0I, size(g4,2), size(g4,2))
    end

    #
    # Now contract into 4body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{T}[]
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

    # should we use ncon or manual?
    if false 
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

        push!(indices, state_indices)

        output_size = [size(coeffs_ket.core[1])...]
        output_size[c1.idx] = size(op,5)
        output_size[c2.idx] = size(op,6)
        output_size[c3.idx] = size(op,7)
        output_size[c4.idx] = size(op,8)

        # loop over global states
        bra_cores = ntuple(r->zeros(T,output_size...), R)
        for r in 1:R

            push!(tensors, coeffs_ket.core[r])

            length(tensors) == length(indices) || error(" mismatch between operators and indices")
            if length(tensors) == 1
                # this means that all the overlaps and the operator is a scalar
                bra_cores[r] .= coeffs_ket.core[r] .* s
            else
                bra_cores[r] .= @ncon(tensors, indices)
                bra_cores[r] .= bra_cores[r] .* s
            end
            deleteat!(tensors,length(tensors))
        end

        new_factors = [coeffs_ket.factors[i] for i in 1:N]
        new_factors[c1.idx] = new_factor1
        new_factors[c2.idx] = new_factor2 
        new_factors[c3.idx] = new_factor3 
        new_factors[c4.idx] = new_factor4 
        return Tucker(bra_cores, NTuple{N}(new_factors))
    else
        new_factors = [coeffs_ket.factors[i] for i in 1:N]
        new_factors[c1.idx] = new_factor1
        new_factors[c2.idx] = new_factor2 
        new_factors[c3.idx] = new_factor3 
        new_factors[c4.idx] = new_factor4 

        dims = ntuple(i->size(new_factors[i],2), N)
        out_tucker = Tucker(ntuple(i->zeros(T,dims), R), NTuple{N}(new_factors))
        #out_tucker = Tucker(zeros(T,dims), NTuple{N}(new_factors))
        #println(length(out_tucker.core), length(coeffs_ket.core))
        bra_cores = contract_dense_H_with_state(term, op, state_sign, out_tucker, coeffs_ket)

        for r in 1:R
            out_tucker.core[r] .= bra_cores[r]
        end
        return out_tucker
    end
    
end
#=}}}=#

function form_sigma_block_expand2(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R},
                            scr::Vector{Vector{T}};  
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
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
    
    scr1 = scr[1]
    scr2 = scr[2]
    scr3 = scr[3]
    scr4 = scr[4]

    np = size(term.ints,1)
    nq = size(term.ints,2)
    
    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    g1a::Array{T,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    #@btime @views gamma1 = $g1a[:,$bra[$c1.idx],$ket[$c1.idx]]
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma1 = g1a[:,bra[c1.idx],ket[c1.idx]]
    Ur = coeffs_ket.factors[c1.idx]

    # gamma1(p,I,J) U(J,j)  = g1(p,I,j)
    nI = size(gamma1, 2) 
    nJ = size(gamma1, 3) 
    nj = size(Ur, 2) 

    #resize!(scr1, np*nI*nj)
    #scr1 = reshape2(scr1, (np*nI,nj))
    #mul!(scr1, reshape2(gamma1,(np*nI,nJ)), Ur)
    #g1 = reshape2(scr1, (np,nI,nj))

    g1 = scr[1]
    resize!(g1, np*nI*nj)
    g1 = reshape2(g1, (np,nI,nj))
    @tensor begin
        g1[p,I,l] = Ur[J,l] * gamma1[p,I,J]
        #g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over

    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))



    D = permutedims(g1, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
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
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor2 = F.U[:,1:nkeep]
    g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
    g2 = permutedims(g2, [2,1,3])

    #
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{T}[]
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

    push!(tensors, coeffs_ket.core)
    push!(indices, state_indices)

    length(tensors) == length(indices) || error(" mismatch between operators and indices")

    bra_core = zeros(1,1)
    if length(tensors) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_core = coeffs_ket.core .* s
    else
        #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", overlaps, indices))
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

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    #new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
    #new_factors[c2.idx] = Matrix(1.0I, size(bra_core,c2.idx), size(bra_core,c2.idx))
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    return Tucker(bra_core, NTuple{N}(new_factors))
end
#=}}}=#


