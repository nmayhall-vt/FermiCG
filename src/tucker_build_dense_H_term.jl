
"""
    Contract integrals and ClusterOps to form dense 4-body Hamiltonian matrix (tensor) in Tucker basis
"""
function build_dense_H_term(term::ClusteredTerm1B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    op = Array{T}[]

    op1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]

    #
    # Get 1body operator and compress it using the cluster's Tucker factors
    op = coeffs_bra.factors[c1.idx]' * (op1[bra[c1.idx],ket[c1.idx]] * coeffs_ket.factors[c1.idx])

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm2B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    op = Array{T}[]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    g1tmp::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2tmp::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    @views gamma1 = g1tmp[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = g2tmp[:,bra[c2.idx],ket[c2.idx]]
    
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #g1 = _compress_local_operator(gamma1, Ul, Ur)
    #g1 = @ncon([gamma1, U1, U2], [[-1,2,3], [2,-2], [3,-3]])

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
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

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm3B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    op = Array{T}[]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    g1tmp::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2tmp::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    g3tmp::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    @views gamma1 = g1tmp[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = g2tmp[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = g3tmp[:,bra[c3.idx],ket[c3.idx]]
    
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #display(("g1/2", size(g1), size(g2)))

    Ul = coeffs_bra.factors[c3.idx]
    Ur = coeffs_ket.factors[c3.idx]
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

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm4B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]
    op = Array{T}[]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    g1tmp::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    g2tmp::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    g3tmp::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    g4tmp::Array{Float64,3} = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    @views gamma1 = g1tmp[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = g2tmp[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = g3tmp[:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = g4tmp[:,bra[c4.idx],ket[c4.idx]]
    
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end
    #display(("g1/2", size(g1), size(g2)))

    Ul = coeffs_bra.factors[c3.idx]
    Ur = coeffs_ket.factors[c3.idx]
    @tensor begin
        tmp[p,k,J] := Ul[I,k] * gamma3[p,I,J]
        g3[p,k,l] := Ur[J,l] * tmp[p,k,J]
    end

    Ul = coeffs_bra.factors[c4.idx]
    Ur = coeffs_ket.factors[c4.idx]
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
    return op
end
#=}}}=#



#
# these functions use scratch files for intermediates to reduce allocations
#

function build_dense_H_term(term::ClusteredTerm1B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker,
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    op = Array{T,2}[]
        
    op1::Array{T,2} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]

    #
    # Get 1body operator and compress it using the cluster's Tucker factors
    op = coeffs_bra.factors[c1.idx]' * (op1[bra[c1.idx],ket[c1.idx]] * coeffs_ket.factors[c1.idx])

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm2B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker, 
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1m::Array{T,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2m::Array{T,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    @views gamma1 = gamma1m[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = gamma2m[:,bra[c2.idx],ket[c2.idx]]
    #@btime @views gamma1::Array{Float64,3} = $cluster_ops[$c1.idx][$term.ops[1]][($fock_bra[$c1.idx],$fock_ket[$c1.idx])][:,$bra[$c1.idx],$ket[$c1.idx]]
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]

    tmp = scr_f[1]
    g1  = scr_f[2]
    g2  = scr_f[3]

    resize!(tmp, size(Ul,2) * size(gamma1,1) * size(gamma1,3))
    resize!(g1, size(Ul,2) * size(Ur,2) * size(gamma1,1))
    
    tmp = reshape2(tmp, (size(gamma1,1), size(Ul,2), size(gamma1,3)))
    g1 = reshape2(g1, (size(gamma1,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp[p,k,J] = Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] = Ur[J,l] * tmp[p,k,J]
    end

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
    
    tmp = scr_f[1]
    resize!(tmp, size(Ul,2) * size(gamma2,1) * size(gamma2,3))
    resize!(g2, size(Ul,2) * size(Ur,2) * size(gamma2,1))
    
    tmp = reshape2(tmp, (size(gamma2,1), size(Ul,2), size(gamma2,3)))
    g2 = reshape2(g2, (size(gamma2,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp[p,k,J] = Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] = Ur[J,l] * tmp[p,k,J]
    end

    tmp2 = scr_f[4]
    resize!(tmp2, size(term.ints,2) * size(g1,3) * size(g1,2) )
    tmp2 = reshape2(tmp2, (size(term.ints,2), size(g1,3), size(g1,2)))
    
    @tensor begin
        tmp2[q,J,I] = term.ints[p,q] * g1[p,I,J]
        op[J,L,I,K] := tmp2[q,J,I] * g2[q,K,L]
    end

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm3B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker,
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1m::Array{T,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2m::Array{T,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3m::Array{T,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    @views gamma1 = gamma1m[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = gamma2m[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = gamma3m[:,bra[c3.idx],ket[c3.idx]]
    
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    g1  = scr_f[3]
    g2  = scr_f[4]
    g3  = scr_f[5]

    resize!(tmp1, size(Ul,2) * size(gamma1,1) * size(gamma1,3))
    resize!(g1, size(Ul,2) * size(Ur,2) * size(gamma1,1))
    
    tmp1 = reshape2(tmp1, (size(gamma1,1), size(Ul,2), size(gamma1,3)))
    g1 = reshape2(g1, (size(gamma1,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp1[p,k,J] = Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] = Ur[J,l] * tmp1[p,k,J]
    end

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
    
    tmp1 = scr_f[1]
    resize!(tmp1, size(gamma2,1) * size(Ul,2) * size(gamma2,3))
    resize!(g2, size(Ul,2) * size(Ur,2) * size(gamma2,1))
    
    tmp1 = reshape2(tmp1, (size(gamma2,1), size(Ul,2), size(gamma2,3)))
    g2 = reshape2(g2, (size(gamma2,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp1[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp1[p,k,J]
    end
    #display(("g1/2", size(g1), size(g2)))

    Ul = coeffs_bra.factors[c3.idx]
    Ur = coeffs_ket.factors[c3.idx]
    
    tmp1 = scr_f[1]
    resize!(tmp1, size(Ul,2) * size(gamma3,1) * size(gamma3,3))
    resize!(g3, size(Ul,2) * size(Ur,2) * size(gamma3,1))
    
    tmp1 = reshape2(tmp1, (size(gamma3,1), size(Ul,2), size(gamma3,3)))
    g3 = reshape2(g3, (size(gamma3,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp1[p,k,J] = Ul[I,k] * gamma3[p,I,J]
        g3[p,k,l] = Ur[J,l] * tmp1[p,k,J]
    end

    #
    # Now contract into 3body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    resize!(tmp1, size(term.ints,2) * size(term.ints,3) * size(g1,2) * size(g1,3))
    resize!(tmp2, size(term.ints,3) * size(g1,2) * size(g1,3) * size(g2,2) * size(g2,3))
    
    tmp1 = reshape2(tmp1, (size(term.ints,2), size(term.ints,3), size(g1,2), size(g1,3)))
    tmp2 = reshape2(tmp2, (size(term.ints,3), size(g1,2), size(g1,3), size(g2,2), size(g2,3)))
    @tensor begin
        tmp1[q,r,I,J] = term.ints[p,q,r] * g1[p,I,J]
        tmp2[r,I,J,K,L] = tmp1[q,r,I,J] * g2[q,K,L]
        op[J,L,N,I,K,M] := tmp2[r,I,J,K,L] * g3[r,M,N]
    end

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm4B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker,
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1m::Array{T,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2m::Array{T,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3m::Array{T,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    gamma4m::Array{T,3} = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    @views gamma1 = gamma1m[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = gamma2m[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = gamma3m[:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = gamma4m[:,bra[c4.idx],ket[c4.idx]]



    Ul1 = coeffs_bra.factors[c1.idx]
    Ul2 = coeffs_bra.factors[c2.idx]
    Ul3 = coeffs_bra.factors[c3.idx]
    Ul4 = coeffs_bra.factors[c4.idx]
    
    Ur1 = coeffs_ket.factors[c1.idx]
    Ur2 = coeffs_ket.factors[c2.idx]
    Ur3 = coeffs_ket.factors[c3.idx]
    Ur4 = coeffs_ket.factors[c4.idx]
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    tmp3 = scr_f[3]
    tmp4 = scr_f[4]
    g1  = scr_f[5]
    g2  = scr_f[6]
    g3  = scr_f[7]
    g4  = scr_f[8]

    resize!(tmp1, size(Ul1,2) * size(gamma1,1) * size(gamma1,3))
    resize!(tmp2, size(Ul2,2) * size(gamma2,1) * size(gamma2,3))
    resize!(tmp3, size(Ul3,2) * size(gamma3,1) * size(gamma3,3))
    resize!(tmp4, size(Ul4,2) * size(gamma4,1) * size(gamma4,3))

    resize!(g1, size(Ul1,2) * size(Ur1,2) * size(gamma1,1))
    resize!(g2, size(Ul2,2) * size(Ur2,2) * size(gamma2,1))
    resize!(g3, size(Ul3,2) * size(Ur3,2) * size(gamma3,1))
    resize!(g4, size(Ul4,2) * size(Ur4,2) * size(gamma4,1))
    
    tmp1 = reshape2(tmp1, (size(gamma1,1), size(Ul1,2), size(gamma1,3)))
    tmp2 = reshape2(tmp2, (size(gamma2,1), size(Ul2,2), size(gamma2,3)))
    tmp3 = reshape2(tmp3, (size(gamma3,1), size(Ul3,2), size(gamma3,3)))
    tmp4 = reshape2(tmp4, (size(gamma4,1), size(Ul4,2), size(gamma4,3)))
    
    g1 = reshape2(g1, (size(gamma1,1), size(Ul1,2), size(Ur1,2)))
    g2 = reshape2(g2, (size(gamma2,1), size(Ul2,2), size(Ur2,2)))
    g3 = reshape2(g3, (size(gamma3,1), size(Ul3,2), size(Ur3,2)))
    g4 = reshape2(g4, (size(gamma4,1), size(Ul4,2), size(Ur4,2)))


    @tensor begin
        tmp1[p,k,J] = Ul1[I,k] * gamma1[p,I,J]
        g1[p,k,l]   = Ur1[J,l] * tmp1[p,k,J]
    end
    @tensor begin
        tmp2[p,k,J] = Ul2[I,k] * gamma2[p,I,J]
        g2[p,k,l]   = Ur2[J,l] * tmp2[p,k,J]
    end
    @tensor begin
        tmp3[p,k,J] = Ul3[I,k] * gamma3[p,I,J]
        g3[p,k,l]   = Ur3[J,l] * tmp3[p,k,J]
    end
    @tensor begin
        tmp4[p,k,J] = Ul4[I,k] * gamma4[p,I,J]
        g4[p,k,l]   = Ur4[J,l] * tmp4[p,k,J]
    end


    #
    # Now contract into 4body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    tmp3 = scr_f[3]
    
    resize!(tmp1, size(term.ints,2) * size(term.ints,3) * size(term.ints,4) * size(g1,3) * size(g1,2))
    resize!(tmp2, size(term.ints,3) * size(term.ints,4) * size(g1,3) * size(g2,3) * size(g1,2) * size(g2,2))
    resize!(tmp3, size(term.ints,4) * size(g1,3) * size(g2,3) * size(g3,3) * size(g1,2) * size(g2,2) * size(g3,2))
    
    tmp1 = reshape2(tmp1, (size(term.ints,2), size(term.ints,3), size(term.ints,4), size(g1,3), size(g1,2)))
    tmp2 = reshape2(tmp2, (size(term.ints,3), size(term.ints,4), size(g1,3), size(g2,3), size(g1,2), size(g2,2)))
    tmp3 = reshape2(tmp3, (size(term.ints,4), size(g1,3), size(g2,3), size(g3,3), size(g1,2), size(g2,2), size(g3,2)))
    
    @tensor begin
        tmp1[q,r,s,J,I] = term.ints[p,q,r,s] * g1[p,I,J]
        tmp2[r,s,J,L,I,K] = tmp1[q,r,s,J,I] * g2[q,K,L]
        tmp3[s,J,L,N,I,K,M] = tmp2[r,s,J,L,I,K] * g3[r,M,N]
        op[J,L,N,P,I,K,M,O] := tmp3[s,J,L,N,I,K,M] * g4[s,O,P]
    end

    return op
end
#=}}}=#


