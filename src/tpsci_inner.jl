"""
    contract_matrix_element(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)

Contraction for local (1body) terms. No contraction is needed,
just a lookup from the correct operator
"""
function contract_matrix_element(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)
#={{{=#
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

    return cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][bra[c1.idx],ket[c1.idx]]
end
#=}}}=#


"""
    contract_matrix_element(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
"""
function contract_matrix_element(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)
#={{{=#
    #display(term)
    #println(bra, ket)

    c1 = term.clusters[1]
    c2 = term.clusters[2]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # <I|p'|J> h(pq) <K|q|L>
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    #@views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]

    mat_elem = 0.0
    #@tensor begin
    #    mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
    #end
    #mat_elem = _contract(term.ints, gamma1, gamma2)
    
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    mat_elem = _contract(term.ints, gamma1, gamma2, bra[c1.idx], ket[c1.idx], bra[c2.idx], ket[c2.idx])
    #@btime _contract($term.ints, $gamma1, $gamma2, $bra[$c1.idx], $ket[$c1.idx], $bra[$c2.idx], $ket[$c2.idx])
    #@code_warntype  _contract(term.ints, gamma1, gamma2)

    return state_sign * mat_elem
#=}}}=#
end



#   
#   old routines which contract taking in views. These are slower I think because of the 
#   allocations needed for views and for not recognizing that data is contiguous
function _contract(ints,gamma1,gamma2)
#={{{=#
    mat_elem = 0.0
    tmp = 0.0
    for j in 1:length(gamma2)
        tmp = gamma2[j]
        @simd for i in 1:length(gamma1)
            mat_elem += gamma1[i]*ints[i,j]*tmp
        end
    end
    return mat_elem
end
#=}}}=#
function _contract(ints,gamma1,gamma2,gamma3)
#={{{=#
    mat_elem = 0.0
    tmp = 0.0
    for k in 1:length(gamma3)
        for j in 1:length(gamma2)
            tmp = gamma2[j]*gamma3[k]
            @simd for i in 1:length(gamma1)
                mat_elem += gamma1[i]*ints[i,j,k]*tmp
            end
        end
    end
    return mat_elem
end
#=}}}=#
function _contract(ints,gamma1,gamma2,gamma3,gamma4)
#={{{=#
    mat_elem = 0.0
    tmp = 0.0
    for l in 1:length(gamma4)
        for k in 1:length(gamma3)
            for j in 1:length(gamma2)
                tmp = gamma2[j]*gamma3[k]*gamma4[l]
                @simd for i in 1:length(gamma1)
                    mat_elem += gamma1[i]*ints[i,j,k,l]*tmp
                end
            end
        end
    end
    return mat_elem
end
#=}}}=#
#   
#   new routines which contract taking in full tensors. Indexed with linear arrays 
function _contract(ints,gamma1::Array{T,3}, gamma2::Array{T,3}, b1, k1, b2, k2) where T
#={{{=#
    mat_elem = 0.0
    tmp = 0.0
    shift2 = (b2-1) * size(gamma2,1) + (k2-1)*size(gamma2,2)*size(gamma2,1)
    shift1 = (b1-1) * size(gamma1,1) + (k1-1)*size(gamma1,2)*size(gamma1,1)

    for j in 1:size(gamma2,1)
        tmp = gamma2[j + shift2]
        #shift_ints = (j-1)*size(ints,1)
        #ints_j = ints[:,j]
        @simd for i in 1:size(gamma1,1)
            @inbounds mat_elem += gamma1[i + shift1]*ints[i,j]*tmp
            #mat_elem += gamma1[i + shift1]*ints[i + shift_ints]*tmp
        end
    end
    return mat_elem
end
#=}}}=#
function _contract(ints,gamma1::Array{T,3}, gamma2::Array{T,3}, gamma3::Array{T,3}, b1, k1, b2, k2, b3, k3) where T
#={{{=#
    mat_elem = 0.0
    tmp = 0.0
    shift3 = (b3-1) * size(gamma3,1) + (k3-1)*size(gamma3,2)*size(gamma3,1)
    shift2 = (b2-1) * size(gamma2,1) + (k2-1)*size(gamma2,2)*size(gamma2,1)
    shift1 = (b1-1) * size(gamma1,1) + (k1-1)*size(gamma1,2)*size(gamma1,1)
    for k in 1:size(gamma3,1)
        for j in 1:size(gamma2,1)
            tmp = gamma2[j+shift2]*gamma3[k+shift3]
            @simd for i in 1:size(gamma1,1)
                @inbounds mat_elem += gamma1[i+shift1]*ints[i,j,k]*tmp
            end
        end
    end
    return mat_elem
end
#=}}}=#
function _contract(ints,gamma1::Array{T,3}, gamma2::Array{T,3}, gamma3::Array{T,3}, gamma4::Array{T,3}, b1, k1, b2, k2, b3, k3, b4, k4) where T
    #={{{=#
    mat_elem = 0.0
    tmp = 0.0
    shift4 = (b4-1) * size(gamma4,1) + (k4-1)*size(gamma4,2)*size(gamma4,1)
    shift3 = (b3-1) * size(gamma3,1) + (k3-1)*size(gamma3,2)*size(gamma3,1)
    shift2 = (b2-1) * size(gamma2,1) + (k2-1)*size(gamma2,2)*size(gamma2,1)
    shift1 = (b1-1) * size(gamma1,1) + (k1-1)*size(gamma1,2)*size(gamma1,1)
    for l in 1:size(gamma4,1)
        for k in 1:size(gamma3,1)
            tmp = gamma3[k+shift3]*gamma4[l+shift4]
            for j in 1:size(gamma2,1)
                tmp = gamma2[j+shift2]*tmp
                @simd for i in 1:size(gamma1,1)
                    @inbounds mat_elem += gamma1[i+shift1]*ints[i,j,k,l]*tmp
                end
            end
        end
    end
    return mat_elem
end
#=}}}=#
"""
    contract_matrix_element(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
"""
function contract_matrix_element(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)
#={{{=#
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

        fock_bra[ci] == fock_ket[ci] || error("wrong fock space:",term,fock_bra, fock_ket) 
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
    # <I|p'|J> h(pq) <K|q|L>
#    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
#    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
#    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
#    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3)
    
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3, 
        bra[c1.idx], ket[c1.idx], bra[c2.idx], ket[c2.idx], bra[c3.idx], ket[c3.idx])
    return state_sign * mat_elem
end
#=}}}=#


"""
    contract_matrix_element(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
"""
function contract_matrix_element(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)
#={{{=#
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

        fock_bra[ci] == fock_ket[ci] || error("wrong fock space:",term,fock_bra, fock_ket) 
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
    # <I|p'|J> h(pq) <K|q|L>
#    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
#    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
#    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
#    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
#    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3, gamma4)
    
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3, gamma4, 
        bra[c1.idx], ket[c1.idx], bra[c2.idx], ket[c2.idx], bra[c3.idx], ket[c3.idx], bra[c4.idx], ket[c4.idx])
    return state_sign * mat_elem
end
#=}}}=#



