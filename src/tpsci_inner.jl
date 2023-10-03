"""
    contract_matrix_element(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)

Contraction for local (1body) terms. No contraction is needed,
just a lookup from the correct operator
"""
function contract_matrix_element(   term::ClusteredTerm1B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig) where {T}
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
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)

Form TPSCI matrix element by contracting operators with integrals for 2body terms. 
"""
function contract_matrix_element(   term::ClusteredTerm2B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig) where {T}
#={{{=#
    #display(term)
    #println(bra, ket)

    c1 = term.clusters[1]
    c2 = term.clusters[2]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        
    # <I|p'|J> h(pq) <K|q|L>
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    #@views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]

    mat_elem = 0.0
    #@tensor begin
    #    mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
    #end
    #mat_elem = _contract(term.ints, gamma1, gamma2)
    
    
    haskey(cluster_ops[c1.idx][term.ops[1]],  (fock_bra[c1.idx],fock_ket[c1.idx])) || return
    haskey(cluster_ops[c2.idx][term.ops[2]],  (fock_bra[c2.idx],fock_ket[c2.idx])) || return
    #@btime haskey($cluster_ops[$c2.idx][$term.ops[2]],  ($fock_bra[$c2.idx],$fock_ket[$c2.idx])) 
    
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    #@views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    
    gamma1::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    #gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    #gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    #@btime gamma2::Array{Float64,3} = $cluster_ops[$c2.idx][$term.ops[2]][($fock_bra[$c2.idx],$fock_ket[$c2.idx])]
    #@views gamma1 = g1[:,bra[c1.idx],ket[c1.idx]]
    #@views gamma2 = g2[:,bra[c2.idx],ket[c2.idx]]
    mat_elem = _contract(term.ints, gamma1, gamma2, bra[c1.idx], ket[c1.idx], bra[c2.idx], ket[c2.idx])
    #@btime _contract($term.ints, $gamma1, $gamma2, $bra[$c1.idx], $ket[$c1.idx], $bra[$c2.idx], $ket[$c2.idx])
    #@code_warntype  _contract(term.ints, gamma1, gamma2)

    return state_sign * mat_elem
end
#=}}}=#
"""
    contract_matrix_element(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)

Form TPSCI matrix element by contracting operators with integrals for 3body terms. 
"""
function contract_matrix_element(   term::ClusteredTerm3B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig) where {T}
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

    haskey(cluster_ops[c1.idx][term.ops[1]],  (fock_bra[c1.idx],fock_ket[c1.idx])) || return
    haskey(cluster_ops[c2.idx][term.ops[2]],  (fock_bra[c2.idx],fock_ket[c2.idx])) || return
    haskey(cluster_ops[c3.idx][term.ops[3]],  (fock_bra[c3.idx],fock_ket[c3.idx])) || return

    gamma1::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    #gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    #gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    #gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3, 
                         bra[c1.idx], ket[c1.idx], bra[c2.idx], ket[c2.idx], bra[c3.idx], ket[c3.idx])
    return state_sign * mat_elem
end
#=}}}=#
"""
    contract_matrix_element(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig)

Form TPSCI matrix element by contracting operators with integrals for 4body terms. 
"""
function contract_matrix_element(   term::ClusteredTerm4B{T}, 
                                    cluster_ops::Vector{ClusterOps{T}},
                                    fock_bra::FockConfig, bra::ClusterConfig, 
                                    fock_ket::FockConfig, ket::ClusterConfig) where {T}
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
    
    haskey(cluster_ops[c1.idx][term.ops[1]],  (fock_bra[c1.idx],fock_ket[c1.idx])) || return
    haskey(cluster_ops[c2.idx][term.ops[2]],  (fock_bra[c2.idx],fock_ket[c2.idx])) || return
    haskey(cluster_ops[c3.idx][term.ops[3]],  (fock_bra[c3.idx],fock_ket[c3.idx])) || return
    haskey(cluster_ops[c4.idx][term.ops[4]],  (fock_bra[c4.idx],fock_ket[c4.idx])) || return
    
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3, gamma4, 
        bra[c1.idx], ket[c1.idx], bra[c2.idx], ket[c2.idx], bra[c3.idx], ket[c3.idx], bra[c4.idx], ket[c4.idx])
    return state_sign * mat_elem
end
#=}}}=#

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
            mat_elem += gamma1[i + shift1]*ints[i,j]*tmp
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
                mat_elem += gamma1[i+shift1]*ints[i,j,k]*tmp
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
            tmp2 = gamma3[k+shift3]*gamma4[l+shift4]
            for j in 1:size(gamma2,1)
                tmp = gamma2[j+shift2]*tmp2
                @simd for i in 1:size(gamma1,1)
                    mat_elem += gamma1[i+shift1]*ints[i,j,k,l]*tmp
                end
            end
        end
    end
    return mat_elem
end
#=}}}=#





"""
    contract_matvec(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
"""
function contract_matvec(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        
    
    #
    # <:|p'|J> h(pq) <:|q|L>

    new_coeffs = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,conf_ket[c1.idx]] * state_sign
    newI = 1:size(new_coeffs,1)

    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()

    _collect_significant!(out, conf_ket, new_coeffs, coef_ket, c1.idx,  newI,  thresh)
            

    return out 
end
#=}}}=#


"""
    contract_matvec(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
"""
function contract_matvec(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
  

    #
    # <:|p'|J> h(pq) <:|q|L>
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    

    #
    # tmp(q,I)   = h(p,q)' * gamma1(p,I)
    # new_coeffs(I,J) = tmp(q,I)' * gamma2(q,J)
    #
    #tmp = [coef_ket...]
    #@tensor begin
    #    new_coeffs[p,J,s] := term.ints[p,q] * gamma2[q,J] 
    #    new_coeffs[I,J,s] := new_coeffs[p,J,s] * gamma1[p,I] 
    #end
    new_coeffs = term.ints' * gamma1 
    new_coeffs = new_coeffs' * gamma2
   
    #coeffs = []
    #for r in 1:R
    #    c = coef_ket[r]*state_sign
    #    push!(coeffs, new_coeff .* c)
    #end

    if state_sign < 0
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(new_coeffs,1)
    newJ = 1:size(new_coeffs,2)

    # multiply by state coeffs
    #new_coeffs = kron(new_coeffs, coef_ket')
    #new_coeffs = reshape(new_coeffs, length(newI), length(newJ), R) 
        
    

    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    out2 = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    #out2 = OrderedDict{NTuple{N,Int16}, MVector{R,T}}()

    #sizehint!(out2,prod(size(new_coeffs)))

    #@btime _collect_significant2!($out2, $conf_ket, $new_coeffs, $coef_ket, $c1.idx, $c2.idx, $newI, $newJ, $thresh)
    #display(size(new_coeffs))
    #display(size(coef_ket))
    #@btime _collect_significant!($out2, $conf_ket, $new_coeffs, $coef_ket, $c1.idx, $c2.idx, $newI, $newJ, $thresh)
    #@code_warntype _collect_significant!(out2, conf_ket, new_coeffs, coef_ket, c1.idx, c2.idx, newI, newJ, thresh)
    #display(length(out2))
    #error("huh")
    _collect_significant!(out, conf_ket, new_coeffs, coef_ket, c1.idx, c2.idx, newI, newJ, thresh)
            

    return out 
end
#=}}}=#

"""
    contract_matvec_M3(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
"""
function contract_matvec_M3(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # <:|p'|J> h(pqr) <:|q|L> <:|r|N>
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,:,conf_ket[c3.idx]]
  
#    newI = []
#    for i in 1:size(gamma1,2)
#        if maximum(abs.(gamma1[:,i])) > thresh
#            push!(newI,i)
#        end
#    end
#
#    newJ = []
#    for i in 1:size(gamma2,2)
#        if maximum(abs.(gamma2[:,i])) > thresh
#            push!(newJ,i)
#        end
#    end
#
#    newK = []
#    for i in 1:size(gamma3,2)
#        if maximum(abs.(gamma3[:,i])) > thresh
#            push!(newK,i)
#        end
#    end

    newI = 1:size(gamma1,2)
    newJ = 1:size(gamma2,2)
    newK = 1:size(gamma3,2)
    
    new_coeffs = []
    @tensor begin
        new_coeffs[p,q,M] := term.ints[p,q,r]  * gamma3[r,M] #* coef_ket
        new_coeffs[p,L,M] := new_coeffs[p,q,M] * gamma2[q,L]
        new_coeffs[J,L,M] := new_coeffs[p,L,M] * gamma1[p,J]
        #new_coeffs[I,J,K] := ((term.ints[p,q,r]  * gamma1[p,I])  * gamma2[q,J])  * gamma3[r,K]
    end

    
    if state_sign < 0
        new_coeffs .= -new_coeffs
    end 


    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    
    _collect_significant!(out, conf_ket, new_coeffs, coef_ket, c1.idx, c2.idx, c3.idx, newI, newJ, newK, thresh)

    return out 
end
#=}}}=#

"""
    contract_matvec_M4(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
"""
function contract_matvec_M4(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # <:|p'|J> h(pqrs) <:|q|L> <:|r|N> <:|s|P>
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,:,conf_ket[c3.idx]]
    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,:,conf_ket[c4.idx]]
    

    new_coeffs = []
    @tensor begin
        new_coeffs[I,J,K,L] := term.ints[p,q,r,s]  * gamma1[p,I]  * gamma2[q,J]  * gamma3[r,K]  * gamma4[s,L] #* coef_ket
    end

    
    if state_sign < 0
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(new_coeffs,1)
    newJ = 1:size(new_coeffs,2)
    newK = 1:size(new_coeffs,3)
    newL = 1:size(new_coeffs,4)

    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    
    _collect_significant!(out, conf_ket, new_coeffs, coef_ket, c1.idx, c2.idx, c3.idx, c4.idx, newI, newJ, newK, newL, thresh)

    return out 
end
#=}}}=#


function _collect_significant!(out, conf_ket, new_coeffs, coeff, c1idx, newI, thresh) 
    #={{{=#
    N = length(conf_ket)
    cket = MVector{N,Int16}([conf_ket.config...])
     cket = [conf_ket.config...]
    for i::Int16 in newI
        if any((abs(new_coeffs[i]*s) > thresh for s in coeff))
            cket[c1idx] = i
            out[ClusterConfig(cket)] = new_coeffs[i]*coeff 
        end
    end
end
#=}}}=#

function _collect_significant!(out, conf_ket, new_coeffs, coeff, c1idx, c2idx, newI, newJ, thresh) 
#={{{=#
    N = length(conf_ket)
    R = length(coeff)
    cket = MVector{N,Int16}([conf_ket.config...])
    #cket = [conf_ket.config...]
    for j::Int16 in newJ
        cket[c2idx] = j
        for i::Int16 in newI
            if any((abs(new_coeffs[i,j]*s) > thresh for s in coeff))
                cket[c1idx] = i
                out[ClusterConfig(cket)] = new_coeffs[i,j]*coeff 
            end
        end
    end
end
#=}}}=#

function _collect_significant!(out, conf_ket, new_coeffs, coeff, c1idx, c2idx, c3idx, newI, newJ, newK, thresh) 
    #={{{=#
    N = length(conf_ket)
    cket = MVector{N,Int16}([conf_ket.config...])
    #cket = [conf_ket.config...]
    for k::Int16 in newK
        cket[c3idx] = k
        for j::Int16 in newJ
            cket[c2idx] = j
            for i::Int16 in newI
                if any((abs(new_coeffs[i,j,k]*s) > thresh for s in coeff))
                    cket[c1idx] = i
                    out[ClusterConfig(cket)] = new_coeffs[i,j,k]*coeff 
                    #out[ClusterConfig{N}(tuple(cket...))] = new_coeffs[i,j,k]*coeff 
                end
            end
        end
    end
end
#=}}}=#

function _collect_significant!(out, conf_ket, new_coeffs, coeff, c1idx, c2idx, c3idx, c4idx, newI, newJ, newK, newL, thresh) 
    #={{{=#
    N = length(conf_ket)
    cket = MVector{N,Int16}([conf_ket.config...])
    #cket = [conf_ket.config...]
    for l::Int16 in newL
        cket[c4idx] = l
        for k::Int16 in newK
            cket[c3idx] = k
            for j::Int16 in newJ
                cket[c2idx] = j
                for i::Int16 in newI
                    if any((abs(new_coeffs[i,j,k,l]*s) > thresh for s in coeff))
                        cket[c1idx] = i
                        out[ClusterConfig(cket)] = new_coeffs[i,j,k,l]*coeff 
                    end
                end
            end
        end
    end
end
#=}}}=#



#############################################################################################################################################
#       M^2 memory versions
#############################################################################################################################################
"""
    contract_matvec(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9, prescreen=true) where {T,R,N}

This version should only use M^2N^2 storage, and n^5 scaling n={MN}
"""
function contract_matvec(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9, prescreen=true) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
    


    #
    # h(pqr) <I|p'|_> <J|q|_> <K|r|_> 
    #
    # X(p,J,K) = h(pqr) <J|q|_> <K|r|_>
    #
    #
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,:,conf_ket[c3.idx]]
    

    XpJK = []
    @tensoropt begin
        XpJK[p,J,K] := term.ints[p,q,r] * gamma2[q,J] * gamma3[r,K]
    end

    
    if state_sign < 0
        XpJK .= -XpJK
    end 

    # preallocate tmp arrays
    
    cket = MVector{N,Int16}([conf_ket.config...])
    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
   
    newI = UnitRange{Int16}(1,size(gamma1,2))
    newJ = UnitRange{Int16}(1,size(gamma2,2))
    newK = UnitRange{Int16}(1,size(gamma3,2))

    if prescreen
        up_bound = upper_bound(term.ints, gamma1, gamma2, gamma3, c=maximum(abs.(coef_ket)))
        if up_bound < thresh
            return out
        end
        #newI, newJ, newK = upper_bound2(term.ints, gamma1, gamma2, gamma3, thresh, c=maximum(abs.(coef_ket)))
        #minimum(length.([newI,newJ,newK])) > 0 || return out
    end

#    scr1 = zeros(size(gamma1,2))
#    for k::Int16 in newK 
#        cket[c3.idx] = k
#        for j::Int16 in newJ 
#            cket[c2.idx] = j
#
#            @views scr1 = gamma1' * XpJK[:,j,k]
#            _collect_significant2!(out, newI, scr1, coef_ket, cket, thresh, c1.idx)
#        end
#    end
   
    scr1 = zeros(size(gamma1,2),size(gamma2,2))
    for k::Int16 in newK 
        cket[c3.idx] = k

        #
        @views BLAS.gemm!('T', 'N', 1.0, gamma1, XpJK[:,:,k], 0.0, scr1)
        #@btime @views BLAS.gemm!('T', 'N', 1.0, $gamma1, $XpJK[:,:,$k], 0.0, $scr1)
        #@views scr1 = gamma1' * XpJK[:,:,k]

        _collect_significant2!(out, newI, newJ, scr1, coef_ket, cket, thresh, c1.idx, c2.idx)
        #@btime _collect_significant2!($out, $newI, $newJ, $scr1, $coef_ket, $cket, $thresh, $c1.idx, $c2.idx)

    end

    return out 
end
#=}}}=#

"""
    contract_matvec(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9, prescreen=true) where {T,R,N}

This version should only use M^2N^2 storage, and n^5 scaling n={MN}
"""
function contract_matvec(   term::ClusteredTerm4B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9, prescreen=true) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
    


    #
    # <I|p'|_> h(pqrs) <J|q|_> <K|r|_> <L|s|_>
    #
    # X(p,q,K,L) = h(pqrs) <K|r|_> <L|s|_>
    #
    #
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,:,conf_ket[c3.idx]]
    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,:,conf_ket[c4.idx]]
    
    newI = UnitRange{Int16}(1,size(gamma1,2))
    newJ = UnitRange{Int16}(1,size(gamma2,2))
    newK = UnitRange{Int16}(1,size(gamma3,2))
    newL = UnitRange{Int16}(1,size(gamma4,2))

    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()
    
    if prescreen

        #
        #   max(H_IJKL) <= sum_pqrs abs(V(pqrs)) max(g1(p)_I) max(g2(q)_J) max(g3(r)_K) max(g4(s)_L) * max(abs(coeffs))

        up_bound = upper_bound(term.ints, gamma1, gamma2, gamma3, gamma4, c=maximum(abs.(coef_ket)))
        #@btime up_bound = upper_bound($term.ints, $gamma1, $gamma2, $gamma3, $gamma4, c=maximum(abs.($coef_ket)))
        up_bound > thresh || return out

        #
        # screen phase 2: ignore indices for each cluster which will produce discarded terms

        newI, newJ, newK, newL = upper_bound2(term.ints, gamma1, gamma2, gamma3, gamma4, thresh, c=maximum(abs.(coef_ket)))

        minimum(length.([newI,newJ,newK,newL])) > 0 || return out

        #@views gamma1 = gamma1[:,newI]
        #@views gamma2 = gamma2[:,newJ]
        #@views gamma3 = gamma3[:,newK]
        #@views gamma4 = gamma4[:,newL]
        gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,newI,conf_ket[c1.idx]]
        gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,newJ,conf_ket[c2.idx]]
        gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,newK,conf_ket[c3.idx]]
        gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,newL,conf_ket[c4.idx]]

    end

    XpqKL = []
    @tensoropt begin
        XpqKL[p,q,K,L] := term.ints[p,q,r,s]  * gamma3[r,K]  * gamma4[s,L] #* coef_ket
    end

    
    if state_sign < 0
        XpqKL .= -XpqKL
    end 

    # preallocate tmp arrays
    scr1 = zeros(size(gamma1,1), size(gamma2,2))
    scr2 = zeros(size(gamma1,2), size(gamma2,2))
    
    cket = MVector{N,Int16}([conf_ket.config...])
    #sizehint!(out,prod([size(gamma1,2),size(gamma2,2),size(gamma3,2),size(gamma4,2)]))
   
    #newI = UnitRange{Int16}(1,size(gamma1,2))
    #newJ = UnitRange{Int16}(1,size(gamma2,2))
    #newK = UnitRange{Int16}(1,size(gamma3,2))
    #newL = UnitRange{Int16}(1,size(gamma4,2))



    g2max = zeros(size(gamma2,1))
    for q in 1:size(gamma2,1)
        g2max[q] = maximum(abs.(gamma2[q,:]))* maximum(abs.(coef_ket))
    end

    for l::Int16 in 1:length(newL)
        cket[c4.idx] = newL[l]
        for k::Int16 in 1:length(newK)
            cket[c3.idx] = newK[k]
           
            if prescreen
                #isum(abs.(XpqKL[:,:,k,l]))*maximum(abs.(coef_ket)) > thresh || continue
                upper_bound(XpqKL[:,:,k,l]', gamma2, c=maximum(abs.(coef_ket))) > thresh || continue
                #bound = 0
                #for q in 1:size(gamma2,1)
                #    bound += maximum(abs.(XpqKL[:,q,k,l])) * g2max[q]
                #end
            end

            #
            # tmp1(p,J) = Xpq * g2(q,J)
            #@views scr1 = XpqKL[:,:,k,l] * gamma2
            #BLAS.gemm!('N','N', 1.0, XpqKL[:,:,k,l], gamma2, 0.0, scr1)
            BLAS.gemm!('N','N', 1.0, XpqKL[:,:,k,l], gamma2, 0.0, scr1)
            #BLAS.gemm!('N','N', 1.0, XpqKL[:,:,k,l], gamma2, 1.0, scr1)
            
            if prescreen
                upper_bound(gamma1,scr1, c=maximum(abs.(coef_ket))) > thresh || continue
            end
            
            #_test1!(scr1, XpqKL, gamma2, k, l)
            #@btime _test2!($scra, $XpqKL, $gamma2, $k, $l)
            #@views scr1 .= XpqKL[:,:,k,l] * gamma2
            
            #
            # tmp2(I,J) = g1(p,I) * tmp(p,J)
            #scr2 .= gamma1'*scr1
            BLAS.gemm!('T','N', 1.0, gamma1, scr1, 0.0, scr2)

            _collect_significant2!(out, newI, newJ, scr2, coef_ket, cket, thresh, c1.idx, c2.idx)
        end
    end

    return out 
end
#=}}}=#

#function _test2!(scr1, XpqKL, gamma2, k, l)
#    @views BLAS.gemm!('N','N', 1.0, XpqKL[:,:,k,l], gamma2, 1.0, scr1)
#    #@views scr1 = XpqKL[:,:,k,l] * gamma2
#end
#
#function _test1!(scr1, XpqKL, gamma2, k, l)
##={{{=#
#    Xshift = size(XpqKL,1)*size(XpqKL,2)*(k-1) + size(XpqKL,1)*size(XpqKL,2)*size(XpqKL,3)*(l-1)
#    np = size(XpqKL,1)
#    nq = size(XpqKL,2)
#    nJ = size(gamma2,2)
#    @inbounds for j in 1:nJ
#        scrshift = np*(j-1)
#        gshift = nq*(j-1)
#        for p in 1:np
#
#            @simd for q in 1:nq
#                scr1[p + scrshift] += XpqKL[p + np*(q-1) + Xshift] * gamma2[q + gshift]
#            end
#        end
#    end
#end
##=}}}=#
#
#
#function _collect_significant2!(out, newI, scr2, coef_ket, cket, thresh, c1idx)
#    for i::Int16 in newI
#        if any((abs(scr2[i]*s) > thresh for s in coef_ket))
#            cket[c1idx] = i
#            out[ClusterConfig(cket)] = scr2[i]*coef_ket
#        end
#    end
#end

function _collect_significant2!(out, newI, newJ, scr2, coef_ket, cket, thresh, c1idx, c2idx)
    thresh_curr = thresh / maximum(abs.(coef_ket))

    #@inbounds for ij in findall(x->(x>thresh_curr) || (x<-thresh_curr), scr2)
    #    cket[c1idx] = ij[1]
    #    cket[c2idx] = ij[2]
    #    out[ClusterConfig(cket)] = scr2[ij]*coef_ket
    #end

    nI = length(newI)
    nJ = length(newJ)
    @inbounds for j::Int16 in 1:nJ 
        cket[c2idx] = newJ[j]
        @fastmath @simd for i in 1:nI
            if (scr2[i,j] > thresh_curr) || (scr2[i,j] < -thresh_curr)
                cket[c1idx] = newI[i]
                out[ClusterConfig(cket)] = scr2[i,j]*coef_ket
            end
        end
    end
end



"""
    upper_bound(g1, g2; c::Float64=1.0)

Return upper bound on the size of matrix elements resulting from matrix multiply 

    V[I,J] =  g1[i,I] * g2[i,J] * c 

    max(|V|) <= sum_i max|g1[i,:]| * max|g2[i,:]| * |c|
"""
function upper_bound(g1, g2; c::Float64=1.0)
#={{{=#
    bound = 0
    n1 = size(g1,1) 
    n2 = size(g2,1) 
    n1 == n2 || throw(DimensionMismatch)

    absc = abs(c)
    @inbounds @simd for p in 1:n1
        pmax = maximum(abs.(g1[p,:]))
        qmax = maximum(abs.(g2[p,:]))
        bound += pmax * qmax * absc
    end

    return bound
end
#=}}}=#

"""
    upper_bound(v::Array{Float64,2}, g1, g2; c::Float64=1.0)

Return upper bound on the size of tensor elements resulting from the following contraction

    V[I,J] = v[i,j] * g1[i,I] * g2[j,J] 

    max(|V|) <= sum_ij |v[ij]| * |g1[i,:]|_8 * |g2[j,:]|_8 * |c|
"""
function upper_bound(v::Array{Float64,2}, g1, g2; c::Float64=1.0)
    #={{{=#
    bound = 0
    n1 = size(g1,1) 
    n2 = size(g2,1) 

    pmax = zeros(n1)
    qmax = zeros(n2)
    for p in 1:n1
        pmax[p] = maximum(abs.(g1[p,:]))
    end
    for p in 1:n2
        qmax[p] = maximum(abs.(g2[p,:]))
    end

    tmp = 0.0
    for q in 1:n2
        tmp = abs(c) * qmax[q]  
        @inbounds @simd for p in 1:n1
            bound += tmp * abs(v[p,q]) * pmax[p]
        end
    end

    return bound
end
#=}}}=#

"""
    upper_bound(v::Array{Float64,3}, g1, g2, g3; c::Float64=1.0)

Return upper bound on the size of tensor elements resulting from the following contraction

    V[I,J,K] = v[i,j,k] * g1[i,I] * g2[j,J] * g3[k,K] 

    max(|V|) <= sum_ijk |v[ijk]| * |g1[i,:]|_8 * |g2[j,:]|_8 * |g3[k,:]|_8 * |c|
"""
function upper_bound(v::Array{Float64,3}, g1, g2, g3; c::Float64=1.0)
#={{{=#
    bound = 0
    n1 = size(g1,1) 
    n2 = size(g2,1) 
    n3 = size(g3,1) 

    pmax = zeros(n1)
    qmax = zeros(n2)
    rmax = zeros(n3)
    for p in 1:n1
        pmax[p] = maximum(abs.(g1[p,:]))
    end
    for p in 1:n2
        qmax[p] = maximum(abs.(g2[p,:]))
    end
    for p in 1:n3
        rmax[p] = maximum(abs.(g3[p,:]))
    end
    
    tmp = 0.0
    @inbounds for r in 1:n3
        for q in 1:n2
            tmp = abs(c) * qmax[q] * rmax[r] 
            @simd for p in 1:n1
                bound += tmp * abs(v[p,q,r]) * pmax[p]
            end
        end
    end

    return bound
end
#=}}}=#

"""
    upper_bound(v::Array{Float64,4}, g1, g2, g3, g4; c::Float64=1.0)

Return upper bound on the size of tensor elements resulting from the following contraction

    V[I,J,K,L] = v[i,j,k,l] * g1[i,I] * g2[j,J] * g3[k,K] * g4[l,L]

    max(|V|) <= sum_ijkl |v[ijkl]| * |g1[i,:]|_8 * |g2[j,:]|_8 * |g3[k,:]|_8 * |g4[l,:]|_8
"""
function upper_bound(v::Array{Float64,4}, g1, g2, g3, g4; c::Float64=1.0)
    #={{{=#
        bound = 0
        n1 = size(g1,1) 
        n2 = size(g2,1) 
        n3 = size(g3,1) 
        n4 = size(g4,1) 
    
        pmax = zeros(n1)
        qmax = zeros(n2)
        rmax = zeros(n3)
        smax = zeros(n4)
        for p in 1:n1
            pmax[p] = maximum(abs.(g1[p,:]))
        end
        for p in 1:n2
            qmax[p] = maximum(abs.(g2[p,:]))
        end
        for p in 1:n3
            rmax[p] = maximum(abs.(g3[p,:]))
        end
        for p in 1:n4
            smax[p] = maximum(abs.(g4[p,:]))
        end
        
        tmp = 0.0
        @inbounds for s in 1:n4
            for r in 1:n3
                for q in 1:n2
                    tmp = abs(c) * qmax[q] * rmax[r] * smax[s] 
                    @simd for p in 1:n1
                        bound += tmp * abs(v[p,q,r,s]) * pmax[p]
                    end
                end
            end
        end
    return bound
end
#=}}}=#


        
"""
    upper_bound2(v::Array{Float64,3}, g1, g2, g3, thresh; c::Float64=1.0)

Get upper bound on the possible values 

    max(H_IJ(K)|_K <= sum_r (sum_pq vpqrs max(g1[p,:]) * max(g2[q,:]) * |c| ) * |g3(r,K)|
"""
function upper_bound2(v::Array{Float64,3}, g1, g2, g3, thresh; c::Float64=1.0)
    #={{{=#
        newI = Vector{Int16}() 
        newJ = Vector{Int16}() 
        newK = Vector{Int16}() 
       
        n1 = size(v,1)
        n2 = size(v,2)
        n3 = size(v,3)

        n1 == size(g1,1) || throw(DimensionMismatch)
        n2 == size(g2,1) || throw(DimensionMismatch)
        n3 == size(g3,1) || throw(DimensionMismatch)
        

        pmax = zeros(n1)
        qmax = zeros(n2)
        rmax = zeros(n3)
        for p in 1:n1
            pmax[p] = maximum(abs.(g1[p,:]))
        end
        for p in 1:n2
            qmax[p] = maximum(abs.(g2[p,:]))
        end
        for p in 1:n3
            rmax[p] = maximum(abs.(g3[p,:]))
        end


        mI = zeros(size(g1,2))
        @inbounds for r in 1:n3
            for q in 1:n2
                for p in 1:n1
                    @. mI += abs(v[p,q,r]) * abs.(g1[p,:]) * qmax[q] * rmax[r] * abs(c) 
                end
            end
        end

        mJ = zeros(size(g2,2))
        @inbounds for r in 1:n3
            for q in 1:n2
                for p in 1:n1
                    @. mJ += abs(v[p,q,r]) * pmax[p] * abs.(g2[q,:]) * rmax[r] * abs(c) 
                end
            end
        end

        mK = zeros(size(g3,2))
        @inbounds for r in 1:n3
            for q in 1:n2
                for p in 1:n1
                    @. mK += abs(v[p,q,r]) * pmax[p] * qmax[q] * abs.(g3[r,:]) * abs(c) 
                end
            end
        end

        for I in 1:size(g1,2)
            if abs(mI[I]) > thresh
                push!(newI,I)
            end
        end

        for J in 1:size(g2,2)
            if abs(mJ[J]) > thresh
                push!(newJ,J)
            end
        end

        for K in 1:size(g3,2)
            if abs(mK[K]) > thresh
                push!(newK,K)
            end
        end

    return newI, newJ, newK
end
#=}}}=#


"""
    upper_bound2(v::Array{Float64,4}, g1, g2, g3, g4, thresh; c::Float64=1.0)

Get upper bound on the possible values 

    max(H_IJK(L)|_L <= sum_s (sum_pqr vpqrs max(g1[p,:]) * max(g2[q,:]) * max(g3[r,:]) * |c| ) * |g4(s,L)|
"""
function upper_bound2(v::Array{Float64,4}, g1, g2, g3, g4, thresh; c::Float64=1.0)
    #={{{=#
        newI = Vector{Int16}() 
        newJ = Vector{Int16}() 
        newK = Vector{Int16}() 
        newL = Vector{Int16}() 
       
        n1 = size(v,1)
        n2 = size(v,2)
        n3 = size(v,3)
        n4 = size(v,4)

        n1 == size(g1,1) || throw(DimensionMismatch)
        n2 == size(g2,1) || throw(DimensionMismatch)
        n3 == size(g3,1) || throw(DimensionMismatch)
        n4 == size(g4,1) || throw(DimensionMismatch)
        

        pmax = zeros(n1)
        qmax = zeros(n2)
        rmax = zeros(n3)
        smax = zeros(n4)
        for p in 1:n1
            pmax[p] = maximum(abs.(g1[p,:]))
        end
        for p in 1:n2
            qmax[p] = maximum(abs.(g2[p,:]))
        end
        for p in 1:n3
            rmax[p] = maximum(abs.(g3[p,:]))
        end
        for p in 1:n4
            smax[p] = maximum(abs.(g4[p,:]))
        end
        
        tmp = 0.0

        mI = zeros(size(g1,2))
        @inbounds for s in 1:n4
            for r in 1:n3
                for q in 1:n2
                    tmp = qmax[q] * rmax[r] * smax[s] * abs(c) 
                    for p in 1:n1
                        @. mI += abs(v[p,q,r,s]) * abs.(g1[p,:]) * tmp  
                    end
                end
            end
        end

        mJ = zeros(size(g2,2))
        @inbounds for s in 1:n4
            for r in 1:n3
                for p in 1:n1
                    tmp = pmax[p] * rmax[r] * smax[s] * abs(c)
                    for q in 1:n2
                        @. mJ += abs(v[p,q,r,s]) * abs.(g2[q,:])  * tmp
                    end
                end
            end
        end

        mK = zeros(size(g3,2))
        @inbounds for s in 1:n4
            for q in 1:n2
                for p in 1:n1
                    tmp = pmax[p] * qmax[q] * smax[s] * abs(c)
                    for r in 1:n3
                        @. mK += abs(v[p,q,r,s]) * abs.(g3[r,:]) * tmp 
                    end
                end
            end
        end

        mL = zeros(size(g4,2))
        @inbounds for r in 1:n3
            for q in 1:n2
                for p in 1:n1
                    tmp =  pmax[p] * qmax[q] * rmax[r] * abs(c) 
                    for s in 1:n4
                        @. mL += abs(v[p,q,r,s]) * abs.(g4[s,:]) * tmp
                    end
                end
            end
        end

        for I in 1:size(g1,2)
            if abs(mI[I]) > thresh
                push!(newI,I)
            end
        end

        for J in 1:size(g2,2)
            if abs(mJ[J]) > thresh
                push!(newJ,J)
            end
        end

        for K in 1:size(g3,2)
            if abs(mK[K]) > thresh
                push!(newK,K)
            end
        end

        for L in 1:size(g4,2)
            if abs(mL[L]) > thresh
                push!(newL,L)
            end
        end

    return newI, newJ, newK, newL 
end
#=}}}=#


#############################################################################################################################################
# under construction
#############################################################################################################################################
function contract_matvec_new1(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig{N}, 
                                    fock_ket::FockConfig{N}, conf_ket::ClusterConfig{N}, coef_ket::MVector{R,T};
                                    thresh=1e-9) where {T,R,N}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # <:|p'|J> h(pqr) <:|q|L> <:|r|N>
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,:,conf_ket[c3.idx]]



    new_coeffs = []
    @tensor begin
        new_coeffs[p,q,M] := term.ints[p,q,r]  * gamma3[r,M] #* coef_ket
        new_coeffs[p,L,M] := new_coeffs[p,q,M] * gamma2[q,L]
        #new_coeffs[I,J,K] := ((term.ints[p,q,r]  * gamma1[p,I])  * gamma2[q,J])  * gamma3[r,K]
    end

   

    if state_sign < 0
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(gamma1,2)
    newJ = 1:size(new_coeffs,2)
    newK = 1:size(new_coeffs,3)
    out = OrderedDict{ClusterConfig{N}, MVector{R,T}}()


    _contract_fill!(out, conf_ket, new_coeffs, gamma1, coef_ket, c1.idx, c2.idx, c3.idx, newI, newJ, newK, thresh, T)
    
    return out 
end
#=}}}=#


function _contract_fill!(out, conf_ket, new_coeffs, gamma1, coeff, c1idx, c2idx, c3idx, newI, newJ, newK, thresh, T) 
#={{{=#
    N = length(conf_ket)
    cket = [conf_ket.config...]
                
    x = T(0)
    for k in newK
        cket[c3idx] = k
        
        for j in newJ
            cket[c2idx] = j
            shiftJK = (j-1)*size(new_coeffs,1) + (k-1)*size(new_coeffs,1)*size(new_coeffs,2)
            
            for i in newI
                shiftI = (i-1)*size(gamma1,1)
                
                x = 0.0 
                @inbounds @simd for p in 1:size(gamma1,1) 
                    x += new_coeffs[p + shiftJK] * gamma1[p + shiftI]
                end
                
                if any((abs(x*s) > thresh for s in coeff))
                    cket[c1idx] = i
                    out[ClusterConfig{N}(tuple(cket...))] = x*coeff 
                end
            end
        end
    end
end#=}}}=#
