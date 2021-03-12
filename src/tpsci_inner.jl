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

    #error("here")
    #if term.ops[1] == "aa" 
    #    display(term)
    #    println(mat_elem)
    #end

#    #
#    # <I|xi|J> h(xi,xi') <K|xi|L>
#    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
#    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
#    
#    mat_elem = 0.0
#    for i in 1:length(gamma1)
#        @simd for j in 1:length(gamma2)
#            mat_elem += gamma1[i]*term.ints[i,j]*gamma2[j]
#        end
#    end

#    if length(term.ops[1]) == 1 && length(term.ops[2]) == 1 
#        #
#        # <I|p'|J> h(pq) <K|q|L>
#        gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
#        gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
#        @tensor begin
#            mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
#        end

#    elseif length(term.ops[1]) == 2 && length(term.ops[2]) == 2 
#        #
#        # <I|p'q|J> v(pqrs) <K|rs|L>
#        gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,bra[c1.idx],ket[c1.idx]]
#        gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,bra[c2.idx],ket[c2.idx]]
#        mat_elem = 0.0
#        @tensor begin
#            mat_elem = (gamma1[p,q] * term.ints[p,q,r,s]) * gamma2[r,s]
#        end
#    else
#        display(term.ops)
#        println(length(term.ops[1]) , length(term.ops[2]))
#        throw(Exception)
#    end
        
    return state_sign * mat_elem
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
            tmp = gamma3[k+shift3]*gamma4[l+shift4]
            for j in 1:size(gamma2,1)
                tmp = gamma2[j+shift2]*tmp
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
    contract_matvec(    term::ClusteredTerm2B, 
                        cluster_ops::Vector{ClusterOps},
                        fock_bra, fock_ket, ket)
"""
function contract_matvec(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, 
                                    fock_ket::FockConfig, conf_ket::ClusterConfig, coef_ket::T;
                                    thresh) where T
#={{{=#
    c1 = term.clusters[1]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # <:|p'|J> h(pq) <:|q|L>

    @views gamma1 = 

    new_coeffs = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,conf_ket[c1.idx]] * coef_ket
    
    if state_sign == -1
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(new_coeffs,1)

    out = OrderedDict{ClusterConfig, SVector{1,T}}()

    _collect_significant!(out, conf_ket, new_coeffs, c1.idx,  newI,  thresh)
            

    return out 
end
#=}}}=#


"""
    contract_matvec(    term::ClusteredTerm2B, 
                        cluster_ops::Vector{ClusterOps},
                        fock_bra, fock_ket, ket)
"""
function contract_matvec(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, 
                                    fock_ket::FockConfig, conf_ket::ClusterConfig, coef_ket::T;
                                    thresh) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket) 
        

    #
    # <:|p'|J> h(pq) <:|q|L>

    #@tensor begin
    #    mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
    #end
    #mat_elem = _contract(term.ints, gamma1, gamma2)
   
    #display(fock_bra)
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,conf_ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,conf_ket[c2.idx]]
    

    #
    # tmp(q,I)   = h(p,q)' * gamma1(p,I)
    # new_coeffs(I,J) = tmp(q,I)' * gamma2(q,J)
    #
    new_coeffs = term.ints' * gamma1
    new_coeffs = new_coeffs' * gamma2
    #@tensor begin
    #    new_confs[q,I] := term.ints[p,q] * gamma1[p,I]
    #    new_confs[I,J] := new_confs[q,I] * gamma2[q,J]
    #end

    
    if state_sign == -1
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(new_coeffs,1)
    newJ = 1:size(new_coeffs,2)

    out = OrderedDict{ClusterConfig, SVector{1,T}}()

#    a = (4,5,6,7,8,89)
#    b = [4,5,6,7,8,89]
#    c = SVector{6}(b)
#
#    @btime hash($a)
#    @btime hash($b)
#    @btime hash($c)
    #@btime hash((4,5,6,7,8,89))
    #@btime hash([4,5,6,7,8,89])
    #@btime _collect_significant!($out, $conf_ket, $new_coeffs, $c1.idx, $c2.idx, $newI, $newJ, $thresh)
    _collect_significant!(out, conf_ket, new_coeffs, c1.idx, c2.idx, newI, newJ, thresh)
            

    return out 
end
#=}}}=#

"""
    contract_matvec(    term::ClusteredTerm3B, 
                        cluster_ops::Vector{ClusterOps},
                        fock_bra, fock_ket, ket)
"""
function contract_matvec(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra::FockConfig, 
                                    fock_ket::FockConfig, conf_ket::ClusterConfig, coef_ket::T;
                                    thresh) where T
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
        new_coeffs[p,q,M] := term.ints[p,q,r]  * gamma3[r,M]
        new_coeffs[p,L,M] := new_coeffs[p,q,M] * gamma2[q,L]
        new_coeffs[J,L,M] := new_coeffs[p,L,M] * gamma1[p,J]
    end

    
    if state_sign == -1
        new_coeffs .= -new_coeffs
    end 

    newI = 1:size(new_coeffs,1)
    newJ = 1:size(new_coeffs,2)
    newK = 1:size(new_coeffs,3)

    out = OrderedDict{ClusterConfig, SVector{1,T}}()

    _collect_significant!(out, conf_ket, new_coeffs, c1.idx, c2.idx, c2.idx, newI, newJ, newK, thresh)
            

    return out 
end
#=}}}=#


function _collect_significant!(out, conf_ket, new_coeffs, c1idx, newI, thresh) 
    #={{{=#
    N = length(conf_ket)
    #test = Dict()
    cket = [conf_ket.config...]
    @inbounds for I::Int16 in newI
        if abs(new_coeffs[I]) > thresh
            cket[c1idx] = I
            out[ClusterConfig{N}(tuple(cket...))] = [new_coeffs[I]]
        end
    end
end
#=}}}=#

function _collect_significant!(out, conf_ket, new_coeffs, c1idx, c2idx, newI, newJ, thresh) 
#={{{=#
    N = length(conf_ket)
    #test = Dict()
    cket = [conf_ket.config...]
    @inbounds for J::Int16 in newJ
        cket[c2idx] = J
        for I::Int16 in newI
            if abs(new_coeffs[I,J]) > thresh
                cket[c1idx] = I
                out[ClusterConfig{N}(tuple(cket...))] = [new_coeffs[I,J]]
                #test[ClusterConfig{N}(tuple(cket...))] = [new_coeffs[I,J]]
                
                #conf_new = replace(conf_ket, (c1idx, c2idx), (I,J))
                #out[conf_new] = [new_coeffs[I,J]]
                
                #test[cket] = [new_coeffs[I,J]]
            end
            #conf_new = replace(conf_ket, (c1idx, c2idx), (I,J))
            #out[conf_new] = [new_coeffs[I,J]]
        end
    end
end
#=}}}=#

function _collect_significant!(out, conf_ket, new_coeffs, c1idx, c2idx, c3idx, newI, newJ, newK, thresh) 
    #={{{=#
    N = length(conf_ket)
    #test = Dict()
    cket = [conf_ket.config...]
    @inbounds for K::Int16 in newK
        cket[c3idx] = K
        for J::Int16 in newJ
            cket[c2idx] = J
            for I::Int16 in newI
                if abs(new_coeffs[I,J]) > thresh
                    cket[c1idx] = I
                    out[ClusterConfig{N}(tuple(cket...))] = [new_coeffs[I,J]]
                end
            end
        end
    end
end
#=}}}=#

