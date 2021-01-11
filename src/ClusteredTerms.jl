
"""
	ops::Vector{String}
	delta::Vector{Int}
	ints


input:
- delta = list of change of Na,Nb,state
			e.g., [(-1,-1),(1,1),(0,0)] means alpha and beta transition
			from cluster 1 to 2, cluster 3 is fock diagonal
- ops   = list of operators
			e.g., ["ab","AB",""]

- ints  = tensor containing the integrals for this block
			e.g., ndarray([p,q,r,s]) where p,q are in 1 and r,s are in 2

- data contained in object
		active: list of clusters which have non-identity operators
			this includes fock-diagonal couplings,
			e.g., ["Aa","","Bb"] would have active = [0,2]
"""
abstract type ClusteredTerm end

struct ClusteredTerm1B <: ClusteredTerm
    ops::Tuple{String}
    delta::TransferConfig
    clusters::Tuple{Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm2B <: ClusteredTerm
    ops::Tuple{String,String}
    #delta::Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}}
    delta::TransferConfig
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm3B <: ClusteredTerm
    ops::Vector{String}
    delta::TransferConfig
    #active::Vector{Int16}
    clusters::Vector{Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm4B <: ClusteredTerm
    ops::Vector{String}
    delta::TransferConfig
    #active::Vector{Int16}
    clusters::Vector{Cluster}
    ints::Array{Float64}
end

#function ClusteredTerm(ops, delta::Vector{Tuple{Int}}, clusters, ints)
#end

function Base.display(t::ClusteredTerm1B)
    @printf( " 1B: %2i    :", t.clusters[1].idx)
    println(t.ops, size(t.ints))
end
function Base.display(t::ClusteredTerm2B)
    @printf( " 2B: %2i %2i :", t.clusters[1].idx, t.clusters[2].idx)
    println(t.ops, " ints: ", size(t.ints))
end


"""
    extract_terms(ints::InCoreInts, clusters)

Extract all ClusteredTerm types from a given 1e integral tensor 
and a list of clusters
"""
function extract_ClusteredTerms(ints::InCoreInts, clusters)
    norb = 0
    for ci in clusters
        norb += length(ci)
    end
    length(size(ints.h1)) == 2 || throw(Exception)
    size(ints.h1,1) == norb || throw(Exception)
    size(ints.h1,2) == norb || throw(Exception)

    terms = Dict{TransferConfig,Vector{ClusteredTerm}}()
    #terms = Dict{Vector{Tuple{Int16,Int16}},Vector{ClusteredTerm}}()
    #terms = Dict{Tuple,Vector{ClusteredTerm}}()
    n_clusters = length(clusters)
    ops_a = Array{String}(undef,n_clusters)
    ops_b = Array{String}(undef,n_clusters)
    fill!(ops_a,"")
    fill!(ops_b,"")
  
    zero_fock::TransferConfig = [(0,0) for i in clusters]
    #zero_fock::Vector{Tuple{Int16,Int16}} = [(0,0) for i in clusters]
    #zero_fock = Tuple([(0,0) for i in clusters])
    terms[zero_fock] = Vector{ClusteredTerm}()
    
    for ci in clusters
        # instead of forming p'q and p'q'sr just precontract and keep them in 
        # ClusterOps
        term = ClusteredTerm1B(("H",), ((0,0),), (ci,), zeros(1,1))
        push!(terms[zero_fock],term)
    end
    for ci in clusters
        for cj in clusters
            ci < cj || continue
            
            #
            # p'q where p is in ci and q is in cj
            ints_i = copy(view(ints.h1, ci.orb_list, cj.orb_list))
           
            term = ClusteredTerm2B(("A","a"), [(1,0),(-1,0)], (ci, cj), ints_i)
            fock = deepcopy(zero_fock)
            fock[ci.idx] = (fock[ci.idx][1]+1, fock[ci.idx][2])
            fock[cj.idx] = (fock[cj.idx][1]-1, fock[cj.idx][2])
            terms[fock] = [term]
            
            term = ClusteredTerm2B(("a","A"), [(-1,0),(1,0)], (ci, cj), -ints_i)
            fock = deepcopy(zero_fock)
            fock[ci.idx] = (fock[ci.idx][1]-1, fock[ci.idx][2])
            fock[cj.idx] = (fock[cj.idx][1]+1, fock[cj.idx][2])
            terms[fock] = [term]
            
            term = ClusteredTerm2B(("B","b"), [(0,1),(0,-1)], (ci, cj), ints_i)
            fock = deepcopy(zero_fock)
            fock[ci.idx] = (fock[ci.idx][1], fock[ci.idx][2]+1)
            fock[cj.idx] = (fock[cj.idx][1], fock[cj.idx][2]-1)
            terms[fock] = [term]
            
            term = ClusteredTerm2B(("b","B"), [(0,-1),(0,1)], (ci, cj), -ints_i)
            fock = deepcopy(zero_fock)
            fock[ci.idx] = (fock[ci.idx][1], fock[ci.idx][2]-1)
            fock[cj.idx] = (fock[cj.idx][1], fock[cj.idx][2]+1)
            terms[fock] = [term]
            

            #
            # (pr|qs) p'q'sr        pr have same spin, and qs have same spin
            #  V     Op
            #  pqsr ->      -> SS(12)  ->  OS(12)      v
            #  1122:    PQsr   AA,aa       AB,ba       1212
            #  1212:   -PsQr  -Aa,Aa      -Ab,Ba       1221 = 1212
            #  1221:    PrQs   Aa,Aa       Aa,Bb       1122 
            #  2112:    QsPr   Aa,Aa       Bb,Aa       2211 = 1122
            #  2121:   -QrPs  -Aa,Aa      -Ba,Ab       2112 = 1212
            #  2211:    srPQ   aa,AA       ba,AB       2121 = 1212
            #
            #   4*6 = 24 terms 12 SS and 12 OS terms
            #
            #
            #   (AA,aa): (12|12)    1
            #   (Aa,Aa):2(11||22)   4
            #   (aa,AA): (12|12)    1
            #   (BB,bb): (12|12)    1
            #   (Bb,Bb):2(11||22)   4
            #   (bb,BB): (12|12)    1
            #   12 total
            #
            #   (E,E): 8(11|22) - 4(12|12) = (Aa+Bb, Aa+Bb) = AaAa(4) + BbBb(4) + AaBb(2) + BbAa(2) = 12 terms
            #   (AA,aa): (12|12)    1
            #   (aa,AA): (12|12)    1
            #   (BB,bb): (12|12)    1
            #   (bb,BB): (12|12)    1
            
            #   (AB,ba):2(12|12)    2
            #   (Ab,Ba):-(12|12)    2
            #   (Ba,Ab):-(12|12)    2
            #   (ba,AB):2(12|12)    2
           
            #x  (BA,ab): (12|12)    
            #x  (ab,BA): (12|12)    
            
            
            # <IJ| sum_pr^A sum_qs^B (pr|qs) p'q'sr |I'J'>
            # <I|p'r|I'> (pr|qs) <J|q's|J'>
            #
            #
            v1122 = .5*copy(view(ints.h2, ci.orb_list, ci.orb_list, cj.orb_list, cj.orb_list))
            v1212 = .5*copy(view(ints.h2, ci.orb_list, cj.orb_list, ci.orb_list, cj.orb_list))
            #
            ## now transpose 1212 so that all terms can be contracted with first cluster (ci.idx<cj.idx) first then second with fast index
            v1212 = copy(permutedims(v1212, [3,1,2,4]))

            tmp = 1
            if tmp == 1 
                term = ClusteredTerm2B(("E1","E1"), [(0,0),(0,0)], (ci, cj), 2*v1122) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Aa","Aa"), [(0,0),(0,0)], (ci, cj), -2*v1212) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Bb","Bb"), [(0,0),(0,0)], (ci, cj), -2*v1212) 
                push!(terms[zero_fock],term)
            elseif tmp == 2 
                term = ClusteredTerm2B(("Aa","Aa"), [(0,0),(0,0)], (ci, cj), -2*v1212) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Bb","Bb"), [(0,0),(0,0)], (ci, cj), -2*v1212) 
                push!(terms[zero_fock],term)
                
                term = ClusteredTerm2B(("Aa","Aa"), [(0,0),(0,0)], (ci, cj), 2*v1122 - 0*v1212) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Bb","Bb"), [(0,0),(0,0)], (ci, cj), 2*v1122 - 0*v1212) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Bb","Aa"), [(0,0),(0,0)], (ci, cj), 2*v1122) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Aa","Bb"), [(0,0),(0,0)], (ci, cj), 2*v1122) 
                push!(terms[zero_fock],term)
            end
            if false
                term = ClusteredTerm2B(("AA","aa"), [(2,0),(-2,0)], (ci, cj), v1212); 
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]+2, fock[ci.idx][2])
                fock[cj.idx] = (fock[cj.idx][1]-2, fock[cj.idx][2])
                terms[fock] = [term]

                term = ClusteredTerm2B(("BB","bb"), [(0,2),(0,-2)], (ci, cj), v1212); 
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1], fock[ci.idx][2]+2)
                fock[cj.idx] = (fock[cj.idx][1], fock[cj.idx][2]-2)
                terms[fock] = [term]

                term = ClusteredTerm2B(("aa","AA"), [(-2,0),(2,0)], (ci, cj), v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]-2, fock[ci.idx][2])
                fock[cj.idx] = (fock[cj.idx][1]+2, fock[cj.idx][2])
                terms[fock] = [term]

                term = ClusteredTerm2B(("bb","BB"), [(0,-2),(0,2)], (ci, cj), v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1], fock[ci.idx][2]-2)
                fock[cj.idx] = (fock[cj.idx][1], fock[cj.idx][2]+2)
                terms[fock] = [term]

                term = ClusteredTerm2B(("AB","ba"), [(1,1),(-1,-1)], (ci, cj), 2*v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]+1, fock[ci.idx][2]+1)
                fock[cj.idx] = (fock[cj.idx][1]-1, fock[cj.idx][2]-1)
                terms[fock] = [term]

                term = ClusteredTerm2B(("ba","AB"), [(-1,-1),(1,1)], (ci, cj), 2*v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]-1, fock[ci.idx][2]-1)
                fock[cj.idx] = (fock[cj.idx][1]+1, fock[cj.idx][2]+1)
                terms[fock] = [term]

                term = ClusteredTerm2B(("Ab","Ba"), [(1,-1),(-1,1)], (ci, cj),-2*v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]+1, fock[ci.idx][2]-1)
                fock[cj.idx] = (fock[cj.idx][1]-1, fock[cj.idx][2]+1)
                terms[fock] = [term]

                term = ClusteredTerm2B(("Ba","Ab"), [(-1,1),(1,-1)], (ci, cj),-2*v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]-1, fock[ci.idx][2]+1)
                fock[cj.idx] = (fock[cj.idx][1]+1, fock[cj.idx][2]-1)
                terms[fock] = [term]
            end
            
        end
    end
    return terms
end


function contract_matrix_element(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
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
    state_sign = 1
    n_elec_hopped = 0
    #println(c1.idx, " ",  c2.idx)
    if length(term.ops[2]) % 2 != 0
        for ci in c1.idx:c2.idx-1
            n_elec_hopped += fock_ket[ci][1] + fock_ket[ci][2]
        end
        if n_elec_hopped % 2 != 0
            state_sign = -1
        end
    end

   
        
    mat_elem = 0.0


    if length(term.ops[1]) == 1 && length(term.ops[2]) == 1 
        #
        # <I|p'|J> h(pq) <K|q|L>
        gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
        gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
        @tensor begin
            mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
        end

    elseif length(term.ops[1]) == 2 && length(term.ops[2]) == 2 
        #
        # <I|p'q|J> v(pqrs) <K|rs|L>
        gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,:,bra[c1.idx],ket[c1.idx]]
        gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,:,bra[c2.idx],ket[c2.idx]]
        mat_elem = 0.0
        @tensor begin
            mat_elem = (gamma1[p,q] * term.ints[p,q,r,s]) * gamma2[r,s]
        end
    else
        display(term.ops)
        println(length(term.ops[1]) , length(term.ops[2]))
        throw(Exception)
    end
        
    return state_sign * mat_elem
end

"""
    contract_matrix_element(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)

Contraction for local (1body) terms. No contraction is needed,
just a lookup from the correct operator
"""
function contract_matrix_element(   term::ClusteredTerm1B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
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

    return cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][bra[c1.idx],ket[c1.idx]]
end


function print_fock_sectors(sector::Vector{Tuple{T,T}}) where T<:Integer
    print("  ")
    for ci in sector
        @printf("(%iα,%iβ)", ci[1],ci[2])
    end
    println()
end
