using Combinatorics
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
    ops::Tuple{String,String,String}
    delta::TransferConfig
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster,Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm4B <: ClusteredTerm
    ops::Tuple{String,String,String,String}
    delta::TransferConfig
    clusters::Tuple{Cluster,Cluster,Cluster,Cluster}
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


function bubble_sort(inp)
    cmpcount, swapcount = 0, 0
    blist = copy(inp)
    bperm = collect(1:length(inp))
    for j in 1:length(blist)
        for i in 1:(length(blist)-j)
            cmpcount += 1
            if blist[i] > blist[i+1]
                swapcount += 1
                blist[i], blist[i+1] = blist[i+1], blist[i]
                bperm[i], bperm[i+1] = bperm[i+1], bperm[i]
            end
        end
    end
    return bperm, swapcount
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

    


    if false
        for ci in clusters
            for cj in clusters
                ci < cj || continue
                #={{{=#
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
                v1212 = copy(permutedims(v1212, [1,3,4,2]))

                v1122 = copy(reshape(v1122, (size(v1122,1)*size(v1122,2), size(v1122,3)*size(v1122,4))))
                v1212 = copy(reshape(v1212, (size(v1212,1)*size(v1212,2), size(v1212,3)*size(v1212,4))))

                # todo (maybe, need to test): 
                #
                # change the shape to a matrix to vectorize the contraction better
                # gamma(i) v(i,j) gamma(j)

                term = ClusteredTerm2B(("E1","E1"), [(0,0),(0,0)], (ci, cj), 2*v1122) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Aa","Aa"), [(0,0),(0,0)], (ci, cj), -2*v1212) 
                push!(terms[zero_fock],term)
                term = ClusteredTerm2B(("Bb","Bb"), [(0,0),(0,0)], (ci, cj), -2*v1212) 
                push!(terms[zero_fock],term)

                #
                #   AA,aa terms
                #
                #   (pr|qs) p'q'sr     1={pq} 2={rs}
                #   transpose =>   v(pq,sr) gamma(pq) gamma(sr)
                #
                #   don't need to do again, same as above
                #v1212 = .5*copy(view(ints.h2, ci.orb_list, cj.orb_list, ci.orb_list, cj.orb_list))

                term = ClusteredTerm2B(("AA","aa"), [(2,0),(-2,0)], (ci, cj), v1212); 
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]+2, fock[ci.idx][2])
                fock[cj.idx] = (fock[cj.idx][1]-2, fock[cj.idx][2])
                terms[fock] = [term]

                term = ClusteredTerm2B(("aa","AA"), [(-2,0),(2,0)], (ci, cj), v1212)
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]-2, fock[ci.idx][2])
                fock[cj.idx] = (fock[cj.idx][1]+2, fock[cj.idx][2])

                term = ClusteredTerm2B(("BB","bb"), [(0,2),(0,-2)], (ci, cj), v1212); 
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1], fock[ci.idx][2]+2)
                fock[cj.idx] = (fock[cj.idx][1], fock[cj.idx][2]-2)
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



                #
                # (pr|qs) p'q'sr        pr have same spin, and qs have same spin
                #  V     Op
                #  pqsr ->      ->  SS(12)  ->  OS(12)      v    =  v       transpose store each as p,qsr
                #  1222:    P,Qsr   A,Aaa       A,Bba       prqs    prqs   
                #  2122:    P,Qsr   A,Aaa       B,Aba      -qrps   -psqr   
                #  2212:    p,QSr   a,AAa       b,ABa       qrsp    psqr      
                #  2221:    p,QSr   a,AAa       a,ABb      -qpsr   -pqsr 
                #                                                           transpose store each as pqs,r 
                #  1112:    PQs,r   AAa,a       ABb,a       prqs    qspr   
                #  1121:    PQs,r   AAa,a       ABa,b      -psqr   -psqr   
                #  1211:    Pqs,R   Aaa,A       Aba,B       psrq    psqr   
                #  2111:    Pqs,R   Aaa,a       Bba,A      -rspq   -pqsr    
                #                                                       
                #
                #
                #
                #  2111:    Qsr,P   Aaa,A       Bba,A      -prqs   -qsrp   
                #  1211:    Qsr,P   Aaa,A       Aba,B       qrps    qrsp   
                #  1121:    QSr,p   AAa,a       ABa,b      -qrsp   -qrsp
                #  1112:    QSr,p   AAa,a       ABb,a       qpsr    srqp
                #
                scale = 2
                v_prqs = scale*copy(view(ints.h2, ci.orb_list, cj.orb_list, cj.orb_list, cj.orb_list))
                v_psqr = scale*copy(view(ints.h2, ci.orb_list, cj.orb_list, cj.orb_list, cj.orb_list))
                v_pqsr = scale*copy(view(ints.h2, ci.orb_list, cj.orb_list, cj.orb_list, cj.orb_list))

                w_qspr = scale*copy(view(ints.h2, ci.orb_list, cj.orb_list, ci.orb_list, ci.orb_list))
                w_psqr = scale*copy(view(ints.h2, ci.orb_list, cj.orb_list, ci.orb_list, ci.orb_list))
                w_pqsr = scale*copy(view(ints.h2, ci.orb_list, cj.orb_list, ci.orb_list, ci.orb_list))

                # permute indices so the align with pqs,r
                v_prqs = copy(permutedims(v_prqs, [1,3,4,2]))
                v_psqr = copy(permutedims(v_psqr, [1,3,2,4]))
                v_pqsr = copy(permutedims(v_pqsr, [1,2,3,4]))

                # permute indices so the align with pqs,r 
                w_qspr = copy(permutedims(w_qspr, [4,1,2,3]))
                w_psqr = copy(permutedims(w_psqr, [4,1,3,2]))
                w_pqsr = copy(permutedims(w_pqsr, [4,3,1,2]))

                v_prqs = copy(reshape(v_prqs, (size(v_prqs,1),size(v_prqs,2)*size(v_prqs,3)*size(v_prqs,4))))
                v_psqr = copy(reshape(v_psqr, (size(v_psqr,1),size(v_psqr,2)*size(v_psqr,3)*size(v_psqr,4))))
                v_pqsr = copy(reshape(v_pqsr, (size(v_pqsr,1),size(v_pqsr,2)*size(v_pqsr,3)*size(v_pqsr,4))))

                w_qspr = copy(reshape(w_qspr, (size(w_qspr,1)*size(w_qspr,2)*size(w_qspr,3),size(w_qspr,4))))
                w_psqr = copy(reshape(w_psqr, (size(w_psqr,1)*size(w_psqr,2)*size(w_psqr,3),size(w_psqr,4))))
                w_pqsr = copy(reshape(w_pqsr, (size(w_pqsr,1)*size(w_pqsr,2)*size(w_pqsr,3),size(w_pqsr,4))))

                term = ClusteredTerm2B(("Aaa","A"), [(-1,0),(1,0)], (ci, cj), 0*w_psqr - w_pqsr); 
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]-1, fock[ci.idx][2]+0)
                fock[cj.idx] = (fock[cj.idx][1]+1, fock[cj.idx][2]-0)
                push!(terms[fock], term)

                term = ClusteredTerm2B(("AAa","a"), [(1,0),(-1,0)], (ci, cj), w_qspr + 0*w_psqr); 
                fock = deepcopy(zero_fock)
                fock[ci.idx] = (fock[ci.idx][1]+1, fock[ci.idx][2]+0)
                fock[cj.idx] = (fock[cj.idx][1]-1, fock[cj.idx][2]-0)
                push!(terms[fock], term)

                if false 

                    term = ClusteredTerm2B(("A","Aaa"), [(1,0),(-1,0)], (ci, cj), v_prqs - v_psqr); 
                    fock = deepcopy(zero_fock)
                    fock[ci.idx] = (fock[ci.idx][1]+1, fock[ci.idx][2]+0)
                    fock[cj.idx] = (fock[cj.idx][1]-1, fock[cj.idx][2]-0)
                    push!(terms[fock], term)

                    term = ClusteredTerm2B(("a","AAa"), [(-1,0),(1,0)], (ci, cj), v_psqr - v_pqsr); 
                    fock = deepcopy(zero_fock)
                    fock[ci.idx] = (fock[ci.idx][1]-1, fock[ci.idx][2]+0)
                    fock[cj.idx] = (fock[cj.idx][1]+1, fock[cj.idx][2]-0)
                    push!(terms[fock], term)

                    term = ClusteredTerm2B(("B","Bbb"), [(0,1),(0,-1)], (ci, cj), v_prqs - v_psqr); 
                    fock = deepcopy(zero_fock)
                    fock[ci.idx] = (fock[ci.idx][1]+0, fock[ci.idx][2]+1)
                    fock[cj.idx] = (fock[cj.idx][1]-0, fock[cj.idx][2]-1)
                    push!(terms[fock], term)

                    term = ClusteredTerm2B(("b","BBb"), [(0,-1),(0,1)], (ci, cj), v_psqr - v_pqsr); 
                    fock = deepcopy(zero_fock)
                    fock[ci.idx] = (fock[ci.idx][1]-0, fock[ci.idx][2]-1)
                    fock[cj.idx] = (fock[cj.idx][1]+0, fock[cj.idx][2]+1)
                    push!(terms[fock], term)

                    term = ClusteredTerm2B(("Bbb","B"), [(0,-1),(0,1)], (ci, cj), v_qrsp - v_qsrp); 
                    fock = deepcopy(zero_fock)
                    fock[ci.idx] = (fock[ci.idx][1]+0, fock[ci.idx][2]-1)
                    fock[cj.idx] = (fock[cj.idx][1]+0, fock[cj.idx][2]+1)
                    push!(terms[fock], term)

                    term = ClusteredTerm2B(("BBb","b"), [(0,1),(0,-1)], (ci, cj), v_srqp - v_qrsp); 
                    fock = deepcopy(zero_fock)
                    fock[ci.idx] = (fock[ci.idx][1]+0, fock[ci.idx][2]+1)
                    fock[cj.idx] = (fock[cj.idx][1]-0, fock[cj.idx][2]-1)
                    push!(terms[fock], term)

                end

            end
        end#=}}}=#
    end
   
    # get 2-body 1-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                #={{{=#
                i = ci.idx
                j = cj.idx

                i < j || continue

                spin_cases =[["A","a"],
                             ["B","b"]
                            ]

                fock_cases =[[(1,0),(-1,0)],
                             [(0,1),(0,-1)]
                            ]

                termstr = []
                append!(termstr,unique(permutations([ci,cj])))
                #append!(termstr,unique(permutations([ci,cj,cj,cj])))
                #append!(termstr,unique(permutations([ci,ci,ci,cj])))

                #
                #   (pr|qs) p'q'sr
                #
                for term in termstr 

                    #
                    #   find permutations and sign needed to sort the indices 
                    #   such that clusters increase from left to right
                    perm, countswap = bubble_sort(term) 
                    perm == sortperm(term, alg=MergeSort)|| throw(Exception) 

                    permsign = 1
                    if countswap%2 != 0 
                        permsign = -1
                    end

                    #vprqs = view(ints.h2,term[1].orb_list, term[2].orb_list, term[3].orb_list, term[4].orb_list) 
                    hpq = view(ints.h1,term[1].orb_list, term[2].orb_list) 

                    #
                    # now align (pqsr) ints so that they align with indices from operators after sorting
                    # in this ordering, one can simply contract by sum(v .* d)
                    h = permsign .* permutedims(hpq,perm)


                    for sidx in 1:length(spin_cases)
                        oper = spin_cases[sidx][perm]
                        fock = fock_cases[sidx][perm]
                        oper1 = ""
                        oper2 = ""
                        fock1 = [0,0]
                        fock2 = [0,0]
                        for cidx in 1:length(term[perm])
                            if term[perm][cidx] == ci
                                oper1 *= oper[cidx]
                                fock1 .+= fock[cidx]
                            elseif term[perm][cidx] == cj
                                oper2 *= oper[cidx]
                                fock2 .+= fock[cidx]
                            else
                                throw(Exception)
                            end
                        end

                        clusteredterm = ClusteredTerm2B((oper1,oper2), [Tuple(fock1),Tuple(fock2)], (ci, cj), h)
                        #display(clusteredterm)
                        focktrans = deepcopy(zero_fock)
                        focktrans[ci.idx] = Tuple(fock1)
                        focktrans[cj.idx] = Tuple(fock2)
                        if haskey(terms,focktrans)
                            push!(terms[focktrans], clusteredterm)
                        else
                            terms[focktrans] = [clusteredterm]
                        end

                    end
                end
            end
        end
        #=}}}=#
    end
    
    # get 2-body 2-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                #={{{=#
                i = ci.idx
                j = cj.idx

                i < j || continue

                spin_cases =[["A","A","a","a"],
                             ["B","B","b","b"],
                             ["A","B","b","a"],
                             ["B","A","a","b"]
                            ]

                fock_cases =[[(1,0),(1,0),(-1,0),(-1,0)],
                             [(0,1),(0,1),(0,-1),(0,-1)],
                             [(1,0),(0,1),(0,-1),(-1,0)],
                             [(0,1),(1,0),(-1,0),(0,-1)]
                            ]


                termstr = []
                append!(termstr,unique(permutations([ci,ci,cj,cj])))
                #append!(termstr,unique(permutations([ci,cj,cj,cj])))
                #append!(termstr,unique(permutations([ci,ci,ci,cj])))

                #
                #   (pr|qs) p'q'sr
                #
                for term in termstr 

                    #
                    #   find permutations and sign needed to sort the indices 
                    #   such that clusters increase from left to right
                    perm, countswap = bubble_sort(term) 
                    perm == sortperm(term, alg=MergeSort)|| error("problem with bubble_sort") 

                    permsign = 1
                    if countswap%2 != 0 
                        permsign = -1
                    end

                    vprqs = view(ints.h2,term[1].orb_list, term[4].orb_list, term[2].orb_list, term[3].orb_list) 
                    #
                    # align (prqs) ints so that they align with indices from operators before sorting
                    vpqsr = permutedims(vprqs,[1,3,4,2])

                    #
                    # now align (pqsr) ints so that they align with indices from operators after sorting
                    # in this ordering, one can simply contract by sum(v .* d)
                    v = (.5 * permsign) .* permutedims(vpqsr,perm)

                    #
                    # now reshape ints so they can be contracted like gamma(pqr) V(pqr,s) gamma(s)
                    newshape = [1,1]

                    for cidx in 1:length(term[perm])
                        if term[perm][cidx] == ci
                            newshape[1] *= size(v,cidx)
                        elseif term[perm][cidx] == cj
                            newshape[2] *= size(v,cidx)
                        else
                            throw(Exception)
                        end
                    end
                    v = copy(reshape(v,newshape...))

                    for sidx in 1:length(spin_cases)
                        oper = spin_cases[sidx][perm]
                        fock = fock_cases[sidx][perm]
                        oper1 = ""
                        oper2 = ""
                        fock1 = [0,0]
                        fock2 = [0,0]
                        for cidx in 1:length(term[perm])
                            if term[perm][cidx] == ci
                                oper1 *= oper[cidx]
                                fock1 .+= fock[cidx]
                            elseif term[perm][cidx] == cj
                                oper2 *= oper[cidx]
                                fock2 .+= fock[cidx]
                            else
                                throw(Exception)
                            end
                        end

                        clusteredterm = ClusteredTerm2B((oper1,oper2), [Tuple(fock1),Tuple(fock2)], (ci, cj), v)
                        #display(clusteredterm)
                        focktrans = deepcopy(zero_fock)
                        focktrans[ci.idx] = Tuple(fock1)
                        focktrans[cj.idx] = Tuple(fock2)
                        if haskey(terms,focktrans)
                            push!(terms[focktrans], clusteredterm)
                        else
                            terms[focktrans] = [clusteredterm]
                        end

                    end

                    #all(isapprox(vpqsr,0.0)) || continue
                end

                #=}}}=#
            end
        end
    end

    # get 3-body 2-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    #={{{=#
                    i = ci.idx
                    j = cj.idx
                    k = ck.idx

                    i < j < k || continue

                    spin_cases =[["A","A","a","a"],
                                 ["B","B","b","b"],
                                 ["A","B","b","a"],
                                 ["B","A","a","b"]
                                ]

                    fock_cases =[[(1,0),(1,0),(-1,0),(-1,0)],
                                 [(0,1),(0,1),(0,-1),(0,-1)],
                                 [(1,0),(0,1),(0,-1),(-1,0)],
                                 [(0,1),(1,0),(-1,0),(0,-1)]
                                ]


                    termstr = []
                    append!(termstr,unique(permutations([ci,ci,cj,ck])))
                    append!(termstr,unique(permutations([ci,cj,cj,ck])))
                    append!(termstr,unique(permutations([ci,cj,ck,ck])))

                    #
                    #   (pr|qs) p'q'sr
                    #
                    for term in termstr 

                        #
                        #   find permutations and sign needed to sort the indices 
                        #   such that clusters increase from left to right
                        perm, countswap = bubble_sort(term) 
                        perm == sortperm(term, alg=MergeSort)|| error("problem with bubble_sort") 

                        permsign = 1
                        if countswap%2 != 0 
                            permsign = -1
                        end

                        vprqs = view(ints.h2,term[1].orb_list, term[4].orb_list, term[2].orb_list, term[3].orb_list) 
                        #
                        # align (prqs) ints so that they align with indices from operators before sorting
                        vpqsr = permutedims(vprqs,[1,3,4,2])

                        #
                        # now align (pqsr) ints so that they align with indices from operators after sorting
                        # in this ordering, one can simply contract by sum(v .* d)
                        v = (.5 * permsign) .* permutedims(vpqsr,perm)

                        #
                        # now reshape ints so they can be contracted like gamma(pqr) V(pqr,s) gamma(s)
                        newshape = [1,1,1]

                        for cidx in 1:length(term[perm])
                            if term[perm][cidx] == ci
                                newshape[1] *= size(v,cidx)
                            elseif term[perm][cidx] == cj
                                newshape[2] *= size(v,cidx)
                            elseif term[perm][cidx] == ck
                                newshape[3] *= size(v,cidx)
                            else
                                throw(Exception)
                            end
                        end
                        v = copy(reshape(v,newshape...))

                        for sidx in 1:length(spin_cases)
                            oper = spin_cases[sidx][perm]
                            fock = fock_cases[sidx][perm]
                            oper1 = ""
                            oper2 = ""
                            oper3 = ""
                            fock1 = [0,0]
                            fock2 = [0,0]
                            fock3 = [0,0]
                            for cidx in 1:length(term[perm])
                                if term[perm][cidx] == ci
                                    oper1 *= oper[cidx]
                                    fock1 .+= fock[cidx]
                                elseif term[perm][cidx] == cj
                                    oper2 *= oper[cidx]
                                    fock2 .+= fock[cidx]
                                elseif term[perm][cidx] == ck
                                    oper3 *= oper[cidx]
                                    fock3 .+= fock[cidx]
                                else
                                    throw(Exception)
                                end
                            end

                            clusteredterm = ClusteredTerm3B((oper1,oper2,oper3), [Tuple(fock1),Tuple(fock2),Tuple(fock3)], (ci, cj, ck), v)
                            #display(clusteredterm)
                            focktrans = deepcopy(zero_fock)
                            focktrans[ci.idx] = Tuple(fock1)
                            focktrans[cj.idx] = Tuple(fock2)
                            focktrans[ck.idx] = Tuple(fock3)
                            if haskey(terms,focktrans)
                                push!(terms[focktrans], clusteredterm)
                            else
                                terms[focktrans] = [clusteredterm]
                            end
                        end
                        #all(isapprox(vpqsr,0.0)) || continue
                    end

                    #=}}}=#
                end
            end
        end
    end

    # get 4-body 2-electron terms
    if true 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    for cl in clusters
                        #={{{=#
                        i = ci.idx
                        j = cj.idx
                        k = ck.idx
                        l = cl.idx

                        i < j < k < l|| continue

                        spin_cases =[["A","A","a","a"],
                                     ["B","B","b","b"],
                                     ["A","B","b","a"],
                                     ["B","A","a","b"]
                                    ]

                        fock_cases =[[(1,0),(1,0),(-1,0),(-1,0)],
                                     [(0,1),(0,1),(0,-1),(0,-1)],
                                     [(1,0),(0,1),(0,-1),(-1,0)],
                                     [(0,1),(1,0),(-1,0),(0,-1)]
                                    ]


                        termstr = []
                        append!(termstr,unique(permutations([ci,cj,ck,cl])))

                        #
                        #   (pr|qs) p'q'sr
                        #
                        for term in termstr 

                            #
                            #   find permutations and sign needed to sort the indices 
                            #   such that clusters increase from left to right
                            perm, countswap = bubble_sort(term) 
                            perm == sortperm(term, alg=MergeSort)|| error("problem with bubble_sort") 

                            permsign = 1
                            if countswap%2 != 0 
                                permsign = -1
                            end

                            vprqs = view(ints.h2,term[1].orb_list, term[4].orb_list, term[2].orb_list, term[3].orb_list) 
                            #
                            # align (prqs) ints so that they align with indices from operators before sorting
                            vpqsr = permutedims(vprqs,[1,3,4,2])

                            #
                            # now align (pqsr) ints so that they align with indices from operators after sorting
                            # in this ordering, one can simply contract by sum(v .* d)
                            v = (.5 * permsign) .* permutedims(vpqsr,perm)

                            #
                            # no reshape needed 

                            for sidx in 1:length(spin_cases)
                                oper = spin_cases[sidx][perm]
                                fock = fock_cases[sidx][perm]
                                oper1 = ""
                                oper2 = ""
                                oper3 = ""
                                oper4 = ""
                                fock1 = [0,0]
                                fock2 = [0,0]
                                fock3 = [0,0]
                                fock4 = [0,0]
                                for cidx in 1:length(term[perm])
                                    if term[perm][cidx] == ci
                                        oper1 *= oper[cidx]
                                        fock1 .+= fock[cidx]
                                    elseif term[perm][cidx] == cj
                                        oper2 *= oper[cidx]
                                        fock2 .+= fock[cidx]
                                    elseif term[perm][cidx] == ck
                                        oper3 *= oper[cidx]
                                        fock3 .+= fock[cidx]
                                    elseif term[perm][cidx] == cl
                                        oper4 *= oper[cidx]
                                        fock4 .+= fock[cidx]
                                    else
                                        throw(Exception)
                                    end
                                end

                                clusteredterm = ClusteredTerm4B((oper1,oper2,oper3,oper4), [Tuple(fock1),Tuple(fock2),Tuple(fock3),Tuple(fock4)], (ci, cj, ck, cl), v)
                                focktrans = deepcopy(zero_fock)
                                focktrans[ci.idx] = Tuple(fock1)
                                focktrans[cj.idx] = Tuple(fock2)
                                focktrans[ck.idx] = Tuple(fock3)
                                focktrans[cl.idx] = Tuple(fock4)
                                if haskey(terms,focktrans)
                                    push!(terms[focktrans], clusteredterm)
                                else
                                    terms[focktrans] = [clusteredterm]
                                end
                            end
                        end
                        #=}}}=#
                    end
                end
            end
        end
    end

    unique!(terms)
    #for (fock,termlist) in terms
    #    for op in termlist
    #        display(op)
    #    end
    #end
    return terms
end


"""
    unique!(ClusteredHam::Dict{TransferConfig,Vector{ClusteredTerm}})

combine terms to keep only unique operators
"""
function unique!(clustered_ham::Dict{TransferConfig,Vector{ClusteredTerm}})

    println(" Remove duplicates")
    #
    # first just remove duplicates
    for (ftrans, terms) in clustered_ham

        display(ftrans)
        for term in terms
            for (i,j) in zip(term.ops,term.clusters)
                println(string(i,",",j.idx)
                #display(zip(term.ops,term.clusters)))
            end
        end
    end

    error()
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
#=}}}=#
end


"""
    contract_matrix_element(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
"""
function contract_matrix_element(   term::ClusteredTerm2B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
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
    state_sign = 1
    for (oi,o) in enumerate(term.ops)
        length(o) % 2 != 0 || continue #only count electrons if operator is odd
        n_elec_hopped = 0
        for ci in 1:oi-1
            n_elec_hopped += fock_ket[ci][1] + fock_ket[ci][2]
        end
        if n_elec_hopped % 2 != 0
            state_sign *= -1
        end
    end

    #
    # <I|p'|J> h(pq) <K|q|L>
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    mat_elem = 0.0
    @tensor begin
        mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
    end

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
#=}}}=#
end


"""
    contract_matrix_element(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
"""
function contract_matrix_element(   term::ClusteredTerm3B, 
                                    cluster_ops::Vector{ClusterOps},
                                    fock_bra, bra, fock_ket, ket)
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
    state_sign = 1
    for (oi,o) in enumerate(term.ops)
        length(o) % 2 != 0 || continue #only count electrons if operator is odd
        n_elec_hopped = 0
        for ci in 1:oi-1
            n_elec_hopped += fock_ket[ci][1] + fock_ket[ci][2]
        end
        if n_elec_hopped % 2 != 0
            state_sign *= -1
        end
    end


    #
    # <I|p'|J> h(pq) <K|q|L>
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    mat_elem = 0.0
    @tensor begin
        mat_elem = ((gamma1[p] * term.ints[p,q,r]) * gamma2[q]) * gamma3[r]
    end
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
                                    fock_bra, bra, fock_ket, ket)
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
    state_sign = 1
    for (oi,o) in enumerate(term.ops)
        length(o) % 2 != 0 || continue #only count electrons if operator is odd
        n_elec_hopped = 0
        for ci in 1:oi-1
            n_elec_hopped += fock_ket[ci][1] + fock_ket[ci][2]
        end
        if n_elec_hopped % 2 != 0
            state_sign *= -1
        end
    end


    #
    # <I|p'|J> h(pq) <K|q|L>
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
    mat_elem = 0.0
    @tensor begin
        mat_elem = (((gamma1[p] * term.ints[p,q,r,s]) * gamma2[q]) * gamma3[r]) * gamma4[s]
    end
    return state_sign * mat_elem
end
#=}}}=#


function print_fock_sectors(sector::Vector{Tuple{T,T}}) where T<:Integer
    print("  ")
    for ci in sector
        @printf("(%iα,%iβ)", ci[1],ci[2])
    end
    println()
end
