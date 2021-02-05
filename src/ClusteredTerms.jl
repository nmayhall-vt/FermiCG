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
    cache::Dict
end

struct ClusteredTerm2B <: ClusteredTerm
    ops::Tuple{String,String}
    #delta::Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}}
    delta::TransferConfig
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster}
    ints::Array{Float64}
    cache::Dict
end

struct ClusteredTerm3B <: ClusteredTerm
    ops::Tuple{String,String,String}
    delta::TransferConfig
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster,Cluster}
    ints::Array{Float64}
    cache::Dict
end

struct ClusteredTerm4B <: ClusteredTerm
    ops::Tuple{String,String,String,String}
    delta::TransferConfig
    clusters::Tuple{Cluster,Cluster,Cluster,Cluster}
    ints::Array{Float64}
    cache::Dict
end

#function ClusteredTerm(ops, delta::Vector{Tuple{Int}}, clusters, ints)
#end

function Base.display(t::ClusteredTerm1B)
    @printf( " 1B: %2i          :", t.clusters[1].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm2B)
    @printf( " 2B: %2i %2i       :", t.clusters[1].idx, t.clusters[2].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm3B)
    @printf( " 2B: %2i %2i %2i    :", t.clusters[1].idx, t.clusters[2].idx, t.clusters[3].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm4B)
    @printf( " 2B: %2i %2i %2i %2i :", t.clusters[1].idx, t.clusters[2].idx, t.clusters[3].idx, t.clusters[4].idx)
    println(t.ops)
end


function bubble_sort(inp)
    #={{{=#
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
#=}}}=#
end


"""
    extract_terms(ints::InCoreInts, clusters)

Extract all ClusteredTerm types from a given 1e integral tensor 
and a list of clusters
returns `terms::Dict{TransferConfig,Vector{ClusteredTerm}}`
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
   
    # 1-body terms
    if true 
        for ci in clusters
#={{{=#
            # instead of forming p'q and p'q'sr just precontract and keep them in 
            # ClusterOps
            term = ClusteredTerm1B(("H",), ((0,0),), (ci,), zeros(1,1),Dict())
            push!(terms[zero_fock],term)
#=}}}=#
        end
    end

    # 2-body 1-electron terms
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

                        clusteredterm = ClusteredTerm2B((oper1,oper2), [Tuple(fock1),Tuple(fock2)], (ci, cj), h, Dict())
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
    
    # 2-body 2-electron terms
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
                append!(termstr,unique(permutations([ci,cj,cj,cj])))
                append!(termstr,unique(permutations([ci,ci,ci,cj])))

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
                        vcurr = deepcopy(v)
                       
                        if true 
                            if oper1 == "BA"
                                oper1 = "AB"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "ba"
                                oper1 = "ab"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "BAa"
                                oper1 = "ABa"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "BAb"
                                oper1 = "ABb"
                                vcurr = -permutedims(vcurr,[2,1,3,4])
                            elseif oper1 == "Bab"
                                oper1 = "Bba"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            elseif oper1 == "Aab"
                                oper1 = "Aba"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            end
                            

                            if oper2 == "BA"
                                oper2 = "AB"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            elseif oper2 == "ba"
                                oper2 = "ab"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            elseif oper2 == "BAa"
                                oper2 = "ABa"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            elseif oper2 == "BAb"
                                oper2 = "ABb"
                                vcurr = -permutedims(vcurr,[1,3,2,4])
                            elseif oper2 == "Bab"
                                oper2 = "Bba"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            elseif oper2 == "Aab"
                                oper2 = "Aba"
                                vcurr = -permutedims(vcurr,[1,2,4,3])
                            end
                        end
                       
                        vcurr = copy(reshape(vcurr,newshape...))

                        clusteredterm = ClusteredTerm2B((oper1,oper2), [Tuple(fock1),Tuple(fock2)], (ci, cj), vcurr, Dict())
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

    # 3-body 2-electron terms
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
                            vcurr = deepcopy(v)

                            if true 
                                if oper1 == "BA"
                                    oper1 = "AB"
                                    vcurr = -permutedims(vcurr,[2,1,3,4])
                                elseif oper1 == "ba"
                                    oper1 = "ab"
                                    vcurr = -permutedims(vcurr,[2,1,3,4])
                                end


                                if oper2 == "BA"
                                    oper2 = "AB"
                                    vcurr = -permutedims(vcurr,[1,3,2,4])
                                elseif oper2 == "ba"
                                    oper2 = "ab"
                                    vcurr = -permutedims(vcurr,[1,3,2,4])
                                end


                                if oper3 == "BA"
                                    oper3 = "AB"
                                    vcurr = -permutedims(vcurr,[1,2,4,3])
                                elseif oper3 == "ba"
                                    oper3 = "ab"
                                    vcurr = -permutedims(vcurr,[1,2,4,3])
                                end
                            end

                            vcurr = copy(reshape(vcurr,newshape...))

                            #core,factors = tucker_decompose(vcurr)

                            clusteredterm = ClusteredTerm3B((oper1,oper2,oper3), [Tuple(fock1),Tuple(fock2),Tuple(fock3)], (ci, cj, ck), vcurr, Dict())
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

    # 4-body 2-electron terms
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

                                clusteredterm = ClusteredTerm4B((oper1,oper2,oper3,oper4), [Tuple(fock1),Tuple(fock2),Tuple(fock3),Tuple(fock4)], (ci, cj, ck, cl), v, Dict())
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
    
    return terms
end


"""
    unique!(ClusteredHam::Dict{TransferConfig,Vector{ClusteredTerm}})

combine terms to keep only unique operators
"""
function unique!(clustered_ham::Dict{TransferConfig,Vector{ClusteredTerm}})
#={{{=#
    println(" Remove duplicates")
    #
    # first just remove duplicates
    nstart = 0
    nfinal = 0
#    for (ftrans, terms) in clustered_ham
#        tmp = deepcopy(terms)
#        for term in terms
#            println(term.ops)
##            swap = false
##            idx = 0
##            for (opidx,op) in enumerate(term.ops)
##                if op == "BA"
##                    swap = true
##                    idx = opidx
##                end
##            end
##            if swap
##                println(size(term.ints))
##                v = term.ints 
##                v = copy(reshape(v, (size(v,1), size(v,2), size(v,3), size(v,4))))
##                v = -permutedims(v,[2,1,3,4])
##                v = reshape(v, (size(v,1)*size(v,2)*size(v,3), size(v,4)))
##
##                newterm = ClusteredTerm2B(("ABb","a"), term.delta, term.clusters, term.ints)
##                push!(tmp,newterm)
##            end
#
#            if term.ops == ("BAb","a")
#               
#                println(size(term.ints))
#                v = term.ints 
#                v = copy(reshape(v, (size(v,1), size(v,2), size(v,3), size(v,4))))
#                v = -permutedims(v,[2,1,3,4])
#                v = reshape(v, (size(v,1)*size(v,2)*size(v,3), size(v,4)))
#
#                newterm = ClusteredTerm2B(("ABb","a"), term.delta, term.clusters, term.ints)
#                push!(tmp,newterm)
#            else
#                push!(tmp,term)
#            end
#        end
#
#        clustered_ham[ftrans] = tmp
#    end
        
#        tmp = deepcopy(terms)
#        for term in tmp
#            for op in term.ops
#                if op == "BAb"
#
#                    fconfig = 
#                    newterm = ClusteredTerm2B((oper1,oper2,oper3,oper4), ftrans, term.clusters, v)



    for (ftrans, terms) in clustered_ham
        unique = Dict()
        for term in terms
            nstart += 1
            keystr = ""
            for (i,j) in zip(term.ops,term.clusters)
                keystr *= string(i,"(",j.idx,")")
            end
            if haskey(unique,keystr)
                unique[keystr].ints .+= term.ints
            else
                unique[keystr] = deepcopy(term)
            end
        end
        clustered_ham[ftrans] = Vector{ClusteredTerm}()
        for (keystr, term) in unique
            push!(clustered_ham[ftrans],term)
        end
        nfinal += length(clustered_ham[ftrans]) 
    end

    @printf(" Number of terms reduced from %5i to %5i\n", nstart, nfinal)
#=}}}=#
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
#=}}}=#
end


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
    # <I|p'|J> h(pq) <K|q|L>
#    display(term)
#    display(fock_bra)
#    display(fock_ket)
    gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    mat_elem = 0.0
    #@tensor begin
    #    mat_elem = gamma1[p] * term.ints[p,q] * gamma2[q]
    #end
    mat_elem = _contract(term.ints, gamma1, gamma2)

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



function _contract(ints,gamma1,gamma2)
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
function _contract(ints,gamma1,gamma2,gamma3)
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
function _contract(ints,gamma1,gamma2,gamma3,gamma4)
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
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    #mat_elem = 0.0
    #@tensor begin
    #    mat_elem = ((gamma1[p] * term.ints[p,q,r]) * gamma2[q]) * gamma3[r]
    #end
    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3)
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
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
    #mat_elem = 0.0
    #@tensor begin
    #    mat_elem = (((gamma1[p] * term.ints[p,q,r,s]) * gamma2[q]) * gamma3[r]) * gamma4[s]
    #end
    mat_elem = _contract(term.ints, gamma1, gamma2, gamma3, gamma4)
    return state_sign * mat_elem
end
#=}}}=#


function compute_terms_state_sign(term::ClusteredTerm, fock_ket::FockConfig)
    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = 1
    for (oi,o) in enumerate(term.ops)
        if length(o) % 2 != 0  #only count electrons if operator is odd
            n_elec_hopped = 0
            for ci in 1:term.clusters[oi].idx-1
                n_elec_hopped += fock_ket[ci][1] + fock_ket[ci][2]
            end
            if n_elec_hopped % 2 != 0
                state_sign = -state_sign
            end
        end
    end
    return state_sign
end


function print_fock_sectors(sector::Vector{Tuple{T,T}}) where T<:Integer
    print("  ")
    for ci in sector
        @printf("(%iα,%iβ)", ci[1],ci[2])
    end
    println()
end
