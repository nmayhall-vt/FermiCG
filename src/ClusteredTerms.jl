
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
    delta::Tuple{Tuple{Int16,Int16}}
    clusters::Tuple{Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm2B <: ClusteredTerm
    ops::Tuple{String,String}
    delta::Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}}
    #active::Vector{Int16}
    clusters::Tuple{Cluster,Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm3B <: ClusteredTerm
    ops::Vector{String}
    delta::Vector{Tuple{Int16}}
    #active::Vector{Int16}
    clusters::Vector{Cluster}
    ints::Array{Float64}
end

struct ClusteredTerm4B <: ClusteredTerm
    ops::Vector{String}
    delta::Vector{Tuple{Int16}}
    #active::Vector{Int16}
    clusters::Vector{Cluster}
    ints::Array{Float64}
end

function Base.display(t::ClusteredTerm1B)
    @printf( " 1B: %3i    :", t.clusters[1].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm2B)
    @printf( " 2B: %3i %3i:", t.clusters[1].idx, t.clusters[2].idx)
    println(t.ops)
end

"""
    extract_1e_terms(h, clusters)

Extract all ClusteredTerm types from a given 1e integral tensor 
and a list of clusters
"""
function extract_1e_terms(h, clusters)
    norb = 0
    for ci in clusters
        norb += length(ci)
    end
    length(size(h)) == 2 || throw(Exception)
    size(h,1) == norb || throw(Exception)
    size(h,2) == norb || throw(Exception)

    terms = Vector{ClusteredTerm}()
    n_clusters = length(clusters)
    ops_a = Array{String}(undef,n_clusters)
    ops_b = Array{String}(undef,n_clusters)
    fill!(ops_a,"")
    fill!(ops_b,"")
    for ci in clusters
        #
        # p'q where p and q are in ci
        ints = copy(view(h, ci.orb_list, ci.orb_list))

        term = ClusteredTerm1B(("Aa",), ((0,0),), (ci,), ints)
        push!(terms,term)
        term = ClusteredTerm1B(("Bb",), ((0,0),), (ci,), ints)
        push!(terms,term)

        for cj in clusters
            ci != cj || continue
            
            #
            # p'q where p is in ci and q is in cj
            ints = copy(view(h, ci.orb_list, cj.orb_list))
            
            term = ClusteredTerm2B(("A","a"), ((1,0),(-1,0)), (ci, cj), ints)
            push!(terms,term)
            term = ClusteredTerm2B(("B","b"), ((0,1),(0,-1)), (ci, cj), ints)
            push!(terms,term)

        end
    end
    return terms
end
