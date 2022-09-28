using Combinatorics
using InteractiveUtils


"""
    ops::Tuple{String}
    delta::TransferConfig{1}
    parity::Tuple{Int}
    clusters::Tuple{Cluster}
    ints::Array{T,1}
    cache::Dict


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
- parity: does each operator have even or odd number of second quantized operators 
"""
abstract type ClusteredTerm{T} end

struct ClusteredTerm1B{T} <: ClusteredTerm{T}
    ops::Tuple{String}
    delta::TransferConfig{1}
    parity::Tuple{Int}
    clusters::Tuple{MOCluster}
    ints::Array{T,1}
    cache::Dict
end

struct ClusteredTerm2B{T} <: ClusteredTerm{T}
    ops::Tuple{String,String}
    delta::TransferConfig{2}
    parity::Tuple{Int,Int}
    #active::Vector{Int16}
    clusters::Tuple{MOCluster,MOCluster}
    ints::Array{T,2}
    cache::Dict
end

struct ClusteredTerm3B{T} <: ClusteredTerm{T}
    ops::Tuple{String,String,String}
    delta::TransferConfig{3}
    parity::Tuple{Int,Int,Int}
    #active::Vector{Int16}
    clusters::Tuple{MOCluster,MOCluster,MOCluster}
    ints::Array{T,3}
    cache::Dict
end

struct ClusteredTerm4B{T} <: ClusteredTerm{T}
    ops::Tuple{String,String,String,String}
    delta::TransferConfig{4}
    parity::Tuple{Int,Int,Int,Int}
    clusters::Tuple{MOCluster,MOCluster,MOCluster,MOCluster}
    ints::Array{T,4}
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
    @printf( " 3B: %2i %2i %2i    :", t.clusters[1].idx, t.clusters[2].idx, t.clusters[3].idx)
    println(t.ops)
end
function Base.display(t::ClusteredTerm4B)
    @printf( " 4B: %2i %2i %2i %2i :", t.clusters[1].idx, t.clusters[2].idx, t.clusters[3].idx, t.clusters[4].idx)
    println(t.ops)
end

   


function check_term(term::ClusteredTerm1B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    return true
end

function check_term(term::ClusteredTerm2B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    #
    #
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    fock_bra[term.clusters[2].idx] == fock_ket[term.clusters[2].idx] .+ term.delta[2] || return false 
    return true
end

function check_term(term::ClusteredTerm3B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue
        ci != term.clusters[3].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    fock_bra[term.clusters[2].idx] == fock_ket[term.clusters[2].idx] .+ term.delta[2] || return false 
    fock_bra[term.clusters[3].idx] == fock_ket[term.clusters[3].idx] .+ term.delta[3] || return false 
    return true
end

function check_term(term::ClusteredTerm4B, 
                            fock_bra::FockConfig, bra::T, 
                            fock_ket::FockConfig, ket::T) where T<:Union{ClusterConfig, TuckerConfig}
    length(fock_bra) == length(fock_ket) || throw(Exception)
    length(bra) == length(ket) || throw(Exception)
    n_clusters = length(bra)
    # 
    # make sure inactive clusters are diagonal
    for ci in 1:n_clusters
        ci != term.clusters[1].idx || continue
        ci != term.clusters[2].idx || continue
        ci != term.clusters[3].idx || continue
        ci != term.clusters[4].idx || continue

        fock_bra[ci] == fock_ket[ci] || return false 
        bra[ci] == ket[ci] || return false
    end
    fock_bra[term.clusters[1].idx] == fock_ket[term.clusters[1].idx] .+ term.delta[1] || return false 
    fock_bra[term.clusters[2].idx] == fock_ket[term.clusters[2].idx] .+ term.delta[2] || return false 
    fock_bra[term.clusters[3].idx] == fock_ket[term.clusters[3].idx] .+ term.delta[3] || return false 
    fock_bra[term.clusters[4].idx] == fock_ket[term.clusters[4].idx] .+ term.delta[4] || return false 
    return true
end




function compute_terms_state_sign(term::ClusteredTerm, fock_ket::FockConfig)
    # 
    # determine sign from rearranging clusters if odd number of operators
    state_sign = 1
    for (oi,o) in enumerate(term.ops)
        if term.parity[oi] == 1  #only count electrons if operator is odd
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

