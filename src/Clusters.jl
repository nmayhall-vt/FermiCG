
struct Cluster
    idx::Integer
    orb_list::Array{Integer,1}
end

struct ClusterBasis
    cluster::Cluster
    ops::Dict{String,Array}
end

function length(c::Cluster)
    return length(c.orb_list)
end

function dim_tot(c::Cluster)
    return 2^(2*length(c))
end





struct ClusteredTerm
	"""
	input:
		delta = list of change of Na,Nb,state
				e.g., [(-1,-1),(1,1),(0,0)] means alpha and beta transition
				from cluster 1 to 2, cluster 3 is fock diagonal
		ops   = list of operators
				e.g., ["ab","AB",""]

		ints  = tensor containing the integrals for this block
				e.g., ndarray([p,q,r,s]) where p,q are in 1 and r,s are in 2

		data:
			active: list of clusters which have non-identity operators
				this includes fock-diagonal couplings,
				e.g., ["Aa","","Bb"] would have active = [0,2]
	"""
	clusters::Vector{ClusterBasis}
	ops::Vector{String}
	delta::Vector{Integer}
	ints
end

struct ClusterTerm1B
	c1::Cluster
end

struct ClusterTerm2B
	c1::Cluster
	c2::Cluster
end

struct ClusterTerm3B
	c1::Cluster
	c2::Cluster
	c3::Cluster
end

struct ClusterTerm4B
	c1::Cluster
	c2::Cluster
	c3::Cluster
	c4::Cluster
end

struct ClusterOperator
end
