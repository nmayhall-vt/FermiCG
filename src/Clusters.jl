
"""
	idx::Integer
	orb_list::Array{Integer,1}
"""
struct Cluster
    idx::Integer
    orb_list::Array{Integer,1}
end

"""
	cluster::Cluster                  # Cluster to which basis belongs
	basis::Dict{Tuple,Vector{Array}}  # Basis vectors (na,nb,ss1,ss2)=>[I,J,pqr]
                                      # Here, ss1, and ss2 denote subspaces in the I and J indices
	ops::Dict{String,Array}           # Operators defined in this basis
"""
struct ClusterBasis
    cluster::Cluster
	basis::Dict{Tuple,Array}
	ops::Dict{String,Array}
end

"""
	length(c::Cluster)

Return number of orbitals in `Cluster`
"""
function Base.length(c::Cluster)
    return length(c.orb_list)
end

"""
	dim_tot(c::Cluster)

Return dimension of hilbert space spanned by number of orbitals in `Cluster`
"""
function dim_tot(c::Cluster)
    return 2^(2*length(c))
end




"""
	clusters::Vector{ClusterBasis}
	ops::Vector{String}
	delta::Vector{Integer}
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
struct ClusteredTerm
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
