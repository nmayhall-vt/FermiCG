using Arpack 
using StaticArrays

"""
    idx::Int
    orb_list::Vector{Int}
"""
struct Cluster
    idx::Int
    orb_list::Vector{Int}
end


"""
    cluster::Cluster                             # Cluster to which basis belongs
    basis::Dict{Tuple,Vector{Array{Float64,2}}}  # Basis vectors (nα, nβ)=>[ss]=>[I,s]
                                                 # Here, ss indicates the subspace of states, s
These basis coefficients map local slater determinants to local vectors
`(nα, nβ)[ss]: 
V[αstring*βstring, cluster_state]`
"""
struct ClusterBasis
    cluster::Cluster
    basis::Dict{Tuple,Vector{Matrix{Float64}}}
end
function ClusterBasis(ci::Cluster)
    return ClusterBasis(ci, Dict{Tuple,Vector{Matrix{Float64}}}([]))
end
function Base.getindex(cb::ClusterBasis,i) 
    return cb.cluster[i] 
end
function Base.display(cb::ClusterBasis) 
    @printf(" ClusterBasis for Cluster: %4i\n",cb.cluster.idx)
    for (sector, subspaces) in cb.basis
        dim = 0
        dims = Integer[]
        for ss in subspaces 
            dim += size(ss,2)
            push!(dims,size(ss,2))
        end
        @printf("   FockSector = (%2iα, %2iβ): Total Dim = %-4i: Dims = ",sector[1],sector[2],dim)
        println.(dims)
    end
end



"""
    config::Vector{Tuple}
"""
struct FockConfig 
    config::Vector{Tuple}
end
function Base.print(f::FockConfig)
    print(f.config)
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
struct ClusteredTerm
	clusters::Vector{ClusterBasis}
	ops::Vector{String}
	delta::Vector{Int}
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


"""
    compute_cluster_eigenbasis(ints::InCoreInts, ci::Cluster, na, nb; 
    verbose=0, max_roots=10, rdm1a=nothing, rdm2a=nothing)

Build a ClusterBasis for Cluster `ci`
- `ints::InCoreInts`: In-core integrals
- `ci::Cluster`: Cluster to form basis for
- `na::Int`: Number of alpha electrons
- `nb::Int`: Number of beta electrons
- `verbose::Int`: Print level
- `max_roots::Int`: Maximum number of vectors for current focksector basis
"""
function compute_cluster_eigenbasis(ints::InCoreInts, ci::Cluster, na, nb; 
                                    verbose=0, max_roots=10, rdm1a=nothing, rdm1b=nothing)

    ints_i = subset(ints,ci.orb_list, rdm1a, rdm1b) 

    problem = FermiCG.StringCI.FCIProblem(length(ci), na, nb)
    display(problem)
    nr = min(max_roots, problem.dim)
    #e, d1, d2 = FermiCG.pyscf_fci(ints_i,na, nb)
    #println(e)
    e = [] 
    v = []
    if verbose > 0 
        @time Hmat = FermiCG.StringCI.build_H_matrix(ints_i, problem)
    else
        Hmat = FermiCG.StringCI.build_H_matrix(ints_i, problem)
    end
    if problem.dim < 1000
        if verbose > 0
            @time F = eigen(Hmat)
            e = F.values
            v = F.vectors
        else
            F = eigen(Hmat)
            e = F.values
            v = F.vectors
        end 
    else
        if verbose > 0
            @time e,v = Arpack.eigs(Hmat, nev = nr, which=:SR)
            e = real(e)
        else
            e,v = Arpack.eigs(Hmat, nev = nr, which=:SR)
            e = real(e)
        end
    end
    state = 1
    for ei in e
        @printf(" State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
        state += 1
    end
    return v
end
    

"""
    possible_focksectors(c::Cluster, delta_elec=nothing)
        
Get list of possible fock spaces accessible to the cluster

- `delta_elec::Vector{Int}`: (nα, nβ, Δ) restricts fock spaces to: nα + nβ ± Δ
"""
function possible_focksectors(c::Cluster; delta_elec::Tuple=())
    ref_a = nothing
    ref_b = nothing
    delta = nothing
    if length(delta_elec) != 0
        length(delta_elec) == 3 || throw(DimensionMismatch)
        ref_a = delta_elec[1]
        ref_b = delta_elec[2]
        delta = delta_elec[3]
    end

    no = length(c)
   
    fsectors::Vector{Tuple} = []
    for na in 0:no
        for nb in 0:no 
            if length(delta_elec) != 0
                if abs(na-ref_a)+abs(nb-ref_b) > delta
                    continue
                end
            end
            push!(fsectors,(na,nb))
        end
    end
    return fsectors
end
