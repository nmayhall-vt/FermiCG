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
	cluster::Cluster                  # Cluster to which basis belongs
	basis::Dict{Tuple,Vector{Array}}  # Basis vectors (na,nb,ss1,ss2)=>[I,J,pqr]
                                      # Here, ss1, and ss2 denote subspaces in the I and J indices
"""
struct ClusterBasis
    cluster::Cluster
    basis::Dict{SVector{2,Int16},Array{Float64,2}}
end
function ClusterBasis(ci::Cluster)
    return ClusterBasis(ci, Dict{SVector{2,Int16},Array{Float64,2}}())
end



"""
    config::Vector{SVector{Int16,1}}
"""
struct FockConfig 
    config::Vector{SVector{Int16,1}}
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
    compute_cluster_eigenbasis(ints::ElectronicInts, ci::Cluster, na, nb, verbose=0, max_roots=10)

Build a ClusterBasis for Cluster `ci`
-`ints::ElectronicInts`: In-core integrals
-`ci::Cluster`: Cluster to form basis for
-`na::Int`: Number of alpha electrons
-`nb::Int`: Number of beta electrons
-`verbose::Int`: Print level
-`max_roots::Int`: Maximum number of vectors for current focksector basis
"""
function compute_cluster_eigenbasis(ints::ElectronicInts, ci::Cluster, na, nb; verbose=0, max_roots=10,
    rdm1a=nothing, rdm2a=nothing)

    ints_i = subset(ints,ci.orb_list) 

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
    possible_focksectors(delta_elec=nothing)
        
Get list of possible fock spaces accessible to the cluster

- `delta_elec::Vector{Int}`:   (ref_alpha, ref_beta, delta) allows restrictions to fock spaces
                        based on a delta from some reference occupancy (ref_alpha, ref_beta)
"""
function possible_focksectors(c::Cluster,delta_elec=nothing)
    ref_a = nothing
    ref_b = nothing
    delta = nothing
    if delta_elec != nothing
        size(delta_elec) == 3 || throw(DimensionMismatch)
        ref_a = delta_elec[0]
        ref_b = delta_elec[1]
        delta = delta_elec[2]
    end

    no = length(c)
   
    fsectors::Vector{SVector{2,Int16}} = []
    for na in 1:no
        for nb in 1:no 
            if delta_elec != nothing
                if abs(na-ref_a)+abs(nb-ref_b) > delta
                    continue
                end
            end
            push!(fsectors,[na,nb])
        end
    end
    return fsectors
end
