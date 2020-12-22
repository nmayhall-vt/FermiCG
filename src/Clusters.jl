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
    #energies::Dict{Tuple,Vector{Vector{Float64}}}
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
    compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{Cluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing)

Return a Vector of `ClusterBasis` for each `Cluster` 
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{Cluster}`: Clusters 
- `verbose::Int`: Print level
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `delta_elec`: number of electrons different from reference (init_fspace)
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1a`: background density matrix for embedding local hamiltonian (alpha)
- `rdm1b`: background density matrix for embedding local hamiltonian (beta)
"""
function compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{Cluster}; 
                init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
                rdm1a=nothing, rdm1b=nothing)

    # initialize output
    cluster_bases = Vector{ClusterBasis}()

    for ci in clusters
        verbose == 0 || display(ci)

        #
        # Get subset of integrals living on cluster, ci
        ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 

        if all( (rdm1a,rdm1b,init_fspace) .!= nothing)
            # 
            # Verify that density matrix provided is consistent with reference fock sectors
            occs = diag(rdm1a)
            occs[ci.orb_list] .= 0
            na_embed = sum(occs)
            occs = diag(rdm1b)
            occs[ci.orb_list] .= 0
            nb_embed = sum(occs)
            verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f", na_embed, nb_embed)
        end
            
        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec)
        end
        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = FermiCG.possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci) 
        for sec in sectors
            
            #
            # Initialize basis for sector as list of matrices, 1 for each subspace
            ee = Vector{Vector{Float64}}() # energies
            vv = Vector{Matrix{Float64}}() # eigenvalues
            e = []
            v = []
            #
            # prepare for FCI calculation for give sector of Fock space
            problem = FermiCG.StringCI.FCIProblem(length(ci), sec[1], sec[2])
            verbose == 0 || display(problem)
            nr = min(max_roots, problem.dim)

            #
            # Build full Hamiltonian matrix in cluster's Slater Det basis
            Hmat = FermiCG.StringCI.build_H_matrix(ints_i, problem)
            if problem.dim < 1000
                F = eigen(Hmat)
                e = F.values[1:nr]
                v = F.vectors[:,1:nr]
            else
                e,v = Arpack.eigs(Hmat, nev = nr, which=:SR)
                e = real(e)[1:nr]
                v = v[:,1:nr]
            end
            push!(ee, e)
            push!(vv, v)
            if verbose > 0
                state=1
                for ei in e
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
            end

            basis_i.basis[sec] = vv 
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end


    

"""
    possible_focksectors(c::Cluster, delta_elec=nothing)
        
Get list of possible fock spaces accessible to the cluster

- `delta_elec::Vector{Int}`: (nα, nβ, Δ) restricts fock spaces to: (nα,nβ) ± Δ electron transitions
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
