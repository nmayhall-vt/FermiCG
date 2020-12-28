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

######################################################################################################

"""
    cluster::Cluster                            # Cluster to which basis belongs
    basis::Dict{Tuple,Matrix{Float64}}          # Basis vectors (nα, nβ)=>[I,s]
These basis coefficients map local slater determinants to local vectors
`(nα, nβ): 
V[αstring*βstring, cluster_state]`
"""
struct ClusterBasis
    cluster::Cluster
    basis::Dict{Tuple,Matrix{Float64}}
end
function ClusterBasis(ci::Cluster)
    return ClusterBasis(ci, Dict{Tuple,Matrix{Float64}}())
end
function Base.getindex(cb::ClusterBasis,i) 
    return cb.basis[i] 
end
function Base.setindex!(cb::ClusterBasis,val,key) 
    cb.basis[key] = val
end
function Base.haskey(cb::ClusterBasis,key) 
    return haskey(cb.basis, key)
end
function Base.display(cb::ClusterBasis) 
    @printf(" ClusterBasis for Cluster: %4i\n",cb.cluster.idx)
    norb = length(cb.cluster)
    for (sector, vecs) in cb.basis
        dim = size(vecs,2)
        total_dim = binomial(norb,sector[1]) * binomial(norb,sector[2]) 
        
        @printf("   FockSector = (%2iα, %2iβ): Total Dim = %5i: Dim = %4i\n", sector[1],sector[2],total_dim, dim)
    end
end


######################################################################################################
struct ClusterOps
    cluster::Cluster
    data::Dict{String,Dict{Tuple,Array}}
end
function Base.getindex(co::ClusterOps,i) 
    return co.data[i] 
end
function Base.setindex!(co::ClusterOps,val,key) 
    co.data[key] = val
end
function Base.haskey(co::ClusterOps,key) 
    return haskey(co.data, key)
end
function Base.iterate(co::ClusterOps) 
    return iterate(co.data) 
end
function Base.display(co::ClusterOps) 
    @printf(" ClusterOps for Cluster: %4i\n",co.cluster.idx)
    norb = length(co.cluster)
    for (op,sectors) in co
        print("   Operator: \n", op)
        print("     Sectors: \n")
        for sector in sectors
            println(sector)
        end
    end
end
function ClusterOps(ci::Cluster)
    dic1 = Dict{Tuple,Array{Float64}}()
    dic2 = Dict{String,typeof(dic1)}() 
    return ClusterOps(ci, dic2)
end

"""
    adjoint(co::ClusterOps; verbose=0)

Take ClusterOps, `co`, and return a new ClusterOps'
"""



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
                basis_i[sec] = F.vectors[:,1:nr]
            else
                e,v = Arpack.eigs(Hmat, nev = nr, which=:SR)
                e = real(e)[1:nr]
                basis_i[sec] = v[:,1:nr]
            end
            if verbose > 0
                state=1
                for ei in e
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
            end
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





"""
    tdm_A(cb::ClusterBasis; verbose=0)

Compute `<s|p'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_A(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+1,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+1)
            else
                throw(DomainError(spin_case))
            end
            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_annihilation(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [1,3,2])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AA(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AA(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+2,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+2)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                #println()
                #println("::::::::::::::: ", fockbra, fockket)
                #println(":: ", size(basis_bra), size(basis_ket))
                dicti[focktrans] = FermiCG.StringCI.compute_AA(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [2,1,4,3])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = (na,nb)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_Aa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
            end
        end
    end
    return dicti
#=}}}=#
end


"""
    tdm_Ab(cb::ClusterBasis; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Ab(cb::ClusterBasis; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -1:norbs+1
        for nb in -1:norbs+1
            fockbra = (na+1,nb-1)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_Ab(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [2,1,4,3])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end
