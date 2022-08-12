using ActiveSpaceSolvers

"""
    cluster::Cluster                            # Cluster to which basis belongs
    basis::Dict{Tuple,Matrix{T}}                # Basis vectors (nα, nβ)=>[I,s]

These basis coefficients map local slater determinants to local vectors
`(nα, nβ): 
V[αstring*βstring, cluster_state]`
"""
struct ClusterBasis{A,T}
    cluster::Cluster
    basis::Dict{Tuple{Int16,Int16},Solution{A,T}}
end
ClusterBasis(ci::Cluster; T::Type = Float64, A=FCIAnsatz) = ClusterBasis(ci, Dict{Tuple{Int16,Int16},Solution{A,T}}())

"""
    ClusterBasis(cb::ClusterBasis, T::Type)

Convert from one data type to another
"""
function ClusterBasis(cb::ClusterBasis{A,TT}, T::Type) where {A,TT}
    out = ClusterBasis{A,T}(cb.cluster, Dict{Tuple{Int16,Int16},Matrix{T}}())
    for (fspace,basis) in cb.basis
        out.basis[fspace] = Solution(basis.ansatz, Vector{T}(basis.energies), Matrix{T}(basis.vectors))
    end
    return out
end

Base.iterate(cb::ClusterBasis, state=1) = iterate(cb.basis, state)
Base.length(cb::ClusterBasis) = length(cb.basis)
Base.getindex(cb::ClusterBasis,i) = cb.basis[i] 
Base.setindex!(cb::ClusterBasis,val,key) = cb.basis[key] = val
Base.haskey(cb::ClusterBasis,key) = haskey(cb.basis, key)
function Base.display(cb::ClusterBasis) 
    @printf(" ClusterBasis for Cluster: %4i\n",cb.cluster.idx)
    norb = length(cb.cluster)
    sum_total_dim = 0
    sum_dim = 0
    
    T = eltype(norb)
    for (sector, vecs) in cb.basis
        na = T(sector[1])
        nb = T(sector[2])
        dim = size(vecs,2)
        total_dim = binomial(norb,na) * binomial(norb,nb) 
        sum_dim += dim
        sum_total_dim += total_dim
        
        @printf("   FockSector = (%2iα, %2iβ): Total Dim = %5i: Dim = %4i\n", sector[1],sector[2],total_dim, dim)
    end
       
    @printf("   -----------------------------\n")
    @printf("   Total Dim = %5i: Dim = %4i\n", sum_total_dim, sum_dim)
end



"""
    rotate!(cb::ClusterBasis, U::Dict{Tuple,Matrix{T}}) where {T} 

Rotate `cb` by unitary matrices in `U`
"""
function rotate!(cb::ClusterBasis,U::Dict{Tuple,Matrix{T}}) where {T} 
#={{{=#
    for (fspace,mat) in U
        cb[fspace].vectors .= cb[fspace] * mat
    end
end
#=}}}=#

    
function check_basis_orthogonality(basis::ClusterBasis; thresh=1e-12)
    for (fspace,mat) in basis
        if check_orthogonality(mat,thresh=thresh) == false
            println(" Cluster:", basis.cluster)
            println(" Fockspace:", fspace)
        end
    end
end

