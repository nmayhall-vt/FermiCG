using Arpack 
using StaticArrays
using LinearAlgebra

using NPZ

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

Return dimension of hilbert space spanned by number of orbitals in `Cluster`. 
This is all sectors
"""
function dim_tot(c::Cluster)
    return 2^(2*length(c))
end
"""
	dim_tot(c::Cluster, na, nb)

Return dimension of hilbert space spanned by number of orbitals in `Cluster`
with `na` and `nb` number of alpha/beta electrons.
"""
function dim_tot(c::Cluster, na, nb)
    nc = length(c)
    T = eltype(nc)
    return binomial(nc, T(na))*binomial(nc, T(nb)) 
end
"""
	display(c::Cluster)
"""
function Base.display(c::Cluster)
    @printf("IDX%03i:DIM%04i:" ,c.idx,dim_tot(c))
    for si in c.orb_list
        @printf("%03i|", si)
    end
    @printf("\n")
end
function Base.isless(ci::Cluster, cj::Cluster)
    return Base.isless(ci.idx, cj.idx)
end
function Base.isequal(ci::Cluster, cj::Cluster)
    return Base.isequal(ci.idx, cj.idx)
end
######################################################################################################


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





function check_orthogonality(mat; thresh=1e-12)
    Id = mat' * mat
    if maximum(abs.(I-Id)) > thresh 
        @warn("problem with orthogonality ", maximum(abs.(I-Id)))
        return false
    end
    return true
end

