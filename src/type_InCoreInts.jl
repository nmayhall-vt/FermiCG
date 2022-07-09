"""
h0::Real                # constant energy shift
h1::Array{T,2}          # one electron integrals
h2::Array{T,4}          # two electron integrals (chemist's notation)

Type to hold a second quantized Hamiltonian coefficients in memory
"""
struct InCoreInts{T}
    h0::T
    h1::Array{T,2}
    h2::Array{T,4}
end

function InCoreInts(ints::InCoreInts{TT}, T::Type) where {TT}
    return InCoreInts{T}(T(ints.h0), Array{T,2}(ints.h1), Array{T,4}(ints.h2))
end


"""
orbital_rotation!(ints::InCoreInts, U)

Transform electronic integrals, by U
i.e.,
```math
h_{pq} = U_{rp}h_{rs}U_{sq}
```
```math
(pq|rs) = (tu|vw)U_{tp}U_{uq}U_{vr}U_{ws}
```
"""
function orbital_rotation!(ints::InCoreInts, U)
    @tensor begin
        ints.h1[p,q] = U[r,p]*U[s,q]*ints.h1[r,s]
        ints.h2[p,q,r,s] = U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
    end
end

@doc raw"""
orbital_rotation(ints::InCoreInts, U)

Transform electronic integrals, by U
i.e.,
```math
h_{pq} = U_{rp}h_{rs}U_{sq}
```
```math
(pq|rs) = (tu|vw)U_{tp}U_{uq}U_{vr}U_{ws}
```
"""
function orbital_rotation(ints::InCoreInts, U)
    @tensor begin
        h1[p,q] := U[r,p]*U[s,q]*ints.h1[r,s]
        # h2[p,q,r,s] := U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
        h2[p,q,r,s] := U[t,p]*ints.h2[t,q,r,s]
        h2[p,q,r,s] := U[t,q]*h2[p,t,r,s]
        h2[p,q,r,s] := U[t,r]*h2[p,q,t,s]
        h2[p,q,r,s] := U[t,s]*h2[p,q,r,t]
    end
    return InCoreInts(ints.h0,h1,h2)
end


"""
subset(ints::InCoreInts, list; rmd1a=nothing, rdm1b=nothing)

Extract a subset of integrals acting on orbitals in list, returned as `InCoreInts` type
- `ints::InCoreInts`: Integrals for full system 
- `list`: list of orbital indices in subset
- `rdm1a`: 1RDM for embedding α density to make CASCI hamiltonian
- `rdm1b`: 1RDM for embedding β density to make CASCI hamiltonian
"""
function subset(ints::InCoreInts{T}, list, rdm1a=nothing, rdm1b=nothing) where {T}
    ints_i = InCoreInts{T}(ints.h0, view(ints.h1,list,list), view(ints.h2,list,list,list,list))
    if rdm1b != nothing 
        if rdm1a == nothing
            throw(Exception)
        end
    end

    if rdm1a != nothing
        if rdm1b == nothing
            throw(Exception)
        end
        da = deepcopy(rdm1a)
        db = deepcopy(rdm1b)
        da[:,list] .= 0
        db[:,list] .= 0
        da[list,:] .= 0
        db[list,:] .= 0
        viirs = ints.h2[list, list,:,:]
        viqri = ints.h2[list, :, :, list]
        fa = zeros(length(list),length(list))
        fb = copy(fa)
        @tensor begin
            ints_i.h1[p,q] += viirs[p,q,r,s] * (da+db)[r,s]
            ints_i.h1[p,s] -= .5*viqri[p,q,r,s] * da[q,r]
            ints_i.h1[p,s] -= .5*viqri[p,q,r,s] * db[q,r]
        end
    end
    return ints_i
end



"""
compute_energy(h0, h1, h2, rdm1, rdm2)

Given an energy shift `h0`, 1e integrals `h1`, and 2e ints `h2`
along with a 1rdm and 2rdm on the same space, return the energy
"""
function compute_energy(h0, h1, h2, rdm1, rdm2)
    e = h0
    e += sum(h1 .* rdm1)
    e += .5*sum(h2 .* rdm2)
    # @tensor begin
    # 	e  += .5 * (ints.h2[p,q,r,s] * rdm2[p,q,r,s])
    # end
    return e
end
"""
compute_energy(ints::InCoreInts, rdm1, rdm2)

Return energy defined by `rdm1` and `rdm2`
"""
function compute_energy(ints::InCoreInts, rdm1, rdm2)
    e = ints.h0
    e += sum(ints.h1 .* rdm1)
    e += .5*sum(ints.h2 .* rdm2)
    # @tensor begin
    # 	e  += .5 * (ints.h2[p,q,r,s] * rdm2[p,q,r,s])
    # end
    return e
end

