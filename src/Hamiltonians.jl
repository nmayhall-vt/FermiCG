using TensorOperations

"""
	h0::Real                # constant energy shift
	h1::Array{Float64,2}    # one electron integrals
	h2::Array{Float64,4}    # two electron integrals (chemist's notation)

Type to hold a second quantized Hamiltonian coefficients in memory
"""
struct ElectronicInts
	h0::Real
	h1::Array{Float64,2}
	h2::Array{Float64,4}
end


# @with_kw struct ElectronicProblem
# 	#=
# 	Structure to hold all problem specific parameters
# 	=#
# 	no::Int = 0
# 	na::Int = 0  # number of alpha
# 	nb::Int = 0  # number of beta
# 	basis::String = "sto-3g"
# 	dima::Int = calc_nchk(no,na)
# 	dimb::Int = calc_nchk(no,nb)
# 	#dim::Int = dima*dimb
# 	#converged::Bool = false
# 	#restarted::Bool = false
# 	#iteration::Int = 0
# 	#algorithm::String = "direct"    #  options: direct/davidson
# 	#n_roots::Int = 1
# end

"""
	id::Integer             # index of atom in the molecule
	symbol::String          # Atomic ID (E.g. H, He, ...)
	xyz::Array{Float64,1}   # list of XYZ coordinates
"""
struct Atom
	#=
	Type defining an atom
	=#
	id::Integer
	symbol::String
	xyz::Array{Float64,1}
end

"""
	charge::Integer          # overall charge on molecule
	multiplicity::Integer    # 2S+1
	atoms::Array{Atom,1}     # Vector of `Atoms`
"""
struct Molecule
	charge::Integer
	multiplicity::Integer
	atoms::Array{Atom,1}
end




"""
	orbital_rotation!(ints::ElectronicInts, U)

Transform electronic integrals, by U
i.e.,
```math
h_{pq} = U_{rp}h_{rs}U_{sq}
```
```math
(pq|rs) = (tu|vw)U_{tp}U_{uq}U_{vr}U_{ws}
```
"""
function orbital_rotation!(ints::ElectronicInts, U)
	@tensor begin
		ints.h1[p,q] = U[r,p]*U[s,q]*ints.h1[r,s]
		ints.h2[p,q,r,s] = U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
	end
end

@doc raw"""
	orbital_rotation(ints::ElectronicInts, U)

Transform electronic integrals, by U
i.e.,
```math
h_{pq} = U_{rp}h_{rs}U_{sq}
```
```math
(pq|rs) = (tu|vw)U_{tp}U_{uq}U_{vr}U_{ws}
```
"""
function orbital_rotation(ints::ElectronicInts, U)
	@tensor begin
		h1[p,q] := U[r,p]*U[s,q]*ints.h1[r,s]
		# h2[p,q,r,s] := U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
		h2[p,q,r,s] := U[t,p]*ints.h2[t,q,r,s]
		h2[p,q,r,s] := U[t,q]*h2[p,t,r,s]
		h2[p,q,r,s] := U[t,r]*h2[p,q,t,s]
		h2[p,q,r,s] := U[t,s]*h2[p,q,r,t]
	end
	return ElectronicInts(ints.h0,h1,h2)
end


"""
	subset(ints::ElectronicInts, list)

Extract a subset of integrals acting on orbitals in list, returned as ElectronicInts type
"""
function subset(ints::ElectronicInts, list)
	# h0 = ints.h0
	# h1 = view(ints.h1,list,list)
	# h2 = view(ints.h1,list,list)
	# ints2 = ElectronicInts(ints.h0, ints.h1[list,list], ints.h2[list,list,list,list])
	ints2 = ElectronicInts(ints.h0, view(ints.h1,list,list), view(ints.h2,list,list,list,list))
	#h1 = ints.h1[:,list][list,:]
	#h2 = ints.h2
	return ints2
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
