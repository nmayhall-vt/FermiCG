using TensorOperations

struct ElectronicInts
	#=
	Type to hold a second quantized Hamiltonian coefficients in memory

	h0  is constant energy shift
	h1  is one body operator
	h2  is two body operator
	=#
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

struct Atom
	#=
	Type defining an atom
	=#
	id::Integer
	symbol::String
	xyz::Array{Float64,1}
end

struct Molecule
	#=
	Type defining a molecule
	charge: overall charge on molecule
	multiplicity: multiplicity
	geometry: XYZ coordinates
	=#
	charge::Integer
	multiplicity::Integer
	atoms::Array{Atom,1}
end





#Functions
function orbital_rotation!(ints::ElectronicInts, U)
	#=
	Transform electronic integrals, by U
	i.e., h(pq) = U(rp)h(rs)U(sq)
	=#
	@tensor begin
		ints.h1[p,q] = U[r,p]*U[s,q]*ints.h1[r,s]
		ints.h2[p,q,r,s] = U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
	end
end
function orbital_rotation(ints::ElectronicInts, U)
	#=
	Transform electronic integrals, by U
	i.e., h(pq) = U(rp)h(rs)U(sq)
	=#
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

function subset(ints::ElectronicInts, list)
	h0 = ints.h0
	h1 = view(ints.h1,list,list)
	h2 = view(ints.h1,list,list)
	ints2 = ElectronicInts(ints.h0, view(ints.h1,list,list), view(ints.h2,list,list,list,list))
	#h1 = ints.h1[:,list][list,:]
	#h2 = ints.h2
	return ints2
end

function compute_energy(h0, h1, h2, rdm1, rdm2)
	e = h0
	e += sum(h1 .* rdm1)
	e += .5*sum(h2 .* rdm2)
	# @tensor begin
	# 	e  += .5 * (ints.h2[p,q,r,s] * rdm2[p,q,r,s])
	# end
	return e
end
