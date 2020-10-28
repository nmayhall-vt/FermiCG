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
		h2[p,q,r,s] := U[t,p]*U[u,q]*U[v,r]*U[w,s]*ints.h2[t,u,v,w]
	end
	return ElectronicInts(ints.h0,h1,h2)
end
