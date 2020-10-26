
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


@with_kw struct ElectronicProblem
	#=
	Structure to hold all problem specific parameters
	=#
	no::Int = 0
	na::Int = 0  # number of alpha
	nb::Int = 0  # number of beta
	dima::Int = calc_nchk(no,na)
	dimb::Int = calc_nchk(no,nb)
	dim::Int = dima*dimb
	converged::Bool = false
	restarted::Bool = false
	iteration::Int = 0
	algorithm::String = "direct"    #  options: direct/davidson
	n_roots::Int = 1
end

@with_kw struct Molecule 
    #=
    Type defining a molecule 
	charge: overall charge on molecule
multiplicity: multiplicity
geometry: XYZ coordinates
=#
charge = 0
multiplicity=1
geometry=['H' 0 0 0; 'H' 1 0 0]
end
