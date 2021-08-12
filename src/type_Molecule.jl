"""
    charge::Integer         #overall charge on molecule
    multiplicity::Integer   #2S+1
    atoms::Vector{Atom}     #Vector of `Atoms`
    basis::String           #Basis set
Molecule essentially as a Vector of atoms, number of electrons and basis set
"""
struct Molecule
	charge::Integer
	multiplicity::Integer
	atoms::Array{Atom,1}
    basis::String
end




