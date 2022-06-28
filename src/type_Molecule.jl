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



#function write_xyz(mol::Molecule; file="mol", append=true)
#    xyz = ""
#    for a in mol.atoms
#        xyz = xyz * a.id * "," * a.xyz[1]  * "," * a.xyz[2]  * "," * a.xyz[3] * "\n"
#    end
#    open(file*".txt", "w") do file
#        write(file, xyz)
#    end
#end
