using FermiCG
using Printf
using NPZ
using JSON
using Pkg



testdir = joinpath(dirname(pathof(FermiCG)), "..", "test")
println(testdir)

ket_a = ConfigString(no=4, ne=4)

filepath = joinpath(testdir, "data_h4/ints_0b.npy")
ints_0b = npzread(filepath);

filepath = joinpath(testdir, "data_h4/ints_1b.npy")
ints_1b = npzread(filepath);

filepath = joinpath(testdir, "data_h4/ints_2b.npy")
ints_2b = npzread(filepath);

filepath = joinpath(testdir, "data_h4/problem.json")
data = JSON.parsefile(filepath)

println(data)
spin = data["spin"]
n_elec = data["n_elec"]
n_elec_a = round(Int,(spin + n_elec)/2)
n_elec_b = n_elec - n_elec_a

ham 	= ElectronicInts(ints_0b, ints_1b, ints_2b)
#display(rand(Float64, (4,4)) )


#using Plots
#N = size(ham.h2)[1]
#println(size(ham.h2))
#heatmap(ham.h1, color = :greys)
#gui()
#print("what?")

# problem = ElectronicProblem(no=size(ints_1b,1), na=n_elec_a, nb=n_elec_b)
# display(problem)

atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[1,0,0]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[1,0,2]))

mol     = Molecule(0,1,atoms)
display(mol)

mf = FermiCG.pyscf_do_scf(mol,"sto-3g")
print(typeof(mf))
FermiCG.pyscf_write_molden(mol,"sto-3g",mf.mo_coeff)
FermiCG.pyscf_write_molden(mf,filename="2.molden")
FermiCG.pyscf_fci(ham,2,2)
