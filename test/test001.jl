using FermiCG
using Printf
using NPZ
using JSON
using Pkg

testdir = joinpath(dirname(pathof(FermiCG)), "..", "test")
println(testdir)

ket_a = ConfigString(no=4, ne=4)

filepath = joinpath(testdir, "data_h4/ints_0b.npy")
ints_0b = npzread(filepath)

filepath = joinpath(testdir, "data_h4/ints_1b.npy")
ints_1b = npzread(filepath)

filepath = joinpath(testdir, "data_h4/ints_2b.npy")
ints_2b = npzread(filepath)

filepath = joinpath(testdir, "data_h4/problem.json")
data = JSON.parsefile(filepath)

println(data)
spin = data["spin"]
n_elec = data["n_elec"]
n_elec_a = round(Int,(spin - n_elec)/2)
n_elec_b = n_elec - n_elec_a

ham 	= ElectronicInts(ints_0b, ints_1b, ints_2b)
#display(rand(Float64, (4,4)) )

using Plots
N = size(ham.h2)[1]
println(size(ham.h2))
heatmap(ham.h1, color = :greys)
gui()
print("what?")

problem = ElectronicProblem(no=size(ints_1b,1), na=n_elec_a, nb=n_elec_b)
mol     = Molecule()
print(mol.geometry)

#get_pyscf_integrals(mol,problem)
pyscf_fci(ham,problem)
