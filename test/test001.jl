using FermiCG 
using Printf
using NPZ
using JSON 



ket_a = ConfigString(no=4, ne=4)

ints_0b = npzread("../src/python/data/ints_0b.npy")
ints_1b = npzread("../src/python/data/ints_1b.npy")
ints_2b = npzread("../src/python/data/ints_2b.npy")
data = JSON.parsefile("../src/python/data/problem.json")
print(data)
spin = data["spin"]
n_elec = data["n_elec"]
n_elec_a = round(Int,(spin - n_elec)/2)
n_elec_b = n_elec - n_elec_a 

ham 	= ElectronicInts(ints_0b, ints_1b, ints_2b)
problem = ElectronicProblem(no=size(ints_1b,1), na=n_elec_a, nb=n_elec_b)

