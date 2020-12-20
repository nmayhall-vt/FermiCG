using FermiCG
using Printf
using NPZ
using JSON
using Pkg
using LinearAlgebra


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

ham 	= InCoreInts(ints_0b, ints_1b, ints_2b)
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
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[0,0,3]))

mol     = Molecule(0,1,atoms)
display(mol)

mf = FermiCG.pyscf_do_scf(mol,"6-31g")

FermiCG.pyscf_write_molden(mol,"sto-3g",mf.mo_coeff)
FermiCG.pyscf_write_molden(mf,filename="2.molden")
FermiCG.pyscf_fci(ham,2,2)

n_orbs = size(mf.mo_coeff)[2]
orb_indices = [i for i = 1:n_orbs]
emb_indices = [i for i = 1:2]
display(emb_indices)
emb_orbs = mf.mo_coeff[:,emb_indices];
emb_density = 2*emb_orbs * emb_orbs';
display(emb_density)
PS = emb_density*mf.get_ovlp();
n = tr(PS)

#ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff[:,[3,4]], emb_density);
#FermiCG.pyscf_fci(ints,0,0)

ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
FermiCG.pyscf_fci(ints,2,2)
