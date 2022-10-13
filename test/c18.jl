using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile 
using Random
using StatProfilerHTML

using PyCall

molecule = "
C     1.259020     0.711710     0.000000
C     2.503660     1.386310     0.000000
C     3.708550     0.701730     0.000000
C     3.708550    -0.701730     0.000000
C     2.503660    -1.386310     0.000000
C     1.259020    -0.711710     0.000000
C    -0.013150    -1.446190     0.000000
C    -0.051250    -2.861390     0.000000
C    -1.246560    -3.562570     0.000000
C    -2.461990    -2.860840     0.000000
C    -2.452410    -1.475080     0.000000
C    -1.245860    -0.734490     0.000000
C    -1.245860     0.734490     0.000000
C    -0.013150     1.446190     0.000000
C    -0.051250     2.861390     0.000000
C    -1.246560     3.562570     0.000000
C    -2.461990     2.860840     0.000000
C    -2.452410     1.475080     0.000000
H     2.529850     2.475260     0.000000
H     4.650090     1.255640     0.000000
H     4.650090    -1.255640     0.000000
H     2.529850    -2.475260     0.000000
H     0.878710    -3.428550     0.000000
H    -1.237620    -4.654920     0.000000
H    -3.412460    -3.399270     0.000000
H    -3.408560    -0.953290     0.000000
H     0.878710     3.428550     0.000000
H    -1.237620     4.654920     0.000000
H    -3.412460     3.399270     0.000000
H    -3.408560     0.953290     0.000000
"

atoms = []
#molecule = lstrip(molecule)
for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
    l = split(line)
    push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
end

cas_nel = 18
cas_norb = 18

clusters    = [(1:6), (7:12), (13:16)]
init_fspace = [(3, 3),(3, 3),(3, 3)]

na = sum([i[1] for i in init_fspace])
nb = sum([i[2] for i in init_fspace])

basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)

tot_na = (18*6 + 12) ÷ 2
tot_nb = (18*6 + 12) ÷ 2
tot_n_elec = tot_na + tot_nb

#np = pyimport("numpy")
#C = np.load("test/c18.orbs.npy")
# get integrals
mf = FermiCG.pyscf_do_scf(mol, verbose=2)
C = mf.mo_coeff
nbas = size(C)[1]
    
FermiCG.pyscf_write_molden(mol, C, filename="hf.molden")

#
# Get active space
act_space = [46, 51, 52, collect(55:68)..., 72]
n_frozen = 51 
#n_frozen = 4*18 + 12
inactive_space = setdiff(1:nbas,act_space)


occ_space = inactive_space[1:n_frozen]
vir_space = inactive_space[n_frozen+1:end]
Cact = C[:,act_space]
Cocc = C[:,occ_space]
Cvir = C[:,vir_space]

ints = FermiCG.pyscf_build_ints(mol, C[:,act_space], 2.0*Cocc*Cocc');


#pyscf = pyimport("pyscf")
#mcscf = pyimport("pyscf.mcscf")
#ao2mo = pyimport("pyscf.ao2mo")
#cas_norb = length(act_space)
#mycas = pyscf.mcscf.CASSCF(mf, cas_norb, 18)
#h1e_cas, ecore = mycas.get_h1eff(mo_coeff = Cocc)  #core core orbs to form ecore and eff
#h2e_cas = ao2mo.kernel(mol, Cact, aosym="s4",compact=false).reshape(4 * ((cas_norb), ))

Cl = FermiCG.localize(Cact, "boys", mf)
# 
# use fiedler vector to reorder orbitals
h,j,k = FermiCG.pyscf_get_jk(mol, C[:,1:tot_na] * C[:,1:tot_na]')
Cl = FermiCG.fiedler_sort(Cl,k)

FermiCG.pyscf_write_molden(mol, Cl, filename="boys.molden")
FermiCG.pyscf_write_molden(mol, Cocc, filename="occ.molden")

n_core = 18
core_space = inactive_space[1:n_core]
sig_space = inactive_space[n_core+1:end]
Csig = FermiCG.localize(C[:,sig_space], "boys", mf)
Csig = FermiCG.fiedler_sort(Csig,(h+j-.5*k))
FermiCG.pyscf_write_molden(mol, Csig, filename="sig.molden")

S = FermiCG.get_ovlp(mf)
U =  Cact' * S * Cl
println(" Rotate Integrals")
flush(stdout)
ints = FermiCG.orbital_rotation(ints,U)
println(" done.")
flush(stdout)




