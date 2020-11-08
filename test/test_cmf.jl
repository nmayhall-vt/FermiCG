using NPZ
using JSON
using Random
using LinearAlgebra
using FermiCG
using Printf


atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[0,0,3]))
push!(atoms,Atom(5,"H",[0,0,4]))
push!(atoms,Atom(6,"H",[0,0,5]))
basis = "sto-3g"

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,basis)
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,3,3)
# @printf(" FCI Energy: %12.8f\n", e_fci)

function compute_npairs(d2)
    npairs = 0
    for i in 1:size(d2)[1]
        for j in 1:size(d2)[1]
            npairs += d2[i,i,j,j]
        end
    end
    println(" NPairs:", round(npairs,digits=3))
end

compute_npairs(d2_fci)

FermiCG.pyscf_write_molden(mol,basis,mf.mo_coeff,filename="scf.molden")

# Cl = FermiCG.localize(mf.mo_coeff,"boys", mf)
# FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="boys.molden")
C = mf.mo_coeff
rdm_mf = C[:,1:2] * C[:,1:2]'
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,basis,Cl,filename="lowdin.molden")
S = FermiCG.get_ovlp(mf)
U =  C' * S * Cl
println(" Build Integrals")
flush(stdout)
ints = FermiCG.orbital_rotation(ints,U)
println(" done.")
flush(stdout)
# ints = FermiCG.pyscf_build_ints(mf.mol,Cl);
# println(" done.")
# flush(stdout)
clusters    = [(1:4),(5:8),(9:12)]
clusters    = [(1:2),(3:4),(5:6)]
init_fspace = [(1,1),(1,1),(1,1)]

clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

rdm1 = zeros(size(ints.h1))
rdm1a = rdm_mf*.5
rdm1b = rdm_mf*.5

#FermiCG.cmf_ci(ints, clusters, init_fspace, rdm1)
U = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, verbose=0, gconv=1e-6)
#U = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, verbose=0, gconv=1e-8, method="cg")

C_cmf = Cl*U

FermiCG.pyscf_write_molden(mol,basis,C_cmf,filename="cmf.molden")

