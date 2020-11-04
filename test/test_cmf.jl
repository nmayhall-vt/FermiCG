using NPZ
using JSON
using Random
using LinearAlgebra
using FermiCG
using Printf


atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[10,0,3]))
push!(atoms,Atom(4,"H",[10,0,4]))
push!(atoms,Atom(5,"H",[20,0,6]))
push!(atoms,Atom(6,"H",[20,0,7]))

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,"6-31g")
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
e_fci, d_fci = FermiCG.pyscf_fci(ints,3,3)
# @printf(" FCI Energy: %12.8f\n", e_fci)

FermiCG.pyscf_write_molden(mol,"6-31g",mf.mo_coeff,filename="scf.molden")

# Cl = FermiCG.localize(mf.mo_coeff,"boys", mf)
# FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="boys.molden")
C = mf.mo_coeff
rdm_mf = C[:,1:2] * C[:,1:2]'
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="lowdin.molden")
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
init_fspace = [(1,1),(1,1),(1,1)]

clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

rdm1 = zeros(size(ints.h1))
rdm1 = rdm_mf
# for ci in clusters
#     flush(stdout)
#     ints_i = FermiCG.subset(ints,ci.orb_list)
#     #display(ints_i)
#     e, d = FermiCG.pyscf_fci(ints_i,init_fspace[ci.idx][1],init_fspace[ci.idx][2])
#     rdm1[ci.orb_list,ci.orb_list] = d
# end
# display(rdm1)
cmf_maxiter = 1
for cmf_iter = 1:cmf_maxiter
    println(" ------------------------------------------ ")
    println(" CMF Iter: ", cmf_iter)
    rdm1_curr = copy(rdm1)
    rdm2 = zeros(size(ints.h2))

    for ci in clusters
        flush(stdout)
        rdm_embed = copy(rdm1_curr)
        rdm_embed[ci.orb_list,:] .= 0
        rdm_embed[:,ci.orb_list] .= 0
        rdm_embed = Cl * rdm_embed * Cl'
        ints_i = FermiCG.pyscf_build_ints(mf.mol,Cl[:,ci.orb_list], rdm_embed);

        # ints_i = FermiCG.subset(ints,ci.orb_list)
        #display(ints_i)
        e, d1, d2 = FermiCG.pyscf_fci(ints_i,init_fspace[ci.idx][1],init_fspace[ci.idx][2])
        rdm1[ci.orb_list,ci.orb_list] = d1
        # rdm2[ci.orb_list, ci.orb_list,ci.orb_list,ci.orb_list] = d2
    end
    e_cmf_curr = FermiCG.compute_energy(ints.h0, ints.h1, ints.h2, rdm1, rdm2)
    @printf(" CMF Curr: %12.8f\n", e_cmf_curr)
end

# display(rdm1)
