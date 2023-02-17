using QCBase
using RDM
using ClusterMeanField
using Printf
using Test
using JLD2

function generate_h8_data()
    atoms = []
    push!(atoms,Atom(1,"H",[0, 0, 0.1]))
    push!(atoms,Atom(2,"H",[0, 1,-1]))
    push!(atoms,Atom(3,"H",[0, 1, 1]))
    push!(atoms,Atom(4,"H",[0, 2, 0]))
    push!(atoms,Atom(5,"H",[0, 4, 0]))
    push!(atoms,Atom(6,"H",[0, 5,-1]))
    push!(atoms,Atom(7,"H",[0, 5, 1]))
    push!(atoms,Atom(8,"H",[0, 6, 0]))
    #basis = "6-31g"
    basis = "sto-3g"

    na = 4
    nb = 4

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

    FermiCG.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    clusters    = [(1:4),(5:8)]
    init_fspace = [(2,2),(2,2)]

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf*.5
    rdm1b = rdm_mf*.5

    rdm1a = rdm1
    rdm1b = rdm1

    display(rdm1a)
    display(rdm1b)
    

    d1 = RDM1(n_orb(ints))
    
    na = 4
    nb = 4

    e_cmf, U, d1  = cmf_oo(ints, clusters, init_fspace, d1,
                                       max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    ints = FermiCG.orbital_rotation(ints,U)

    @save "_testdata_cmf_h8.jld2" ints d1 clusters init_fspace e_fci
end

generate_h8_data()

