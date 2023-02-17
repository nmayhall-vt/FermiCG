using QCBase 
using ClusterMeanField
using InCoreIntegrals
using Printf
using JLD2

function run()
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))
    push!(atoms,Atom(5,"H",[0,0,4]))
    push!(atoms,Atom(6,"H",[0,0,5]))
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)


    mf = pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = pyscf_fci(ints,1,1)
    @printf(" FCI Energy: %12.8f\n", e_fci)

    C = mf.mo_coeff
    Cl = localize(mf.mo_coeff,"lowdin",mf)
    pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)


    clusters    = [(1:2),(3:4),(5:6)]
    #clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1)]

    max_roots = 20

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    @save "_testdata_hf_h6.jld2" ints clusters init_fspace e_fci
end


run()
