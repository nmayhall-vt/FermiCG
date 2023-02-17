using QCBase
using RDM
using ClusterMeanField
using Printf
using Test
using JLD2

function generate_h9_data()
    
    atoms = []
    push!(atoms,Atom(1,"H", [1, 0, 0]))
    push!(atoms,Atom(2,"H", [2, 0, 0]))
    push!(atoms,Atom(3,"H", [3, 0, 0]))
    push!(atoms,Atom(4,"H", [1, 2, 0]))
    push!(atoms,Atom(5,"H", [2, 2, 0]))
    push!(atoms,Atom(6,"H", [3, 2, 0]))
    push!(atoms,Atom(7,"H", [1, 4, 0]))
    push!(atoms,Atom(8,"H", [2, 4, 0]))
    push!(atoms,Atom(9,"H", [3, 4, 0]))


    clusters    = [(1:3),(4:6),(7:9)]
    init_fspace = [(2,1),(1,2),(2,1)]
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    na = 5
    nb = 4
    
    basis = "sto-3g"
    mol     = Molecule(0,2,atoms,basis)

    nroots = 3 

    # get integrals
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
	
    @printf(" Do FCI\n")
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 200 
    cisolver.conv_tol = 1e-8
    nelec = na + nb
    norb = size(ints.h1,1)
    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)
    for e in e_fci
        @printf(" FCI Energy: %12.8f\n", e)
    end
    
    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    #FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin_h3_3.molden")
    flush(stdout)

    # Do CMF
    e_cmf, U, d1 = ClusterMeanField.cmf_oo(  ints, clusters, init_fspace, RDM1(n_orb(ints)),
                            verbose=0, gconv=1e-6, method="bfgs",sequential=true)
    ints = orbital_rotation(ints, U)

    @save "_testdata_cmf_h9.jld2" ints d1 clusters init_fspace e_fci
end

generate_h9_data()
