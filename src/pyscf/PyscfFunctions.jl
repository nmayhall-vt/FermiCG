using PyCall
using PrettyTables

#pydir = joinpath(dirname(pathof(FermiCG)), "python")
#pushfirst!(PyVector(pyimport("sys")."path"), pydir)
#ENV["PYTHON"] = Sys.which("python")
#print(ENV)

"""
	pyscf_do_scf(molecule::Molecule, conv_tol=1e-10)

Use PySCF to compute Hartree-Fock for a given molecule and basis set
and return a PYSCF mean field object
"""
function pyscf_do_scf(molecule::Molecule; conv_tol=1e-10, verbose=0)
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    pymol = make_pyscf_mole(molecule)

    println(pymol.basis)
    #pymol.max_memory = 1000 # MB
    #pymol.symmetry = true
    mf = pyscf.scf.RHF(pymol).run(conv_tol=conv_tol, verbose=verbose)
    enu = mf.energy_nuc()
    println("MO Energies")
    display(mf.mo_energy)
    # print(np.linalg.eig(mf.get_fock())[0])

    # if pymol.symmetry == True:
    # 	from pyscf import symm
    # 	mo = symm.symmetrize_orb(mol, mf.mo_coeff)
    # 	osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    # 	#symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    # 	for i in range(len(osym)):
    # 		print("%4d %8s %16.8f"%(i+1,osym[i],mf.mo_energy[i]))

    return mf
end


"""
	make_pyscf_mole(molecule::Molecule)

Create a `pyscf.gto.Mole()` object
"""
function make_pyscf_mole(molecule::Molecule)
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    pymol = pyscf.gto.Mole()
    pymol.basis = molecule.basis
    geomstr = ""
    for i in molecule.atoms
        geomstr = geomstr * string(i.symbol,", ", join(map(string, i.xyz), ", "),"\n")
    end
    pymol.atom = geomstr
    pymol.charge = molecule.charge
    pymol.spin = molecule.multiplicity-1
    pymol.build()
    return pymol
end

"""
	pyscf_write_molden(molecule::Molecule, C; filename="orbitals.molden")

# Arguments
- `molecule::Molecule`: Molecule object
- `C`: MO Coefficients
- `filename`: Filename to write to

Write MO coeffs `C` to a molden file for visualizing
"""
function pyscf_write_molden(molecule::Molecule, C; filename="orbitals.molden")
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    molden = pyimport("pyscf.tools.molden")
    pymol = make_pyscf_mole(molecule)
    molden.from_mo(pymol, filename, C)
    return 1
end


"""
	pyscf_write_molden(mf; filename="orbitals.molden")

# Arguments
- `mf`: PySCF mean field object
- `filename`: Filename to write to

Write MO coeffs `C` to a molden file for visualizing
"""
function pyscf_write_molden(mf; filename="orbitals.molden")
    pyscf = pyimport("pyscf")
    tools = pyimport("pyscf.tools")
    molden = pyimport("tools.molden")
    molden.from_mo(mf.mol, filename, mf.mo_coeff)
    return 1
end


"""
	pyscf_build_1e(mol::Molecule)

build 1 electron integrals in AO basis 
# Arguments
- `mol::Molecule` 

returns a 2D matrix 
"""
function pyscf_build_1e(mol::Molecule)

    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    # 
    # get pyscf molecule type
    pymol = FermiCG.make_pyscf_mole(mol)
    return pyscf.scf.hf.get_hcore(pymol)
end


"""
	pyscf_build_eri(mol::Molecule, c1::Matrix, c2::Matrix, c3::Matrix, c4::Matrix)

build 2 electron integrals between different orbital spaces, (c1c2|c3c4) 
# Arguments
- `mol::Molecule` 
- `c1`: active space orbital MO coeffs for index 1
- `c2`: active space orbital MO coeffs for index 2
- `c3`: active space orbital MO coeffs for index 3
- `c4`: active space orbital MO coeffs for index 4

returns a 4D tensor 
"""
function pyscf_build_eri(mol::Molecule, c1::Matrix, c2::Matrix, c3::Matrix, c4::Matrix)

    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    # 
    # get pyscf molecule type
    pymol = FermiCG.make_pyscf_mole(mol)
    eri = pymol.intor("int2e")
    h2 = pyscf.ao2mo.incore.general(eri, (c1,c2,c3,c4))

    nm1 = size(c1)[2]
    nm2 = size(c2)[2]
    nm3 = size(c3)[2]
    nm4 = size(c4)[2]
    h2 = reshape(h2, (nm1, nm2, nm3, nm4))

    return h2
end



"""
	pyscf_get_jk(mol, density)

Build exchange matrix in AO basis

# Arguments
- `mol`: PySCF Molecule object
- `density`: 1rdm density matrix for 

"""
function pyscf_get_jk(mol::Molecule, density)
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)

    # 
    # get pyscf molecule type
    pymol = FermiCG.make_pyscf_mole(mol)

    h0 = pyscf.gto.mole.energy_nuc(pymol)
    h  = pyscf.scf.hf.get_hcore(pymol)
    j, k = pyscf.scf.hf.get_jk(pymol, density, hermi=1)
    return h, j, k
end 

"""
pyscf_build_ints(mol, C)

build 1 and 2 electron integrals using a pyscf SCF object
# Arguments
- `mol`: PySCF Molecule object
- `c`: MO coeffs

returns an `InCoreInts` type
"""
function pyscf_build_ints(mol::Molecule, C)

    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)

    norb = size(C)[2]
    # get pyscf molecule type
    pymol = FermiCG.make_pyscf_mole(mol)

    h0 = pyscf.gto.mole.energy_nuc(pymol)
    h1  = pyscf.scf.hf.get_hcore(pymol)

    # now rotate to MO basis
    h1 = C' * h1 * C
    h2 = pyscf.ao2mo.kernel(pymol, C, aosym="s4",compact=false)
    h2 = reshape(h2, (norb, norb, norb, norb))

    return InCoreInts(h0, h1, h2)
end


"""
pyscf_build_ints(mol, c_act, d1_embed)

build 1 and 2 electron integrals using a pyscf SCF object
# Arguments
- `mol`: PySCF Molecule object
- `c_act`: active space orbital MO coeffs
- `d1_embed`: 1rdm density matrix for the frozen part in the AO basis (e.g, doccs or frozen clusters)

returns an `InCoreInts` type
"""
function pyscf_build_ints(mol::Molecule, c_act, d1_embed)

    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)

    nact = size(c_act)[2]
    #mycas = pyscf.mcscf.CASSCF(mf, length(active), 0)
    # 
    # get pyscf molecule type
    pymol = FermiCG.make_pyscf_mole(mol)

    h0 = pyscf.gto.mole.energy_nuc(pymol)
    h  = pyscf.scf.hf.get_hcore(pymol)
    j, k = pyscf.scf.hf.get_jk(pymol, d1_embed, hermi=1)

    # get core energy
    #h0 = tr(d1_embed * ( h + .5*j - .5*k))
    #mf = pyscf.scf.RHF(pymol)
    #h0 = pyscf.scf.hf.energy_elec(mf, d1_embed)[1]
    h0 += tr(d1_embed * ( h + .5*j - .25*k))

    # now rotate to MO basis
    h = c_act' * h * c_act
    j = c_act' * j * c_act;
    k = c_act' * k * c_act;
    h2 = pyscf.ao2mo.kernel(pymol, c_act, aosym="s4",compact=false)
    h2 = reshape(h2, (nact, nact, nact, nact))

    # The use of d1_embed only really makes sense if it has zero electrons in the
    # active space. Let's warn the user if that's not true
    S = pymol.intor("int1e_ovlp_sph")
    n_act = tr(S * d1_embed * S * c_act * c_act')
    if isapprox(abs(n_act),0,atol=1e-8) == false
        println(n_act)
        display(d1_embed)
        error(" I found embedded electrons in the active space?!")
    end

    #println(size(e2))
    #println(h0)

    h1 = h + j - .5*k;
    #display(h + j - .5*k)

    h = InCoreInts(h0, h1, h2);
    return h
end

#"""
#	pyscf_build_ints(mol, c_act)
#
#build 1 and 2 electron integrals using a pyscf SCF object
#active is list of orbital indices which are active
#
#returns an `InCoreInts` type
#"""
#function pyscf_build_ints(mol, c_act)
#
#	pyscf = pyimport("pyscf")
#
#	nact = size(c_act)[2]
#	#mycas = pyscf.mcscf.CASSCF(mf, length(active), 0)
#
#	h0 = pyscf.gto.mole.energy_nuc(mol)
#	h1 = c_act' * pyscf.scf.hf.get_hcore(mol) * c_act
#	h2 = pyscf.ao2mo.kernel(mol, c_act, aosym="s4",compact=false)
#	h2 = reshape(h2, (nact, nact, nact, nact))
#
#	# println(size(c_act))
#	h = InCoreInts(h0, h1, h2);
#	return h
#end


"""
    pyscf_fci(ham, na, nb; 
        max_cycle=40, conv_tol=1e-11, nroots=1, verbose=1, do_rdm1=true, do_rdm2=true)

Use PySCF to compute Full CI
"""
function pyscf_fci(ham, na, nb; 
        max_cycle=40, conv_tol=1e-11, nroots=1, verbose=1, do_rdm1=true, do_rdm2=true)
    # println(" Use PYSCF to compute FCI")
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = max_cycle
    cisolver.conv_tol = conv_tol
    nelec = na + nb
    norb = size(ham.h1)[1]
    efci, ci = cisolver.kernel(ham.h1, ham.h2, norb , (na,nb), ecore=0, nroots =nroots, verbose=verbose)
    #@printf(" Length of CI Vector: %i\n", length(ci[1]))
    #println(size(ci[1]))
    fci_dim = size(ci,1)*size(ci,2)
    # d1 = cisolver.make_rdm1(ci, norb, nelec)

    d1a = Array([])
    d1b = Array([])
    d1  = Array([])
    d2  = Array([])

    if do_rdm1 
        d1a,d1b = cisolver.make_rdm1s(ci, norb, (na,nb))
    end

    if do_rdm2
        d1,d2 = cisolver.make_rdm12(ci, norb, (na,nb))
    end

    #@printf(" Energy2: %12.8f\n", FermiCG.compute_energy(ham.h0, ham.h1, ham.h2, d1a+d1b, d2))
    # print(" PYSCF 1RDM: ")
    if do_rdm1
        F = eigen(d1)
        occs = F.values
        sum_n = sum(occs)
        # @printf(" Sum of diagonals = %12.8f\n", sum_n)
        if verbose > 1
            @printf(" Natural Orbital Occupations:\n")
            [@printf(" %4i %12.8f\n",i,occs[i]) for i in 1:size(occs)[1] ]
            @printf(" -----------------\n")
            @printf(" %4s %12.8f\n\n","sum",sum_n)
        end
        if verbose>1
            pretty_table(d1; formatters = ft_printf("%5.3f"), noheader=true)
        end
    end
    if verbose>0
        if nroots == 1
            @printf(" FCI:        %12.8f %12.8f \n", efci+ham.h0, efci)
        elseif nroots > 1
            for r in 1:nroots
                @printf(" FCI:        %12.8f %12.8f \n", efci[r]+ham.h0, efci[r])
            end
        end
    end

    return efci, d1a,d1b, d2, ci
end



function get_nuclear_rep(mol::Molecule)
    pyscf = pyimport("pyscf")
    pymol = FermiCG.make_pyscf_mole(mol)
    return pyscf.gto.mole.energy_nuc(pymol)
end

"""
	localize(C::Array{Float64,2}, method::String, mol::Molecule)

Localize the orbitals using method = `method` (pm, boys, lowdin)
"""
function localize(C::Array{Float64,2}, method::String, mol::Molecule; verbose=4)
    """
    mf is a pyscf scf object
    """
    pyscf = pyimport("pyscf")
    #pyscf.lib.num_threads(1)
    pyscflo = pyimport("pyscf.lo")
    pymol = make_pyscf_mole(mol)

    if lowercase(method) == "lowdin"
	    ClS = pymol.intor("int1e_ovlp_sph")
        F = svd(Cl)
        # display(Cl - F.U * Diagonal(F.S) * F.Vt)
        # display(Cl - F.vectors * Diagonal(F.values) * F.vectors')
        return F.U * Diagonal(F.S.^(-.5)) * F.Vt
    elseif lowercase(method) == "pm"
        return pyscflo.PM(pymol).kernel(C, verbose=verbose);
    elseif lowercase(method) == "boys"
        return pyscflo.Boys(pymol).kernel(C, verbose=verbose);
        #Cl = pyscflo.Boys(mf.mol, C).kernel(verbose=4)
        #return Cl
    end
end

"""
	localize(C::Array{Float64,2},method::String, mf)

Localize the orbitals using method = `method`
"""
function localize(C::Array{Float64,2}, method::String, mf)
    """
    mf is a pyscf scf object
    """
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    pyscflo = pyimport("pyscf.lo")
    if lowercase(method) == "lowdin"
        Cl = mf.mol.intor("int1e_ovlp_sph")
        F = svd(Cl)
        # display(Cl - F.U * Diagonal(F.S) * F.Vt)
        # display(Cl - F.vectors * Diagonal(F.values) * F.vectors')
        return F.U * Diagonal(F.S.^(-.5)) * F.Vt
    elseif lowercase(method) == "boys"
        Cl = pyscflo.Boys(mf.mol, C).kernel(verbose=4)
        return Cl
    end
end

"""
	get_ovlp(mf)

Get overlap matrix from pyscf using mean-field object
"""
function get_ovlp(mf)
		return mf.mol.intor("int1e_ovlp_sph")
end

