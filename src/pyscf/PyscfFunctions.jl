using PyCall
using PrettyTables

pydir = joinpath(dirname(pathof(FermiCG)), "python")
pushfirst!(PyVector(pyimport("sys")."path"), pydir)
ENV["PYTHON"] = Sys.which("python")
#print(ENV)

function pyscf_do_scf(molecule::Molecule, basis::String; conv_tol=1e-10)
	pyscf = pyimport("pyscf")
	pymol = make_pyscf_mole(molecule, basis)

	println(pymol.basis)
	#pymol.max_memory = 1000 # MB
	#pymol.symmetry = true
	mf = pyscf.scf.RHF(pymol).run(conv_tol=conv_tol)
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

function make_pyscf_mole(molecule::Molecule, basis::String)
	pyscf = pyimport("pyscf")
	pymol = pyscf.gto.Mole()
	pymol.basis = basis
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

function pyscf_write_molden(molecule, basis, C; filename="orbitals.molden")
	pyscf = pyimport("pyscf")
	molden = pyimport("pyscf.molden")
	pymol = make_pyscf_mole(molecule, basis)
	molden.from_mo(pymol, filename, C)
	return 1
end

function pyscf_write_molden(mf; filename="orbitals.molden")
	pyscf = pyimport("pyscf")
	molden = pyimport("pyscf.molden")
	molden.from_mo(mf.mol, filename, mf.mo_coeff)
	return 1
end


function pyscf_build_ints(mol, c_act, d1_embed)
	"""
	build 1 and 2 electron integrals using a pyscf SCF object
	active is list of orbital indices which are active
	d1_embed is a density matrix for the frozen part (e.g, doccs or frozen clusters)

	returns an ElectronicInts type
	"""
	pyscf = pyimport("pyscf")

	nact = size(c_act)[2]
	#mycas = pyscf.mcscf.CASSCF(mf, length(active), 0)

	h0 = pyscf.gto.mole.energy_nuc(mol)
	h = c_act' * pyscf.scf.hf.get_hcore(mol) * c_act
	j, k = pyscf.scf.hf.get_jk(mol, d1_embed, hermi=1)
	j = c_act' * j * c_act;
	k = c_act' * k * c_act;
	h2 = pyscf.ao2mo.kernel(mol, c_act, aosym="s4",compact=false)
	h2 = reshape(h2, (nact, nact, nact, nact))

	# The use of d1_embed only really makes sense if it has zero electrons in the
	# active space. Let's warn the user if that's not true
	n_act = tr(c_act' * d1_embed * c_act)
	if isapprox(abs(n_act),0,atol=1e-8) == false
		println(n_act)
		error(" I found embedded electrons in the active space?!")
	end

	#println(size(e2))
	#println(h0)

	h1 = h + j - .5*k;
	#display(h + j - .5*k)

	h = ElectronicInts(h0, h1, h2);
	return h
end


function pyscf_build_ints(mol, c_act)
	"""
	build 1 and 2 electron integrals using a pyscf SCF object
	active is list of orbital indices which are active

	returns an ElectronicInts type
	"""
	pyscf = pyimport("pyscf")

	nact = size(c_act)[2]
	#mycas = pyscf.mcscf.CASSCF(mf, length(active), 0)

	h0 = pyscf.gto.mole.energy_nuc(mol)
	h1 = c_act' * pyscf.scf.hf.get_hcore(mol) * c_act
	h2 = pyscf.ao2mo.kernel(mol, c_act, aosym="s4",compact=false)
	h2 = reshape(h2, (nact, nact, nact, nact))

	println(size(c_act))
	h = ElectronicInts(h0, h1, h2);
	return h
end


function pyscf_fci(ham, na, nb; max_cycle=20, conv_tol=1e-8, nroots=1)
	println(" Use PYSCF to compute FCI")
	pyscf = pyimport("pyscf")
	fci = pyimport("pyscf.fci")
	cisolver = pyscf.fci.direct_spin1.FCI()
	cisolver.max_cycle = max_cycle
	cisolver.conv_tol = conv_tol
	nelec = na + nb
	norb = size(ham.h1)[1]
	efci, ci = cisolver.kernel(ham.h1, ham.h2, norb , nelec, ecore=ham.h0, nroots =nroots, verbose=100)
	fci_dim = size(ci)[1]*size(ci)[2]
	d1 = cisolver.make_rdm1(ci, norb, nelec)
	print(" PYSCF 1RDM: ")
	F = eigen(d1)
	occs = F.values
	sum_n = sum(occs)
	@printf(" Sum of diagonals = %12.8f\n", sum_n)
	[@printf("%4i %12.8f\n",i,occs[i]) for i in 1:size(occs)[1] ]

	pretty_table(d1; formatters = ft_printf("%5.3f"), noheader=true)
	@printf(" FCI:        %12.8f Dim:%6d\n", efci,fci_dim)
	#for i in range(0,nroots):
	#    print("FCI %10.8f"%(efci[i]))
	#exit()
	#fci_dim =1

	return efci, d1, fci_dim
end


function localize(C::Array{Float64,2},method::String, mf)
	"""
	mf is a pyscf scf object
	"""
	pyscf = pyimport("pyscf")
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

function get_ovlp(mf)
		return mf.mol.intor("int1e_ovlp_sph")
end
