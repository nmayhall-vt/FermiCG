using PyCall
using PrettyTables
pydir = joinpath(dirname(pathof(FermiCG)), "python")
pushfirst!(PyVector(pyimport("sys")."path"), pydir)
ENV["PYTHON"] = Sys.which("python")
#print(ENV)

function get_pyscf_integrals(molecule::Molecule, problem)
	print("NYI")
	exit()
	math = pyimport("math")
	math.sin(math.pi / 4)

	pyscf = pyimport("pyscf")
	print(" Here I am","\n")

	mol = pyscf.gto.Mole()
	geomstr = ""
	for i in molecule.geometry
		println(i)
		geomstr = geomstr * string(i)
	end
	print(geomstr)

	print("Geometry done.")
	mol.atom = geomstr

	mol.max_memory = 1000 # MB
	mol.symmetry = true
	mol.charge = charge
	mol.spin = spin
	mol.basis = basis_set
	mol.build()
end

function pyscf_fci(ham, problem; max_cycle=20, conv_tol=1e-8, nroots=1)
	println(" Use PYSCF to compute FCI")
	pyscf = pyimport("pyscf")
	fci = pyimport("pyscf.fci")
	cisolver = pyscf.fci.direct_spin1.FCI()
	cisolver.max_cycle = max_cycle
	cisolver.conv_tol = conv_tol
	nelec = problem.na + problem.nb
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
