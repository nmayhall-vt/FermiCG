using PyCall

pydir = joinpath(dirname(pathof(FermiCG)), "python")
pushfirst!(PyVector(pyimport("sys")."path"), pydir)
ENV["PYTHON"] = Sys.which("python")
#print(ENV)

function get_pyscf_integrals(molecule::Molecule, problem)
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
	mol.symmetry = True
	mol.charge = charge
	mol.spin = spin
	mol.basis = basis_set
	mol.build()
end
