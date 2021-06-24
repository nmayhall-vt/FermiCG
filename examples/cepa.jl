using LinearAlgebra
using FermiCG
using Printf
using Test


molecule = "
H   0.0     0.0     0.0
H   0.0     0.0     1.0
H   0.0     1.0     2.0
H   0.0     1.0     3.0
H   0.0     2.0     4.0
H   0.0     2.0     5.0
H   0.0     3.0     6.0
H   0.0     3.0     7.0
H   0.0     4.0     8.0
H   0.0     4.0     9.0
H   0.0     5.0     10.0
H   0.0     5.0     11.0
"

atoms = []
for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
l = split(line)
push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
end


clusters_in    = [(1:4), (5:8), (9:12)]
init_fspace = [(2, 2),(2, 2),(2, 2)]

(na,nb) = sum(init_fspace)


basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)

# get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
#e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=1);
e_fci = -18.33022092 + ints.h0
display(e_fci)


# localize orbitals
C = mf.mo_coeff
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
S = FermiCG.get_ovlp(mf)
U =  C' * S * Cl
println(" Rotate Integrals")
flush(stdout)
ints = FermiCG.orbital_rotation(ints,U)
println(" done.")
flush(stdout)

# define clusters
clusters = [Cluster(i,collect(clusters_in[i])) for i = 1:length(clusters_in)]
display(clusters)


rdm1 = zeros(size(ints.h1))
rdm1a = rdm1
rdm1b = rdm1


e_cmf, U, Da, Db  = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1a,rdm1b,
                                        max_iter_oo=20, verbose=0, gconv=1e-7, method="bfgs");
ints = FermiCG.orbital_rotation(ints,U)


max_roots = 400
# Build Cluster basis
cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots,
        init_fspace=init_fspace, rdm1a=Da, rdm1b=Db);

#
# Build ClusteredOperator
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters);

# Build Cluster Operators
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

p_spaces = Vector{FermiCG.ClusterSubspace}()
q_spaces = Vector{FermiCG.ClusterSubspace}()

# define p spaces
for ci in clusters
    tss = FermiCG.ClusterSubspace(ci)
    tss[init_fspace[ci.idx]] = 1:1
    push!(p_spaces, tss)
end

# define q spaces
for tssp in p_spaces
    tss = FermiCG.get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
    push!(q_spaces, tss)
end

println(" ================= Cluster P Spaces ===================")
display.(p_spaces);
println(" ================= Cluster Q Spaces ===================")
display.(q_spaces);

nroots = 1
ref_vector = FermiCG.TuckerState(clusters, p_spaces, q_spaces, na,nb)
#
# initialize with eye
FermiCG.set_vector!(ref_vector, Matrix(1.0I, length(ref_vector),nroots))

ref  = FermiCG.CompressedTuckerState(ref_vector, thresh=-1);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, Da, Db);

for i in 2:20
    thresh =  0.1^i
    #e_cepa, v_cepa = FermiCG.do_fois_cepa(ref, cluster_ops, clustered_ham, thresh_foi=thresh, max_iter=50, nbody=2);
    #e_pt2, v_pt2 =    FermiCG.do_fois_pt2(ref, cluster_ops, clustered_ham, thresh_foi=1e-5, max_iter=50, nbody=2, tol=1e-8);
    e_ci, v_ci =       FermiCG.do_fois_ci(ref, cluster_ops, clustered_ham, thresh_foi=thresh, max_iter=50, nbody=2, tol=1e-8);
    @printf(" CI  : %12.6f \n",(e_ci+ints.h0))
    #@printf(" PT2 : %12.6f \n",(e_pt2+ints.h0))
    #@printf("CEPA : %12.6f \n",(e_cepa+ints.h0))
    println(thresh)
end

#e_pt2, v_pt2 =    FermiCG.do_fois_pt2(ref, cluster_ops, clustered_ham, thresh_foi=1e-5, max_iter=50, nbody=2, tol=1e-8);
#e_ci, v_ci =       FermiCG.do_fois_ci(ref, cluster_ops, clustered_ham, thresh_foi=1e-5, max_iter=50, nbody=2, tol=1e-8);

#@printf(" cMF : %12.6f \n",(e_cmf))
#@printf(" PT2 : %12.6f \n",(e_pt2+ints.h0))
#@printf(" CI  : %12.6f \n",(e_ci+ints.h0))
display(e_fci)
