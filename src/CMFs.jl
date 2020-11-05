
function form_casci_ints(ints::ElectronicInts, ci::Cluster, rdm1a, rdm1b)
	da = deepcopy(rdm1a)
	db = deepcopy(rdm1b)
	da[:,ci.orb_list] .= 0
	db[:,ci.orb_list] .= 0
	da[ci.orb_list,:] .= 0
	db[ci.orb_list,:] .= 0
	viirs = ints.h2[ci.orb_list, ci.orb_list,:,:]
	viqri = ints.h2[ci.orb_list, :, :, ci.orb_list]
	fa = zeros(length(ci),length(ci))
	fb = copy(fa)
	ints_i = subset(ints, ci.orb_list)
	@tensor begin
		ints_i.h1[p,q] += viirs[p,q,r,s] * (da+db)[r,s]
		# fb = deepcopy(fa)
		# fa[p,s] -= viqri[p,q,r,s] * da[q,r]
		# fb[p,s] -= viqri[p,q,r,s] * db[q,r]
		ints_i.h1[p,s] -= .5*viqri[p,q,r,s] * da[q,r]
		ints_i.h1[p,s] -= .5*viqri[p,q,r,s] * da[q,r]

	end
	# println(size(fa))
	# println(size(fb))
	# ints_i.h1 += .5*(fa + fb)
	return ints_i
end

function compute_cmf_energy(ints, rdm1s, rdm2s, clusters)
	"""
	Compute the energy of a cluster-wise product state (CMF),
	specified by a list of 1 and 2 particle rdms local to each cluster

	ints: ElectronicInts object for full system
	rdm1s: list of 1rdms from each cluster (spin summed)
	rdm2s: list of 2rdms from each cluster (spin summed)
	"""
	e1 = zeros((length(clusters),1))
	e2 = zeros((length(clusters),length(clusters)))
	for ci in clusters
		ints_i = subset(ints, ci.orb_list)
		# ints_i = ints
		e1[ci.idx] = FermiCG.compute_energy(ints_i.h0, ints_i.h1, ints_i.h2, rdm1s[ci.idx], rdm2s[ci.idx])
	end
	for ei = 1:length(e1)
		@printf(" Cluster %3i E =%12.8f\n",ei,e1[ei])
	end
	# pretty_table(e1; formatters = ft_printf("%5.3f"))

end

function cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace)
	rdm1s = []
	rdm2s = []
	for ci in clusters
		flush(stdout)
		ints_i = form_casci_ints(ints, ci, rdm1a, rdm1b)
		# ints_i = FermiCG.pyscf_build_ints(mf.mol,Cl[:,ci.orb_list], rdm_embed);
		# ints_i = FermiCG.subset(ints,ci.orb_list)
		#display(ints_i)
		e, d1, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2])
		# e1 = FermiCG.compute_energy(ints_i.h0, ints_i.h1, ints_i.h2, d1, d2)
		# println(e1)
		# println(d1)
		append!(rdm1s, d1)
		append!(rdm2s, d2)
	end
	compute_cmf_energy(ints, rdm1s, rdm2s, clusters)
end
