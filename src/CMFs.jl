
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
		ints_i.h1[p,q] += .5*viirs[p,q,r,s] * (da+db)[r,s]
		# fb = deepcopy(fa)
		# fa[p,s] -= viqri[p,q,r,s] * da[q,r]
		# fb[p,s] -= viqri[p,q,r,s] * db[q,r]
		ints_i.h1[p,s] -= .25*viqri[p,q,r,s] * da[q,r]
		ints_i.h1[p,s] -= .25*viqri[p,q,r,s] * da[q,r]

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
	rdm1s: dictionary of 1rdms from each cluster (spin summed)
	rdm2s: dictionary of 2rdms from each cluster (spin summed)
	"""
	e1 = zeros((length(clusters),1))
	e2 = zeros((length(clusters),length(clusters)))
	for ci in clusters
		ints_i = subset(ints, ci.orb_list)
		# ints_i = ints
		# display(rdm1s)
		# h_pq   = ints.h1[ci.orb_list, ci.orb_list]
		#
		# v_pqrs = ints.h2[ci.orb_list,ci.orb_list,ci.orb_list,ci.orb_list]
		# # println(ints_i.h2 - v_pqrs)
		# # return
		# tmp = 0
		# @tensor begin
		# 	tmp += h_pq[p,q] * rdm1s[ci.idx][q,p]
		# 	tmp += .5 * v_pqrs[p,q,r,s] * rdm2s[ci.idx][p,q,r,s]
		# end
		e1[ci.idx] = FermiCG.compute_energy(0, ints_i.h1, ints_i.h2, rdm1s[ci.idx], rdm2s[ci.idx])
		# e1[ci.idx] = tmp
	end
	for ci in clusters
		for cj in clusters
			if ci.idx >= cj.idx
				continue
			end
			v_pqrs = ints.h2[ci.orb_list, ci.orb_list, cj.orb_list, cj.orb_list]
			v_psrq = ints.h2[ci.orb_list, cj.orb_list, cj.orb_list, ci.orb_list]
			# v_pqrs = ints.h2[ci.orb_list, ci.orb_list, cj.orb_list, cj.orb_list]
			tmp = 0
			@tensor begin
				tmp  = v_pqrs[p,q,r,s] * rdm1s[ci.idx][p,q] * rdm1s[cj.idx][r,s]
				tmp -= .5*v_psrq[p,s,r,q] * rdm1s[ci.idx][p,q] * rdm1s[cj.idx][r,s]
			end
			e2[ci.idx, cj.idx] = tmp
		end
	end
	for ei = 1:length(e1)
		@printf(" Cluster %3i E =%12.8f\n",ei,e1[ei])
	end
	return ints.h0 + sum(e1) + sum(e2)
	# display(e2)
	# pretty_table(e1; formatters = ft_printf("%5.3f"))

end

function cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace)
	# rdm1s = fill(Array{Real,2}, length(clusters), 1)
	# rdm2s = fill(Array{Real,4}, length(clusters), 1)
	# rdm2s = Array{Array{Real,4},1}([],length(clusters),1)
	# rdm2s = Array{Array{Real,4},1}
	rdm1_dict = Dict{Integer,Array}()
	rdm2_dict = Dict{Integer,Array}()
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
		rdm1_dict[ci.idx] = d1
		rdm2_dict[ci.idx] = d2
		# display(ints.h1)
	end
	# println(rdm1_dict[clusters[1].idx])
	# return
	e_curr = compute_cmf_energy(ints, rdm1_dict, rdm2_dict, clusters)
	@printf(" CMF Curr: Electronic %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)

	rdm1a_out = zeros(size(rdm1a))
	rdm1b_out = zeros(size(rdm1b))
	for ci in clusters
		# for (iidx,i) in enumerate(ci.orb_list)
		# 	for (jidx,j) in enumerate(ci.orb_list)
		# 		rdm1a_out[i,j] = rdm1_dict[ci.idx][iidx,jidx]
		# 		rdm1b_out[i,j] = rdm1_dict[ci.idx][iidx,jidx]
		# 	end
		# end
		rdm1a_out[ci.orb_list, ci.orb_list] .= rdm1_dict[ci.idx]
		rdm1b_out[ci.orb_list, ci.orb_list] .= rdm1_dict[ci.idx]
	end
	return e_curr,rdm1a_out, rdm1b_out
end

function cmf_ci(ints, clusters, fspace, dguess, max_iter=10)
	rdm1a = deepcopy(dguess)
	rdm1b = deepcopy(dguess)
	energies = []
	for iter = 1:max_iter
		println()
	    println(" ------------------------------------------ ")
	    println(" CMF CI Iter: ", iter)
		println(" ------------------------------------------ ")
	    e_curr, rdm1a, rdm1b = cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace)
		append!(energies,e_curr)
	end
	println(" Energy per Iteration:")
	for i in energies
		@printf(" Elec: %12.8f Total: %12.8f\n", i-ints.h0, i)
	end
end
