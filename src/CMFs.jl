using LinearAlgebra
using Random
using Optim

"""
	form_casci_ints(ints::ElectronicInts, ci::Cluster, rdm1a, rdm1b)

Obtain a subset of integrals which act on the orbitals in Cluster,
embedding the 1rdm from the rest of the system

Returns an ElectronicInts type
"""
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
	return ints_i
end

"""
	compute_cmf_energy(ints, rdm1s, rdm2s, clusters)

Compute the energy of a cluster-wise product state (CMF),
specified by a list of 1 and 2 particle rdms local to each cluster

ints: ElectronicInts object for full system
rdm1s: dictionary (ci.idx => Array) of 1rdms from each cluster (spin summed)
rdm2s: dictionary (ci.idx => Array) of 2rdms from each cluster (spin summed)
"""
function compute_cmf_energy(ints, rdm1s, rdm2s, clusters; verbose=1)
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
    if verbose>=1
        for ei = 1:length(e1)
            @printf(" Cluster %3i E =%12.8f\n",ei,e1[ei])
        end
    end
	return ints.h0 + sum(e1) + sum(e2)
	# display(e2)
	# pretty_table(e1; formatters = ft_printf("%5.3f"))

end

"""
    cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace; verbose=1)

Perform single CMF-CI iteration, returning new energy, and density
"""
function cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace; verbose=1)
	# rdm1s = fill(Array{Real,2}, length(clusters), 1)
	# rdm2s = fill(Array{Real,4}, length(clusters), 1)
	# rdm2s = Array{Array{Real,4},1}([],length(clusters),1)
	# rdm2s = Array{Array{Real,4},1}
	rdm1_dict = Dict{Integer,Array}()
	rdm2_dict = Dict{Integer,Array}()
	for ci in clusters
		flush(stdout)
		ints_i = form_casci_ints(ints, ci, rdm1a, rdm1b)
		e, d1, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
		rdm1_dict[ci.idx] = d1
		rdm2_dict[ci.idx] = d2
	end
	e_curr = compute_cmf_energy(ints, rdm1_dict, rdm2_dict, clusters, verbose=verbose)
    if verbose > 1
        @printf(" CMF-CI Curr: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
    end

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
	return e_curr,rdm1a_out, rdm1b_out, rdm1_dict, rdm2_dict
end


"""
    cmf_ci(ints, clusters, fspace, dguess; max_iter=10, dconv=1e-6, econv=1e-10, verbose=1)

Optimize the 1RDM for CMF-CI
"""
function cmf_ci(ints, clusters, fspace, dguess; max_iter=10, dconv=1e-6, econv=1e-10, verbose=1)
	rdm1a = deepcopy(dguess)
	rdm1b = deepcopy(dguess)
	energies = []
	e_prev = 0

	rdm1_dict = 0
	rdm2_dict = 0
	rdm1_dict = Dict{Integer,Array}()
	rdm2_dict = Dict{Integer,Array}()
	# rdm2_dict = Dict{Integer, Array}()
    for iter = 1:max_iter
        if verbose > 0
            println()
            println(" ------------------------------------------ ")
            println(" CMF CI Iter: ", iter)
            println(" ------------------------------------------ ")
        end
        e_curr, rdm1a_curr, rdm1b_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace, verbose=verbose)
        append!(energies,e_curr)
        error = (rdm1a_curr+rdm1b_curr) - (rdm1a+rdm1b)
        d_err = norm(error)
        e_err = e_curr-e_prev
        if verbose>0
            @printf(" CMF-CI Energy: %12.8f | Change: RDM: %6.1e Energy %6.1e\n\n", e_curr, d_err, e_err)
        end
		e_prev = e_curr*1
		rdm1a = rdm1a_curr
		rdm1b = rdm1b_curr
		if (abs(d_err) < dconv) && (abs(e_err) < econv)
			if verbose>0
                @printf("*CMF-CI: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
            end
			break
		end
	end
    if verbose>0
        println(" Energy per Iteration:")
        for i in energies
            @printf(" Elec: %12.8f Total: %12.8f\n", i-ints.h0, i)
        end
    end
	return e_prev, rdm1a, rdm1b, rdm1_dict, rdm2_dict
end




"""
    cmf_oo(ints::ElectronicInts, clusters::Vector{Cluster}, fspace, dguess; 
            max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs")

Do CMF with orbital optimization
"""
function cmf_oo(ints::ElectronicInts, clusters::Vector{Cluster}, fspace, dguess; 
            max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs")
    norb = size(ints.h1)[1]
    #kappa = zeros(norb*(norb-1))
    # e, da, db = cmf_oo_iteration(ints, clusters, fspace, max_iter_ci, dguess, kappa)

    function g(k)
        stepsize = 1e-5
        grad = zeros(size(k))
        for (ii,i) in enumerate(k)
            k1 = deepcopy(k)
            k1[ii] += stepsize
            e1 = cmf_oo_iteration(ints, clusters, fspace, max_iter_ci, dguess, k1)[1]
            k2 = deepcopy(k)
            k2[ii] -= stepsize
            e2 = cmf_oo_iteration(ints, clusters, fspace, max_iter_ci, dguess, k2)[1]
            grad[ii] = (e1-e2)/(2*stepsize)
        end
        return grad
    end

    #   
    #   Initialize optimization data
    #
    e_curr = 0
    g_curr = 0
    e_err = 0
    da = zeros(size(ints.h1))
    db = zeros(size(ints.h1))
    iter = 0
    kappa = zeros(norb*(norb-1))

    #
    #   Define Objective function (energy)
    #
    function f(k)
        K = unpack_gradient(k, norb)
        U = exp(K)
        ints2 = orbital_rotation(ints,U)
        e, da1, db1, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da+db, dconv=gconv/10.0, verbose=0)
        e_err = e-e_curr
        e_curr = e
        return e
    end

    #   
    #   Define Callback for logging and checking for convergence
    #
    function callback(k)
        iter += 1
        if (g_curr < gconv) 
            @printf("*ooCMF Iter: %4i Elec= %16.12f Total= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return true 
        else
            @printf(" ooCMF Iter: %4i Elec= %16.12f Total= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return false 
        end
    end

    #
    #   Define Gradient function
    #
	function g_analytic(kappa)
		norb = size(ints.h1)[1]
		# println(" In g_analytic")
		K = unpack_gradient(kappa, norb)
		U = exp(K)
		ints2 = orbital_rotation(ints,U)

		e, gd1a, gd1b, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da+db, dconv=gconv/10.0, verbose=verbose)
		grad = zeros(size(ints2.h1))
		for ci in clusters
			grad_1 = grad[:,ci.orb_list]
			h_1	   = ints2.h1[:,ci.orb_list]
			v_111  = ints2.h2[:, ci.orb_list, ci.orb_list, ci.orb_list]
			@tensor begin
				grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,u,w,v]
				grad_1[p,q] += h_1[p,r] * rdm1_dict[ci.idx][r,q]
			end
			for cj in clusters
				if ci.idx == cj.idx
					continue
				end
				v_212 = ints2.h2[:,cj.orb_list, ci.orb_list, cj.orb_list]
				v_122 = ints2.h2[:,ci.orb_list, cj.orb_list, cj.orb_list]
				d1 = rdm1_dict[ci.idx]
				d2 = rdm1_dict[cj.idx]

				@tensor begin
					grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
					grad_1[p,q] -= .5*v_212[p,v,u,w] * d1[q,u] * d2[w,v]
				end
			end
			grad[:,ci.orb_list] .= -2*grad_1
		end
		grad = grad'-grad
		gout = pack_gradient(grad, norb)
        g_curr = norm(gout)
		return gout
	end

#	grad1 = g(kappa)
#	grad2 = g_analytic(kappa)
#	display(round.(grad1,digits=6))
#	display(round.(grad2,digits=6))
#   return

   
    optmethod = BFGS()
    if method=="bfgs"
        optmethod = BFGS()
    elseif method=="cg"
        optmethod = ConjugateGradient()
    end
   
    options = Optim.Options(
        callback = callback, 
        g_tol=gconv,
    )
    
    res = optimize(f, g_analytic, kappa, optmethod, options; inplace = false )
    summary(res)
    e = Optim.minimum(res)
    display(res)
    @printf(" ooCMF %12.8f ", e - ints.h0)
    
    kappa = Optim.minimizer(res)
    K = unpack_gradient(kappa, norb)
	U = exp(K)

    return e, U
end




function unpack_gradient(kappa,norb)
	# n = round(.5+sqrt(1+4k)/2)
	# println(n)
	K = zeros(norb,norb)
	ind = 1
	for i in 1:norb
		for j in i+1:norb
			K[i,j] = kappa[ind]
			K[j,i] = -kappa[ind]
			ind += 1
		end
	end
	return K
end
function pack_gradient(K,norb)
	kout = zeros(norb*(norb-1))
	ind = 1
	for i in 1:norb
		for j in i+1:norb
			kout[ind] = K[i,j]
			ind += 1
		end
	end
	return kout
end
