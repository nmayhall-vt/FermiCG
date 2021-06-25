using LinearAlgebra
using Random
using Optim


"""
	compute_cmf_energy(mol::Molecule, C::Matrix, rdm1s, rdm2s, clusters)

Compute the energy of a cluster-wise product state (CMF),
specified by a list of 1 and 2 particle rdms local to each cluster

#Arguments
- `mol::Molecule`
- `C::Matrix`: MO coefficients
- `rdm1s`: dictionary (`ci.idx` => Array) of 1rdms from each cluster (spin summed)
- `rdm2s`: dictionary (`ci.idx` => Array) of 2rdms from each cluster (spin summed)
- `clusters::Vector{Cluster}`: vector of cluster objects

return the total CMF energy
"""
function compute_cmf_energy(mol::Molecule, C::Matrix, rdm1s, rdm2s, clusters; verbose=0)
    e1 = zeros((length(clusters),1))
    e2 = zeros((length(clusters),length(clusters)))
    nbas = size(C)[1]
    Dzero = zeros(nbas,nbas) # no embedding
    for ci in clusters
        ints_i = FermiCG.pyscf_build_ints(mol, C[:,ci.orb_list], Dzero);
        e1[ci.idx] = FermiCG.compute_energy(0, ints_i.h1, ints_i.h2, rdm1s[ci.idx], rdm2s[ci.idx])

        moi = C[:,ci.orb_list]
        for cj in clusters
            moj = C[:,cj.orb_list]
            if ci.idx >= cj.idx
                continue
            end
            v_pqrs = FermiCG.pyscf_build_eri(mol,moi,moi,moj,moj) 
            v_psrq = FermiCG.pyscf_build_eri(mol,moi,moj,moj,moi) 
            tmp = 0
            @tensor begin
                tmp  = v_pqrs[p,q,r,s] * rdm1s[ci.idx][p,q] * rdm1s[cj.idx][r,s]
                tmp -= .5*v_psrq[p,s,r,q] * rdm1s[ci.idx][p,q] * rdm1s[cj.idx][r,s]
            end
            e2[ci.idx, cj.idx] = tmp
        end
    end
    if verbose>1
        for ei = 1:length(e1)
            @printf(" Cluster %3i E =%12.8f\n",ei,e1[ei])
        end
    end
    e0 = FermiCG.get_nuclear_rep(mol)
    return e0 + sum(e1) + sum(e2)
end



"""
    cmf_ci_iteration(mol::Molecule, C, rdm1a, rdm1b, clusters, fspace; 
                     verbose=0)

Perform single CMF-CI iteration, returning new energy, and density.
This method forms the eri's on the fly to avoid global N^4 storage

# Arguments
- `mol::Molecule`: a FermiCG.Molecule type
- `C`: MO coefficients for full system (spin restricted)
- `rdm1a`: 1particle density matrix (alpha) 
- `rdm1b`: 1particle density matrix (beta) 
- `clusters::Vector{Cluster}`: vector of Cluster objects
- `fspace::Vector{Vector{Int}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `verbose`: how much to print

See also: [`cmf_ci_iteration`](@ref)
"""
function cmf_ci_iteration(mol::Molecule, C, rdm1a, rdm1b, clusters, fspace; verbose=0)
    rdm1_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    for ci in clusters
        flush(stdout)
        # converge the MO density matrix into AOs for PYSCF
        Dembed = .5*(rdm1a + rdm1b)
        Dembed[:,ci.orb_list] .= 0
        Dembed[ci.orb_list,:] .= 0
        Dembed = C * Dembed * C'
        #
        # form integrals in subspace
        ints_i = FermiCG.pyscf_build_ints(mol, C[:,ci.orb_list], Dembed);
        #
        # use pyscf to compute FCI energy
        e, d1a,d1b, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
        rdm1_dict[ci.idx] = [d1a,d1b]
        rdm2_dict[ci.idx] = d2
    end
    e_curr = compute_cmf_energy(mol, C, rdm1_dict, rdm2_dict, clusters, verbose=verbose)
    enuc = FermiCG.get_nuclear_rep(mol)
    if verbose > 1
        @printf(" CMF-CI Curr: Elec %12.8f Total %12.8f\n", e_curr-enuc, e_curr)
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
    cmf_ci(mol::Molecule, C::Matrix, clusters::Vector{Cluster}, fspace::Vector, dguess; 
            max_iter=10, dconv=1e-6, econv=1e-10, verbose=0)

Optimize the 1RDM for CMF-CI, without requiring an InCoreInts object 

# Arguments
- `mol::Molecule`: a FermiCG.Molecule type
- `C`: MO coefficients for full system (spin restricted)
- `clusters::Vector{Cluster}`: vector of Cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `dguess`: initial guess for 1particle density matrix (spin summed) 
- `dconv`: Convergence threshold for change in density 
- `econv`: Convergence threshold for change in energy 
- `verbose`: how much to print
"""
function cmf_ci(mol::Molecule, C::Matrix, clusters::Vector{Cluster}, fspace::Vector, in_rdm1a, in_rdm1b; 
                max_iter=10, dconv=1e-6, econv=1e-10, verbose=0,squential=false)
    #rdm1a = deepcopy(dguess)
    #rdm1b = deepcopy(dguess)
    rdm1a = deepcopy(in_rdm1a)
    rdm1b = deepcopy(in_rdm1b)
    energies = []
    e_prev = 0
    e0 = FermiCG.get_nuclear_rep(mol)

    rdm1_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    for iter = 1:max_iter
        if verbose > 1
            println()
            println(" ------------------------------------------ ")
            println(" CMF CI Iter: ", iter)
            println(" ------------------------------------------ ")
        end
        e_curr, rdm1a_curr, rdm1b_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(mol, C, rdm1a, rdm1b, clusters, fspace, verbose=verbose,sequential=sequential)
        append!(energies,e_curr)
        error = (rdm1a_curr+rdm1b_curr) - (rdm1a+rdm1b)
        d_err = LinearAlgebra.norm(error)
        e_err = e_curr-e_prev
        if verbose>1
            @printf(" CMF-CI Energy: %12.8f | Change: RDM: %6.1e Energy %6.1e\n\n", e_curr, d_err, e_err)
        end
        e_prev = e_curr*1
        rdm1a = rdm1a_curr
        rdm1b = rdm1b_curr
        if (abs(d_err) < dconv) && (abs(e_err) < econv)
            if verbose>1
                @printf("*CMF-CI: Elec %12.8f Total %12.8f\n", e_curr-e0, e_curr)
            end
            break
        end
    end
    if verbose>0
        println(" Energy per Iteration:")
        for i in energies
            @printf(" Elec: %12.8f Total: %12.8f\n", i-e0, i)
        end
    end
    return e_prev, rdm1a, rdm1b, rdm1_dict, rdm2_dict
end



"""
    cmf_oo(mol::Molecule, Cguess::Matrix, clusters::Vector{Cluster}, fspace, dguess; 
            max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs")

Do CMF with orbital optimization with on the fly integrals
"""
function cmf_oo(mol::Molecule, Cguess::Matrix, clusters::Vector{Cluster}, fspace, dguess; 
                max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs")
    norb = size(Cguess)[2]
    
    e0 = FermiCG.get_nuclear_rep(mol)

    #   
    #   Initialize optimization data
    #
    e_curr = 0
    g_curr = 0
    e_err = 0
    da = zeros(norb, norb)
    db = zeros(norb, norb)
    iter = 0
    kappa = zeros(norb*(norb-1))

    #
    #   Define Objective function (energy)
    #
    function f(k)
        K = unpack_gradient(k, norb)
        U = exp(K)
        C = Cguess * U
        e, da1, db1, rdm1_dict, rdm2_dict = cmf_ci(mol, C, clusters, fspace, da+db, dconv=gconv/10.0, verbose=0)
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
            @printf("*ooCMF Iter: %4i Elec= %16.12f Total= %16.12f G= %12.2e\n", iter, e_curr, e_curr-e0, g_curr)
            return true 
        else
            @printf(" ooCMF Iter: %4i Elec= %16.12f Total= %16.12f G= %12.2e\n", iter, e_curr, e_curr-e0, g_curr)
            return false 
        end
    end

    #
    #   Define Gradient function
    #
    function g_analytic(kappa)
        # println(" In g_analytic")
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        C = Cguess * U

        e, gd1a, gd1b, rdm1_dict, rdm2_dict = cmf_ci(mol, C, clusters, fspace, da+db, dconv=gconv/10.0, verbose=verbose)
        grad = zeros(norb,norb)
        h1 = C' * FermiCG.pyscf_build_1e(mol) 
        for ci in clusters
            grad_1 = grad[:,ci.orb_list]
            moi = C[:,ci.orb_list]
            h_1   = h1 * moi
            v_111 = FermiCG.pyscf_build_eri(mol,C,moi,moi,moi) 
            @tensor begin
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,u,w,v]
                grad_1[p,q] += h_1[p,r] * rdm1_dict[ci.idx][r,q]
            end
            for cj in clusters
                if ci.idx == cj.idx
                    continue
                end
                moj = C[:,cj.orb_list]
                v_212 = FermiCG.pyscf_build_eri(mol,C,moj,moi,moj) 
                v_122 = FermiCG.pyscf_build_eri(mol,C,moi,moj,moj) 
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
                            iterations=max_iter_oo,
                           )

    res = optimize(f, g_analytic, kappa, optmethod, options; inplace = false )
    summary(res)
    e = Optim.minimum(res)
    display(res)
    @printf(" ooCMF %12.8f ", e - e0)

    kappa = Optim.minimizer(res)
    K = unpack_gradient(kappa, norb)
    U = exp(K)

    return e, U
end




