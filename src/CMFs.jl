using LinearAlgebra
using Random
using Optim



"""
	form_1rdm_dressed_ints(ints::InCoreInts, orb_list, rdm1a, rdm1b)

Obtain a subset of integrals which act on the orbitals in Cluster,
embedding the 1rdm from the rest of the system

Returns an `InCoreInts` type
"""
function form_1rdm_dressed_ints(ints::InCoreInts, orb_list, rdm1a, rdm1b)
    norb_act = size(orb_list)[1]
    full_orb = size(rdm1a)[1]
    h = zeros(norb_act,norb_act)
    f = zeros(norb_act,norb_act)
    v = zeros(norb_act,norb_act,norb_act,norb_act)
    da = zeros(norb_act,norb_act)
    db = zeros(norb_act,norb_act)


    for (pi,p) in enumerate(orb_list)
    	for (qi,q) in enumerate(orb_list)
	    h[pi,qi] = ints.h1[p,q]
	    da[pi,qi] = rdm1a[p,q]
	    db[pi,qi] = rdm1b[p,q]
	end
    end


    for (pi,p) in enumerate(orb_list)
    	for (qi,q) in enumerate(orb_list)
    	    for (ri,r) in enumerate(orb_list)
    		for (si,s) in enumerate(orb_list)
		    v[pi,qi,ri,si] = ints.h2[p,q,r,s]
		end
	    end
	end
    end

    println(" Compute single particle embedding potential")
    denv_a = 1.0*rdm1a
    denv_b = 1.0*rdm1b
    dact_a = 0.0*rdm1a
    dact_b = 0.0*rdm1b

    for (pi,p) in enumerate(orb_list)
    	for (qi,q) in enumerate(1:size(rdm1a)[1])
        denv_a[p,q] = 0
	    denv_b[p,q] = 0
	    denv_a[q,p] = 0
	    denv_b[q,p] = 0

	    dact_a[p,q] = rdm1a[p,q]
	    dact_b[p,q] = rdm1b[p,q]
	    dact_a[q,p] = rdm1a[q,p]
	    dact_b[q,p] = rdm1b[q,p]
	end
    end

    println(" Trace of env 1RDM: %12.8f",tr(denv_a + denv_b))
    #print(" Compute energy of 1rdm:")

    ga =  zeros(size(ints.h1)) 
    gb =  zeros(size(ints.h1)) 

    @tensor begin
        ga[r,s] += ints.h2[p,q,r,s] * (denv_a[p,q] + denv_b[p,q])
        ga[q,r] -= ints.h2[p,q,r,s] * (denv_a[p,s])

        gb[r,s] += ints.h2[p,q,r,s] * (denv_a[p,q] + denv_b[p,q])
        gb[q,r] -= ints.h2[p,q,r,s] * (denv_b[p,s])
    end

    De = denv_a + denv_b
    Fa = ints.h1 + .5*ga
    Fb = ints.h1 + .5*gb
    F = ints.h1 + .25*(ga + gb)
    Eenv = tr(De * F) 
   
    f = zeros(norb_act,norb_act)
    for (pi,p) in enumerate(orb_list)
        for (qi,q) in enumerate(orb_list)
            f[pi,qi] =  F[p,q]
	end
    end

    t = 2*f-h
    ints_i = InCoreInts(ints.h0, t, v)
    return ints_i
end





"""
	form_casci_ints(ints::InCoreInts, ci::Cluster, rdm1a, rdm1b)

Obtain a subset of integrals which act on the orbitals in Cluster,
embedding the 1rdm from the rest of the system

Returns an `InCoreInts` type
"""
function form_casci_ints(ints::InCoreInts, ci::Cluster, rdm1a, rdm1b)
    da = deepcopy(rdm1a)
    db = deepcopy(rdm1b)
    da[:,ci.orb_list] .= 0
    db[:,ci.orb_list] .= 0
    da[ci.orb_list,:] .= 0
    db[ci.orb_list,:] .= 0
    viirs = ints.h2[ci.orb_list, ci.orb_list,:,:]
    viqri = ints.h2[ci.orb_list, :, :, ci.orb_list]
    fa = zeros(length(ci),length(ci))
    Ja = zeros(length(ci),length(ci))
    fb = copy(fa)
    ints_i = subset(ints, ci.orb_list)
    @tensor begin
        ints_i.h1[p,q] += viirs[p,q,r,s] * (da+db)[r,s]
        # fb = deepcopy(fa)
        # fa[p,s] -= viqri[p,q,r,s] * da[q,r]
        # fb[p,s] -= viqri[p,q,r,s] * db[q,r]
        ints_i.h1[p,s] -= .5*viqri[p,q,r,s] * da[q,r]
        ints_i.h1[p,s] -= .5*viqri[p,q,r,s] * db[q,r]
    end
    return ints_i
end

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
	compute_cmf_energy(ints, rdm1s, rdm2s, clusters)

Compute the energy of a cluster-wise product state (CMF),
specified by a list of 1 and 2 particle rdms local to each cluster.
This method uses the full system integrals.

- `ints::InCoreInts`: integrals for full system
- `rdm1s`: dictionary (`ci.idx => Array`) of 1rdms from each cluster (spin summed)
- `rdm2s`: dictionary (`ci.idx => Array`) of 2rdms from each cluster (spin summed)
- `clusters::Vector{Cluster}`: vector of cluster objects

return the total CMF energy
"""
function compute_cmf_energy(ints, rdm1s, rdm2s, clusters; verbose=0)
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
        e1[ci.idx] = FermiCG.compute_energy(0, ints_i.h1, ints_i.h2, rdm1s[ci.idx][1]+rdm1s[ci.idx][2], rdm2s[ci.idx])

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

            #@tensor begin
            #    tmp  = v_pqrs[p,q,r,s] * rdm1s[ci.idx][p,q] * rdm1s[cj.idx][r,s]
            #    tmp -= .5*v_psrq[p,s,r,q] * rdm1s[ci.idx][p,q] * rdm1s[cj.idx][r,s]
            #end

            @tensor begin
                tmp  = v_pqrs[p,q,r,s] * rdm1s[ci.idx][1][p,q] * rdm1s[cj.idx][1][r,s]
                tmp -= v_psrq[p,s,r,q] * rdm1s[ci.idx][1][p,q] * rdm1s[cj.idx][1][r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx][2][p,q] * rdm1s[cj.idx][2][r,s]
                tmp -= v_psrq[p,s,r,q] * rdm1s[ci.idx][2][p,q] * rdm1s[cj.idx][2][r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx][1][p,q] * rdm1s[cj.idx][2][r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx][2][p,q] * rdm1s[cj.idx][1][r,s]
            end


            e2[ci.idx, cj.idx] = tmp
        end
    end
    if verbose>1
        for ei = 1:length(e1)
            @printf(" Cluster %3i E =%12.8f\n",ei,e1[ei])
        end
    end
    return ints.h0 + sum(e1) + sum(e2)
end


"""
    cmf_ci_iteration(mol::Molecule, C, rdm1a, rdm1b, clusters, fspace; 
                     verbose=1)

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
- `ci_max_iter`: How many FCI iterations do we allow for each cluster. Default=40
- `ci_conv_tol`: How what is the convergence (gradient norm) for the FCI on each cluster. Default=1e-8 

See also: [`cmf_ci_iteration`](@ref)
"""
function cmf_ci_iteration(mol::Molecule, C, rdm1a, rdm1b, clusters, fspace; 
                          verbose=0, 
                          ci_max_iter=40,
                          ci_conv_tol=1e-8)
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
        e, d1a,d1b, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2],
                                          max_cycle=ci_max_iter, conv_tol=ci_conv_tol, verbose=verbose)
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
    cmf_ci_iteration(ints::InCoreInts, clusters::Vector{Cluster}, rdm1a, rdm1b, fspace; verbose=1)

Perform single CMF-CI iteration, returning new energy, and density
"""
function cmf_ci_iteration(ints::InCoreInts, clusters::Vector{Cluster}, rdm1a, rdm1b, fspace; verbose=1,sequential=false)
    rdm1_dict = Dict{Integer,Array}()
    rdm1s_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    for ci in clusters
        flush(stdout)

        ansatz = FCIAnsatz(length(ci), fspace[ci.idx][1],fspace[ci.idx][2])
        verbose < 2 || display(ansatz)
        ints_i = form_casci_ints(ints, ci, rdm1a, rdm1b)

        no = length(ci)
        e = 0.0
        d1 = zeros(no, no)
        d1a =zeros(no,no)
        d1b =zeros(no,no)
        d2 = zeros(no, no, no, no)
        if ansatz.dim == 1

            #
            # we have a slater determinant. Compute energy and dms

            na = fspace[ci.idx][1]
            nb = fspace[ci.idx][2]

            if (na == no) && (nb == no)
                #
                # a doubly occupied space
                d1 = Matrix(1.0I, no, no)
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2[p,q,r,s] = 2*d1[p,q]*d1[r,s] - d1[p,s]*d1[r,q]
                end
                d1a = d1 
                d1b = d1 
                d2 *= 2.0
                e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

            elseif (na == no) && (nb == 0)
                #
                # singly occupied space
                d1 = Matrix(1.0I, no, no)
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2[p,q,r,s] = d1[p,q]*d1[r,s] - d1[p,s]*d1[r,q]
                end
                d1a  = d1
                d1b  = zeros(no,no)
                e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

            elseif (na == 0) && (nb == no)
                #
                # singly occupied space
                d1 = Matrix(1.0I, no, no)
                for p in 1:no, q in 1:no, r in 1:no, s in 1:no
                    d2[p,q,r,s] = d1[p,q]*d1[r,s] - d1[p,s]*d1[r,q]
                end
                d1a  = zeros(no,no)
                d1b  = d1
                e = compute_energy(0, ints_i.h1, ints_i.h2, d1, d2)
                verbose == 0 || @printf(" Slater Det Energy: %12.8f\n", e)

            elseif (na == 0) && (nb==0)
                # 
                # a virtual space (do nothing)
            else
                error(" How can this be?")
            end
            #e, d1, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
        else
            #
            # run PYSCF FCI
            e, d1a,d1b, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
        end

        rdm1_dict[ci.idx] = [d1a,d1b]
        rdm1s_dict[ci.idx] = d1a+d1b
        rdm2_dict[ci.idx] = d2
        #display(d1a-d1b)

        if sequential==true
            rdm1a[ci.orb_list,ci.orb_list] = d1a
            rdm1b[ci.orb_list,ci.orb_list] = d1b
        end
    end
    e_curr = compute_cmf_energy(ints, rdm1_dict, rdm2_dict, clusters, verbose=verbose)
    if verbose > 1
        @printf(" CMF-CI Curr: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
    end

    rdm1a_out = zeros(size(rdm1a))
    rdm1b_out = zeros(size(rdm1b))
    for ci in clusters
        rdm1a_out[ci.orb_list, ci.orb_list] .= rdm1_dict[ci.idx][1]
        rdm1b_out[ci.orb_list, ci.orb_list] .= rdm1_dict[ci.idx][2]
    end
    return e_curr,rdm1a_out, rdm1b_out, rdm1_dict, rdm2_dict
end


"""
    cmf_ci(mol::Molecule, C::Matrix, clusters::Vector{Cluster}, fspace::Vector, dguess; 
            max_iter=10, dconv=1e-6, econv=1e-10, verbose=1)

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
                max_iter=10, dconv=1e-6, econv=1e-10, verbose=1,squential=false)
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
        e_curr, rdm1a_curr, rdm1b_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(mol, C, rdm1a, rdm1b, clusters, fspace, verbose=verbose)
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
    cmf_ci(ints, clusters, fspace, in_rdm1a, in_rdm1b; 
                max_iter=10, dconv=1e-6, econv=1e-10, verbose=1,sequential=false)

Optimize the 1RDM for CMF-CI

#Arguments
- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{Cluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `in_rdm1a`: initial guess for 1particle density matrix for alpha electrons
- `in_rdm1b`: initial guess for 1particle density matrix for beta electrons
- `dconv`: Convergence threshold for change in density 
- `econv`: Convergence threshold for change in energy 
- `sequential`: Use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
"""
function cmf_ci(ints, clusters, fspace, in_rdm1a, in_rdm1b; 
                max_iter=10, dconv=1e-6, econv=1e-10, verbose=1,sequential=false)
    rdm1a = deepcopy(in_rdm1a)
    rdm1b = deepcopy(in_rdm1b)
    energies = []
    e_prev = 0

    rdm1_dict = 0
    rdm2_dict = 0
    rdm1_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    # rdm2_dict = Dict{Integer, Array}()
    for iter = 1:max_iter
        if verbose > 1
            println()
            println(" ------------------------------------------ ")
            println(" CMF CI Iter: ", iter)
            println(" ------------------------------------------ ")
        end
        e_curr, rdm1a_curr, rdm1b_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace, verbose=verbose,sequential=sequential)
        append!(energies,e_curr)
        error = (rdm1a_curr+rdm1b_curr) - (rdm1a+rdm1b)
        d_err = norm(error)
        e_err = e_curr-e_prev
        if verbose>1
            @printf(" CMF-CI Energy: %12.8f | Change: RDM: %6.1e Energy %6.1e\n\n", e_curr, d_err, e_err)
        end
        e_prev = e_curr*1
        rdm1a = rdm1a_curr
        rdm1b = rdm1b_curr
        if (abs(d_err) < dconv) && (abs(e_err) < econv)
            if verbose>1
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








"""
    cmf_oo(ints::InCoreInts, clusters::Vector{Cluster}, fspace, dguess_a, dguess_b; 
                max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs", alpha=nothing,sequential=false)

Do CMF with orbital optimization

#Arguments
- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{Cluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `dguess_a`: initial guess for 1particle density matrix for alpha electrons
- `dguess_b`: initial guess for 1particle density matrix for beta electrons
- `max_iter_oo`: Max iter for the orbital optimization iterations 
- `max_iter_ci`: Max iter for the cmf iteration for the cluster states 
- `gconv`: Convergence threshold for change in gradient of energy 
- `sequential`: If true use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
- `method`: optimization method
"""
function cmf_oo(ints::InCoreInts, clusters::Vector{Cluster}, fspace, dguess_a, dguess_b; 
                max_iter_oo=100, 
                max_iter_ci=100, 
                gconv=1e-6, 
                verbose=0, 
                method="bfgs", 
                alpha=nothing,
                sequential=false)
    norb = size(ints.h1)[1]
    #kappa = zeros(norb*(norb-1))
    # e, da, db = cmf_oo_iteration(ints, clusters, fspace, max_iter_ci, dguess, kappa)

    function g_numerical(k)
        stepsize = 1e-6
        grad = zeros(size(k))
        for (ii,i) in enumerate(k)
            k1 = deepcopy(k)
            k1[ii] += stepsize
            e1 = f(k1) 
            k2 = deepcopy(k)
            k2[ii] -= stepsize
            e2 = f(k2) 
            grad[ii] = (e1-e2)/(2*stepsize)
        end
        g_curr = norm(grad)
        return grad
    end

    #   
    #   Initialize optimization data
    #
    e_prev = 0
    e_curr = 0
    g_curr = 0
    e_err = 0
    #da = zeros(size(ints.h1))
    #db = zeros(size(ints.h1))
    da = deepcopy(dguess_a)
    db = deepcopy(dguess_b)

    da1 = deepcopy(dguess_a)
    db1 = deepcopy(dguess_b)

    da2 = deepcopy(dguess_a)
    db2 = deepcopy(dguess_b)

    iter = 0
    kappa = zeros(norb*(norb-1))

    #
    #   Define Objective function (energy)
    #
    function f(k)
        #display(norm(k))
        K = unpack_gradient(k, norb)
        U = exp(K)
        ints2 = orbital_rotation(ints,U)
        da1 = U'*da*U
        db1 = U'*db*U
        e, da1, db1, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da1, db1, dconv=gconv/10.0, verbose=0,sequential=sequential)
        da2 = U*da1*U'
        db2 = U*db1*U'
        e_err = e-e_curr
        e_curr = e
        return e
    end

    #   
    #   Define Callback for logging and checking for convergence
    #
    function callback(k)
       
        # reset initial RDM guess for each cmf_ci
        da = deepcopy(da2)
        db = deepcopy(db2)

        #if e_err > 0
        #    @warn " energy increased"
        #    return true
        #end
        iter += 1
        if (g_curr < gconv) 
            @printf("*ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return true 
        else
            @printf(" ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return false 
        end
    end

    #
    #   Define Gradient function
    #
    function g(kappa)
        norb = size(ints.h1)[1]
        # println(" In g_analytic")
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        #println(size(U), size(kappa))
        ints2 = orbital_rotation(ints,U)
        da1 = U'*da*U
        db1 = U'*db*U
        
        e, gd1a, gd1b, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da1, db1, dconv=gconv/10.0, verbose=verbose)
        grad = zeros(size(ints2.h1))
        for ci in clusters
            grad_1 = grad[:,ci.orb_list]
            h_1	   = ints2.h1[:,ci.orb_list]
            v_111  = ints2.h2[:, ci.orb_list, ci.orb_list, ci.orb_list]
            @tensor begin
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,v,u,w]
                #grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,u,w,v]
                #grad_1[p,q] += h_1[p,r] * rdm1_dict[ci.idx][r,q]
                grad_1[p,q] += h_1[p,r] * (rdm1_dict[ci.idx][1][r,q]+rdm1_dict[ci.idx][2][r,q])
            end
            for cj in clusters
                if ci.idx == cj.idx
                    continue
                end
                v_212 = ints2.h2[:,cj.orb_list, ci.orb_list, cj.orb_list]
                v_122 = ints2.h2[:,ci.orb_list, cj.orb_list, cj.orb_list]
                d1 = rdm1_dict[ci.idx][1] + rdm1_dict[ci.idx][2]
                d2 = rdm1_dict[cj.idx][1] + rdm1_dict[cj.idx][2]
                
                d1a = rdm1_dict[ci.idx][1]
                d1b = rdm1_dict[ci.idx][2]
                d2a = rdm1_dict[cj.idx][1]
                d2b = rdm1_dict[cj.idx][2]

                @tensor begin
                    #grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                    #grad_1[p,q] -= .5*v_212[p,v,u,w] * d1[q,u] * d2[w,v]
                    grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                    grad_1[p,q] -= v_212[p,v,u,w] * d1a[q,u] * d2a[w,v]
                    grad_1[p,q] -= v_212[p,v,u,w] * d1b[q,u] * d2b[w,v]
                end
            end
            grad[:,ci.orb_list] .= -2*grad_1
        end
        grad = grad'-grad
        gout = pack_gradient(grad, norb)
        g_curr = norm(gout)
        return gout
    end

#    grad1 = g_numerical(kappa)
#    grad2 = g(kappa)
#    display(round.(unpack_gradient(grad1, norb),digits=6))
#    display(round.(unpack_gradient(grad2, norb),digits=6))
#    return

    #display("here:")
    #gerr = g_numerical(kappa) - g(kappa)
    #display(norm(gerr))
    #for i in gerr
    #    @printf(" err: %12.8f\n",i)
    #end
    #return

    if (method=="bfgs") || (method=="cg") || (method=="gd")
        optmethod = BFGS()
        if method=="cg"
            optmethod = ConjugateGradient()
        elseif method=="gd"

            if alpha == nothing
                optmethod = GradientDescent()
            else 
                optmethod = GradientDescent(alphaguess=alpha)
            end
        end

        options = Optim.Options(
                                callback = callback, 
                                g_tol=gconv,
                                iterations=max_iter_oo,
                               )

        #res = optimize(f, g_numerical, kappa, optmethod, options; inplace = false )
        res = optimize(f, g, kappa, optmethod, options; inplace = false )
        summary(res)
        e = Optim.minimum(res)
        display(res)
        @printf("*ooCMF %12.8f \n", e)

        kappa = Optim.minimizer(res)
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        da1 = U'*da*U
        db1 = U'*db*U
        return e, U, da1, db1
    elseif method=="diis"
        res = do_diis(f, g, callback, kappa, gconv, max_iter_oo, method)
    end

end



function do_diis(f,g,callback,kappa, gconv,max_iter, method)
    throw("Not yet implemented")
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
