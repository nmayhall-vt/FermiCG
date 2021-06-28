using LinearAlgebra
using Random
using Optim



"""
	form_1rdm_dressed_ints(ints::InCoreInts, orb_list, rdm1a, rdm1b)

Obtain a subset of integrals which act on the orbitals in Cluster,
embedding the 1rdm from the rest of the system

Returns an `InCoreInts` type
"""
function form_1rdm_dressed_ints(ints::InCoreInts, orb_list, rdm1a, rdm1b, verbose=0)
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

    verbose == 0 || println(" Compute single particle embedding potential")
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

    verbose == 0 || @printf(" Trace of env 1RDM: %12.8f\n",tr(denv_a + denv_b))
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
    #return form_1rdm_dressed_ints(ints, ci.orb_list, rdm1a, rdm1b)
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
	compute_cmf_energy(ints, rdm1s, rdm2s, clusters)

Compute the energy of a cluster-wise product state (CMF),
specified by a list of 1 and 2 particle rdms local to each cluster.
This method uses the full system integrals.

- `ints::InCoreInts`: integrals for full system
- `rdm1s`: dictionary (`ci.idx => Array`) of 1rdms from each cluster as list: [Da,Db] 
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
        # 	tmp += h_pq[p,q] * rdm-1s[ci.idx][q,p]
        # 	tmp += .5 * v_pqrs[p,q,r,s] * rdm2s[ci.idx][p,q,r,s]
        # end
        e1[ci.idx] = FermiCG.compute_energy(0, ints_i.h1, ints_i.h2, rdm1s[ci.idx][1]+rdm1s[ci.idx][2], rdm2s[ci.idx])
        #display(("nick ", e1[ci.idx]))
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
    if verbose>0
        for ei = 1:length(e1)
            @printf(" Cluster %3i E =%12.8f\n",ei,e1[ei])
        end
    end
    return ints.h0 + sum(e1) + sum(e2)
end


"""
    cmf_ci_iteration(ints::InCoreInts, clusters::Vector{Cluster}, rdm1a, rdm1b, fspace; verbose=0)

Perform single CMF-CI iteration, returning new energy, and density
"""
function cmf_ci_iteration(ints::InCoreInts, clusters::Vector{Cluster}, rdm1a, rdm1b, fspace; 
                          verbose=0, sequential=false, max_ci_iter=100)
    rdm1_dict = Dict{Integer,Array}()
    rdm1s_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    #verbose = 2
    for ci in clusters
        flush(stdout)

        problem = FermiCG.StringCI.FCIProblem(length(ci), fspace[ci.idx][1],fspace[ci.idx][2])
        verbose < 2 || display(problem)
        ints_i = form_casci_ints(ints, ci, rdm1a, rdm1b)

        no = length(ci)
        e = 0.0
        d1 = zeros(no, no)
        d1a =zeros(no,no)
        d1b =zeros(no,no)
        d2 = zeros(no, no, no, no)
        if problem.dim == 1

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
            e, d1a, d1b, d2 = FermiCG.pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
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
        rdm1a_out[ci.orb_list, ci.orb_list] = rdm1_dict[ci.idx][1]
        rdm1b_out[ci.orb_list, ci.orb_list] = rdm1_dict[ci.idx][2]
    end
    return e_curr, rdm1a_out, rdm1b_out, rdm1_dict, rdm2_dict
end


"""
    cmf_ci(ints, clusters, fspace, dguess; 
            max_iter=10, dconv=1e-6, econv=1e-10, verbose=0)

Optimize the 1RDM for CMF-CI
"""
function cmf_ci(ints, clusters, fspace, in_rdm1a, in_rdm1b; 
                max_iter=100, dconv=1e-6, econv=1e-10, verbose=0,sequential=false)
#={{{=#
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
        e_curr, rdm1a_curr, rdm1b_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, fspace, verbose=verbose)
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
        eprev = 0.0
        for i in energies
            @printf(" Elec: %12.8f Total: %12.8f Change: %8.2e\n", i-ints.h0, i, i-eprev)
            eprev = i
        end
    end
    return e_prev, rdm1a, rdm1b, rdm1_dict, rdm2_dict
end
#=}}}=#






"""
Orbital gradient 
"""
function cmf_orb_grad(ints::InCoreInts, clusters, fspace, rdm1_dict, rdm2_dict)
    #={{{=#
    norb = size(ints.h1,1)

    if 1==0
        full_2rdm = zeros(size(ints.h2))
        full_1rdm = zeros(size(ints.h1))
        full_1rdmb = zeros(size(ints.h1))
        full_1rdma = zeros(size(ints.h1))
        for ci in clusters
            for (pi,p) in enumerate(ci.orb_list)
                for (qi,q) in enumerate(ci.orb_list)
                    full_1rdma[p,q] = rdm1_dict[ci.idx][1][pi,qi] 
                    full_1rdmb[p,q] = rdm1_dict[ci.idx][2][pi,qi] 
                end
            end
        end
        full_1rdm = full_1rdma + full_1rdmb

        for p in 1:norb
            for q in 1:norb
                for r in 1:norb
                    for s in 1:norb
                        full_2rdm[p,q,r,s] = full_1rdm[p,q]*full_1rdm[r,s] - full_1rdma[p,s]*full_1rdma[r,q] - full_1rdmb[p,s]*full_1rdmb[r,q]
                    end
                end
            end
        end
        for ci in clusters
            for (pi,p) in enumerate(ci.orb_list)
                for (qi,q) in enumerate(ci.orb_list)
                    for (ri,r) in enumerate(ci.orb_list)
                        for (si,s) in enumerate(ci.orb_list)
                            full_2rdm[p,q,r,s] = rdm2_dict[ci.idx][pi,qi,ri,si]
                        end
                    end
                end
            end
        end
        etest = compute_energy(ints, full_1rdm, full_2rdm)
        println(" nick: ", etest)
        etest = compute_cmf_energy(ints, rdm1_dict, rdm2_dict, clusters)
        println(" nick: ", etest)
        F = zeros(size(full_1rdm))   
        @tensor begin
            F[p,q] += ints.h1[p,v] * full_1rdm[v,q] 
            F[p,q] += ints.h2[p,v,u,w] * full_2rdm[q,v,u,w] 
        end
        grad = -2.0.*(F'-F)
        gout = pack_gradient(grad, norb)
        g_curr = norm(gout)
        #error("test")
        return gout
    end
    grad = zeros(size(ints.h1))
    for ci in clusters
        grad_1 = grad[:,ci.orb_list]
        h_1	   = ints.h1[:,ci.orb_list]
        v_111  = ints.h2[:, ci.orb_list, ci.orb_list, ci.orb_list]
        @tensor begin
            grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,v,u,w]
            grad_1[p,q] += h_1[p,r] * (rdm1_dict[ci.idx][1][r,q]+rdm1_dict[ci.idx][2][r,q])
            #grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,u,w,v]
            #grad_1[p,q] += h_1[p,r] * rdm1_dict[ci.idx][r,q]
        end
        for cj in clusters
            if ci.idx == cj.idx
                continue
            end
            v_212 = ints.h2[:,cj.orb_list, ci.orb_list, cj.orb_list]
            v_122 = ints.h2[:,ci.orb_list, cj.orb_list, cj.orb_list]
            #v_221 = ints2.h2[:,cj.orb_list, cj.orb_list, ci.orb_list]
            d1 = rdm1_dict[ci.idx][1] + rdm1_dict[ci.idx][2]
            d2 = rdm1_dict[cj.idx][1] + rdm1_dict[cj.idx][2]

            d1a = rdm1_dict[ci.idx][1]
            d1b = rdm1_dict[ci.idx][2]
            d2a = rdm1_dict[cj.idx][1]
            d2b = rdm1_dict[cj.idx][2]

            @tensor begin
                #grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                #grad_1[p,q] -= .5*v_212[p,v,u,w] * d1[q,u] * d2[w,v]

                #grad_1[p,q] += v_122[p,v,u,w] * d1a[q,v] * d2a[u,w]
                #grad_1[p,q] -= v_221[p,v,u,w] * d1a[q,w] * d2a[u,v]
                #grad_1[p,q] += v_122[p,v,u,w] * d1b[q,v] * d2b[u,w]
                #grad_1[p,q] -= v_221[p,v,u,w] * d1b[q,w] * d2b[u,v]
                #grad_1[p,q] += v_122[p,v,u,w] * d1a[q,v] * d2b[u,w]
                #grad_1[p,q] += v_122[p,v,u,w] * d1b[q,v] * d2a[u,w]

                grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[u,w]
                grad_1[p,q] -= v_212[p,v,u,w] * d1a[q,u] * d2a[w,v]
                grad_1[p,q] -= v_212[p,v,u,w] * d1b[q,u] * d2b[w,v]
            end
        end
        grad[:,ci.orb_list] .+= -2*grad_1
    end
    grad = grad'-grad
    gout = pack_gradient(grad, norb)
    g_curr = norm(gout)
    return gout
end


#=}}}=#





"""
    cmf_oo(ints::InCoreInts, clusters::Vector{Cluster}, fspace, dguess; 
            max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs")

Do CMF with orbital optimization
"""
function cmf_oo(ints::InCoreInts, clusters::Vector{Cluster}, fspace, dguess_a, dguess_b; 
                max_iter_oo=100, max_iter_ci=100, gconv=1e-6, verbose=0, method="bfgs", alpha=nothing)
    norb = size(ints.h1)[1]
    #kappa = zeros(norb*(norb-1))
    # e, da, db = cmf_oo_iteration(ints, clusters, fspace, max_iter_ci, dguess, kappa)

    function g_numerical(k; num=nothing)
        stepsize = 1e-5
        grad = zeros(size(k))
        if num == nothing 
            num = length(k)
        end
        @printf(" Compute finite-difference orbital gradients for %12i parameters\n",num)
        for (ii,i) in enumerate(k)
            ii < num || continue 
            #display(ii)
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
    kappa = zeros(norb*(norb-1)÷2)

    rdm1_dict_curr  = Dict()
    rdm2_dict_curr  = Dict()

    rdm1_dict_curr2 = Dict()
    rdm2_dict_curr2 = Dict()

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
        e, da1, db1, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da1, db1, dconv=gconv/10.0, verbose=verbose)
        
        # save data which we can set as the official current set when we hit callback
        rdm1_dict_curr2 = rdm1_dict
        rdm2_dict_curr2 = rdm2_dict
        rdm1_dict_curr = deepcopy(rdm1_dict_curr2)
        rdm2_dict_curr = deepcopy(rdm2_dict_curr2) 

        da2 = U*da1*U'
        db2 = U*db1*U'
        e_err = e-e_curr
        e_curr = e
        #@printf(" Energy in energy  : %16.12f\n", e)
        return e
    end

    #   
    #   Define Callback for logging and checking for convergence
    #
    function callback(k)
       
        # reset initial RDM guess for each cmf_ci
        da = deepcopy(da2)
        db = deepcopy(db2)
        
        rdm1_dict_curr = deepcopy(rdm1_dict_curr2)
        rdm2_dict_curr = deepcopy(rdm2_dict_curr2) 

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
    
    function g2(kappa)
        norb = size(ints.h1)[1]
        # println(" In g_analytic")
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        #println(size(U), size(kappa))
        ints2 = orbital_rotation(ints,U)
        da1 = U'*da*U
        db1 = U'*db*U
       
        e, gd1a, gd1b, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da1, db1, dconv=gconv/100.0, verbose=verbose)

        return cmf_orb_grad(ints2, clusters, fspace, rdm1_dict, rdm2_dict)
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
       
        e, gd1a, gd1b, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, da1, db1, dconv=gconv/100.0, verbose=verbose)
       
        @printf(" Energy in gradient: %16.12f\n", e)
        #rdm1_dict = rdm1_dict_curr
        #rdm2_dict = rdm2_dict_curr
        if 1==1
            full_2rdm = zeros(size(ints2.h2))
            full_1rdm = zeros(size(ints2.h1))
            full_1rdmb = zeros(size(ints2.h1))
            full_1rdma = zeros(size(ints2.h1))
            for ci in clusters
                for (pi,p) in enumerate(ci.orb_list)
                    for (qi,q) in enumerate(ci.orb_list)

                        full_1rdm[p,q] = rdm1_dict[ci.idx][1][pi,qi] + rdm1_dict[ci.idx][2][pi,qi]
                        full_1rdma[p,q] = rdm1_dict[ci.idx][1][pi,qi] 
                        full_1rdmb[p,q] = rdm1_dict[ci.idx][2][pi,qi] 
                        for (ri,r) in enumerate(ci.orb_list)
                            for (si,s) in enumerate(ci.orb_list)
                                full_2rdm[p,q,r,s] = rdm2_dict[ci.idx][pi,qi,ri,si]
                            end
                        end
                    end
                end
            end
            for ci in clusters
                for (pi,p) in enumerate(ci.orb_list)
                    for (qi,q) in enumerate(ci.orb_list)
                        Di = rdm1_dict[ci.idx][1] + rdm1_dict[ci.idx][2]
                        Dai = rdm1_dict[ci.idx][1]
                        Dbi = rdm1_dict[ci.idx][2]
                        
                        for cj in clusters
                            ci.idx != cj.idx || continue
                            
                            Dj = rdm1_dict[cj.idx][1] + rdm1_dict[cj.idx][2]
                            Daj = rdm1_dict[cj.idx][1]
                            Dbj = rdm1_dict[cj.idx][2]
                            for (ri,r) in enumerate(cj.orb_list)
                                for (si,s) in enumerate(cj.orb_list)
                                    full_2rdm[p,q,r,s] += Di[pi,qi] * Dj[ri,si] 
                                    full_2rdm[p,s,r,q] -= Dai[pi,qi] * Daj[ri,si] 
                                    full_2rdm[p,s,r,q] -= Dbi[pi,qi] * Dbj[ri,si] 
                                end
                            end
                        end
                    end

                end
            end

            for p in 1:norb
                for q in 1:norb
                    for r in 1:norb
                        for s in 1:norb
                            full_2rdm[p,q,r,s] = full_1rdm[p,q]*full_1rdm[r,s] - full_1rdma[p,s]*full_1rdma[r,q]- full_1rdmb[p,s]*full_1rdmb[r,q]
                        end
                    end
                end
            end
            for ci in clusters
                for (pi,p) in enumerate(ci.orb_list)
                    for (qi,q) in enumerate(ci.orb_list)
                        for (ri,r) in enumerate(ci.orb_list)
                            for (si,s) in enumerate(ci.orb_list)
                                full_2rdm[p,q,r,s] = rdm2_dict[ci.idx][pi,qi,ri,si]
                            end
                        end
                    end
                end
            end
            etest = compute_energy(ints2, full_1rdm, full_2rdm)
            println(" nick: ", etest)
            F = zeros(size(full_1rdm))   
            @tensor begin
                F[p,q] += ints.h1[p,v] * full_1rdm[v,q] 
                F[p,q] += ints.h2[p,v,u,w] * full_2rdm[q,v,u,w] 
            end
            grad = -2.0.*(F'-F)
            gout = pack_gradient(grad, norb)
            g_curr = norm(gout)
            println(" nick: ", g_curr)
            return gout

        end

        
        grad = zeros(size(ints2.h1))
        for ci in clusters
            grad_1 = grad[:,ci.orb_list]
            h_1	   = ints2.h1[:,ci.orb_list]
            v_111  = ints2.h2[:, ci.orb_list, ci.orb_list, ci.orb_list]
            @tensor begin
                grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,v,u,w]
                grad_1[p,q] += h_1[p,r] * (rdm1_dict[ci.idx][1][r,q]+rdm1_dict[ci.idx][2][r,q])
                #grad_1[p,q] += v_111[p,v,u,w] * rdm2_dict[ci.idx][q,u,w,v]
                #grad_1[p,q] += h_1[p,r] * rdm1_dict[ci.idx][r,q]
            end
            for cj in clusters
                if ci.idx == cj.idx
                    continue
                end
                v_212 = ints2.h2[:,cj.orb_list, ci.orb_list, cj.orb_list]
                v_122 = ints2.h2[:,ci.orb_list, cj.orb_list, cj.orb_list]
                #v_221 = ints2.h2[:,cj.orb_list, cj.orb_list, ci.orb_list]
                d1 = rdm1_dict[ci.idx][1] + rdm1_dict[ci.idx][2]
                d2 = rdm1_dict[cj.idx][1] + rdm1_dict[cj.idx][2]
                
                d1a = rdm1_dict[ci.idx][1]
                d1b = rdm1_dict[ci.idx][2]
                d2a = rdm1_dict[cj.idx][1]
                d2b = rdm1_dict[cj.idx][2]

                @tensor begin
                    #grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[w,u]
                    #grad_1[p,q] -= .5*v_212[p,v,u,w] * d1[q,u] * d2[w,v]
                    
                    #grad_1[p,q] += v_122[p,v,u,w] * d1a[q,v] * d2a[u,w]
                    #grad_1[p,q] -= v_221[p,v,u,w] * d1a[q,w] * d2a[u,v]
                    #grad_1[p,q] += v_122[p,v,u,w] * d1b[q,v] * d2b[u,w]
                    #grad_1[p,q] -= v_221[p,v,u,w] * d1b[q,w] * d2b[u,v]
                    #grad_1[p,q] += v_122[p,v,u,w] * d1a[q,v] * d2b[u,w]
                    #grad_1[p,q] += v_122[p,v,u,w] * d1b[q,v] * d2a[u,w]
                    
                    grad_1[p,q] += v_122[p,v,u,w] * d1[q,v] * d2[u,w]
                    grad_1[p,q] -= v_212[p,v,u,w] * d1a[q,u] * d2a[w,v]
                    grad_1[p,q] -= v_212[p,v,u,w] * d1b[q,u] * d2b[w,v]
                end
            end
            grad[:,ci.orb_list] .+= -2*grad_1
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


    if 1==1 
        display("here:")
        #g1 = g_numerical(kappa,num=10)
        g2 = g(kappa)
        g1 = g_numerical(kappa)
        #g2[10:end] .= 0.0
        #g2[10:end] .= 0.0
        gerr = g1-g2
        println(" Numerical: ")
        display(unpack_gradient(g1, norb))
        println(" Analytical: ")
        display(unpack_gradient(g2, norb))
        #gerr = g_numerical(kappa) - g(kappa)
        #gerr = g_numerical(kappa,num=10) - g(kappa)
        println(" Numerical: ")
        display(norm(g1))
        println(" Analytical: ")
        display(norm(g2))
        println(" Error: ")
        display(norm(gerr))
        for i in gerr
            @printf(" err: %12.8f\n",i)
        end
        error(" testing gradients")
    end

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

        res = optimize(f, g_numerical, kappa, optmethod, options; inplace = false )
        #res = optimize(f, g, kappa, optmethod, options; inplace = false )
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
    kout = zeros(norb*(norb-1)÷2)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            kout[ind] = K[i,j]
            ind += 1
        end
    end
    return kout
end
