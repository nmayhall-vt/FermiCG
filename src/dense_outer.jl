#
#
#       None of this probably works!!!
#
#
"""
0 = <x|H - E0|x'>v(x') + <x|H - E0|p>v(p) 
0 = <x|H - E0|x'>v(x') + <x|H|p>v(p) 
A(x,x')v(x') = -H(x,p)v(p)

here, x is outside the reference space, and p is inside

Ax=b

works for one root at a time
"""
function tucker_cepa_solve!(ref_vector::BSstate, ci_vector::BSstate, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    fold!(ref_vector) 
    fold!(ci_vector) 
    sig = deepcopy(ref_vector) 
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham)
    e0 = dot(ref_vector, sig)
    size(e0) == (1,1) || error("Only one state at a time please")
    e0 = e0[1,1]
    @printf(" Reference Energy: %12.8f\n",e0)
    

    x_vector = deepcopy(ci_vector)
    #
    # now remove reference space from ci_vector
    for (fock,configs) in ref_vector
        if haskey(x_vector, fock)
            for (config,coeffs) in configs
                if haskey(x_vector[fock], config)
                    delete!(x_vector[fock], config)
                end
            end
        end
    end

    b = deepcopy(x_vector) 
    zero!(b)
    build_sigma!(b, ref_vector, cluster_ops, clustered_ham)
    bv = -get_vector(b) 

    function mymatvec(v)
        unfold!(x_vector)
        set_vector!(x_vector, v)
        fold!(x_vector)
        sig = deepcopy(x_vector)
        zero!(sig)
        build_sigma!(sig, x_vector, cluster_ops, clustered_ham)
        unfold!(x_vector)
        unfold!(sig)
        
        sig_out = get_vector(sig)
        sig_out .-= e0*get_vector(x_vector)
        return sig_out
    end
    dim = length(x_vector)
    Axx = LinearMap(mymatvec, dim, dim)
    #Axx = LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
    
    x, solver = cg!(get_vector(x_vector), Axx,bv,log=true)

    set_vector!(x_vector, x)
   
    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = dot(sig,ref_vector)
    size(ecorr) == (1,1) || error(" Dimension Error")
    ecorr = ecorr[1]
  
    zero!(ci_vector)
    add!(ci_vector, ref_vector)
    add!(ci_vector, x_vector)

    #x, info = linsolve(Hmap,zeros(size(v0)))
    return ecorr+e0, x
end#=}}}=#

"""
    get_foi(v::BSstate, clustered_ham, q_spaces; nroots=1, nbody=2)
Compute the first-order interacting space as defined by clustered_ham

e.g., 
 1(p') 3(r) 4(q's)  *  v[(1,1),(1,1),(1,1),(1,1)][1:1,1:1,1:1,1:1]  =>  v[(2,1), (1,1), (0,1), (1,1)][1:N, 1:1, 1:N, 1:N]
"""
function get_foi(v::BSstate, clustered_ham, q_spaces; nroots=1, nbody=2)
    println(" Prepare empty BSstate spanning the FOI of input")#={{{=#
    foi = deepcopy(v)
    na = 0
    nb = 0


    for (fock, tconfigs) in v
            
        na = sum([f[1] for f in fock])
        nb = sum([f[2] for f in fock])
        for (fock_trans, terms) in clustered_ham

            # 
            # new fock sector configuration
            new_fock = fock + fock_trans


            # 
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in v.clusters
                if new_fock[ci.idx][1] > length(ci) || new_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if new_fock[ci.idx][1] < 0 || new_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue


            if haskey(v.data, new_fock) == false
                add_fockconfig!(foi, new_fock)
            end
            

            #
            # find the cluster state index ranges (TuckerConfig) reached by Hamiltonian
            for (tconfig, coeffs) in tconfigs 
                for term in terms
                    new_tconfig = deepcopy(tconfig)

                    length(term.clusters) <= nbody || continue

                    #
                    # for current term, expand index ranges for active clusters
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        #tmp_spaces[ci.idx] = q_spaces[ci.idx][new_fock[ci.idx]]
                        #start = 1
                        #stop  = size(bases[ci.idx][new_fock[ci.idx]])[2]
                        #new_tconfig[ci.idx] = start:stop 
                        new_tconfig[ci.idx] = q_spaces[ci.idx][new_fock[ci.idx]]
                    end

                    # 
                    # remove any previously defined TuckerConfigs which are subspaces of the new TuckerConfig 
                    for key in keys(foi[new_fock])
                        if is_subset(key, new_tconfig)
                            delete!(foi[new_fock], key)
                        end
                    end

                    # 
                    # determine if new TuckerConfig is a subspace of any previously defined TuckerConfigs 
                    is_new = true
                    for key in keys(foi[new_fock])
                        if is_subset(new_tconfig, key)
                            is_new = false
                        end
                    end
                    if is_new == true
                        foi[new_fock][new_tconfig] = zeros(length.(new_tconfig)..., nroots) 
                    end
                end
            end
        end
    end
    prune_empty_fock_spaces!(foi)
    return foi
#=}}}=#
end

"""
    get_nbody_tucker_space(v::BSstate, p_spaces, q_spaces; nroots=1, nbody=2)
Get a vector dimensioned according to the n-body Tucker scheme
- `v::BSstate` = reference P-space vector
- `p_spaces` = `Vector{ClusterSubspace}` denoting all the cluster P-spaces
- `q_spaces` = `Vector{ClusterSubspace}` denoting all the cluster Q-spaces
- `nbody`    = n-body order
"""
function get_nbody_tucker_space(v::BSstate, p_spaces, q_spaces, na, nb; nroots=1, nbody=2)
    clusters = v.clusters
    println(" Prepare empty BSstate spanning the n-body Tucker space with nbody = ", nbody)#={{{=#
    ci_vector = deepcopy(v)
    if nbody >= 1 
        for ci in clusters
            tmp_spaces = copy(p_spaces)
            tmp_spaces[ci.idx] = q_spaces[ci.idx]
            FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, nb))
        end
    end
    if nbody >= 2 
        for ci in clusters
            for cj in clusters
                ci.idx < cj.idx || continue
                tmp_spaces = copy(p_spaces)
                tmp_spaces[ci.idx] = q_spaces[ci.idx]
                tmp_spaces[cj.idx] = q_spaces[cj.idx]
                FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, na))
            end
        end
    end
    if nbody >= 3 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    ci.idx < cj.idx || continue
                    cj.idx < ck.idx || continue
                    tmp_spaces = copy(p_spaces)
                    tmp_spaces[ci.idx] = q_spaces[ci.idx]
                    tmp_spaces[cj.idx] = q_spaces[cj.idx]
                    tmp_spaces[ck.idx] = q_spaces[ck.idx]
                    FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, na))
                end
            end
        end
    end
    if nbody >= 4 
        for ci in clusters
            for cj in clusters
                for ck in clusters
                    for cl in clusters
                        ci.idx < cj.idx || continue
                        cj.idx < ck.idx || continue
                        ck.idx < cl.idx || continue
                        tmp_spaces = copy(p_spaces)
                        tmp_spaces[ci.idx] = q_spaces[ci.idx]
                        tmp_spaces[cj.idx] = q_spaces[cj.idx]
                        tmp_spaces[ck.idx] = q_spaces[ck.idx]
                        tmp_spaces[cl.idx] = q_spaces[cl.idx]
                        FermiCG.add!(ci_vector, FermiCG.BSstate(clusters, tmp_spaces, na, nb))
                    end
                end
            end
        end
    end
    return ci_vector 
#=}}}=#
end

