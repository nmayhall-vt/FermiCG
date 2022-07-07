
"""
    function add_single_excitons(v::BSstate{T,N,R}) where {T,N,R}

Add a Q space to all currently defined `TuckerConfigs`.
Return new `BSstate`
"""
function add_single_excitons!(v::BSstate{T,N,R}) where {T,N,R}
#={{{=#
    unfold!(v)
    for (fspace,tconfigs) in v.data
        for (tconfig,coeffs) in [tconfigs...]
            for ci in 1:N
                config_i = replace(tconfig, [ci], [v.q_spaces[ci][fspace[ci]]])
                v[fspace][config_i] = zeros(T,dim(config_i),R) 
            end
        end

    end
    return 
end
#=}}}=#

"""
    function add_1electron_transfers(v::BSstate{T,N,R}) where {T,N,R}
"""
function add_1electron_transfers!(v::BSstate{T,N,R}) where {T,N,R}
#={{{=#
    unfold!(v)
    for (fspace,tconfigs) in [v.data...]

        #alpha transfer
        for ci in 1:N
            for cj in 1:N
                ci != cj || continue 

                fconfig_ij = replace(fspace, [ci,cj], [(fspace[ci][1]+1, fspace[ci][2]), 
                                                       (fspace[ci][1]-1, fspace[ci][2])])


                tconf = Vector{UnitRange{Int16}}()
                for (ck,fk) in enumerate(fconfig_ij.config)
                    if haskey(v.p_spaces[ck], fk)
                        push!(tconf,v.p_spaces[ck][fk])
                    else
                        push!(tconf,v.q_spaces[ck][fk])
                    end
                end
                tconf = TuckerConfig(tconf)

                add_fockconfig!(v, fconfig_ij)

                v[fconfig_ij][tconf] = zeros(T,dim(tconf),R) 
            end
        end

        #beta transfer
        for ci in 1:N
            for cj in 1:N
                ci != cj || continue 

                fconfig_ij = replace(fspace, [ci,cj], [(fspace[ci][1], fspace[ci][2]+1), 
                                                       (fspace[ci][1], fspace[ci][2]-1)])


                tconf = Vector{UnitRange{Int16}}()
                for (ck,fk) in enumerate(fconfig_ij.config)
                    if haskey(v.p_spaces[ck], fk)
                        push!(tconf,v.p_spaces[ck][fk])
                    else
                        push!(tconf,v.q_spaces[ck][fk])
                    end
                end
                tconf = TuckerConfig(tconf)

                add_fockconfig!(v, fconfig_ij)

                v[fconfig_ij][tconf] = zeros(T,dim(tconf),R) 
            end
        end
    end
    return 
end
#=}}}=#




"""
    ci_solve(ci_vector::BSstate{T,N,R}, cluster_ops, clustered_ham; 
                         conv_thresh = 1e-5,
                         max_ss_vecs = 12,
                         max_iter    = 40,
                         shift       = nothing,
                         precond     = false,
                         verbose     = 0,
                         solver      = "davidson") where {T,N,R}

Solve for ground state in the space spanned by `ci_vector`'s basis 
# Arguments
- `conv_thresh`: residual convergence threshold
- `max_ss_vecs`: max number of subspace vectors
- `max_iter`: Max iterations in solver
- `shift`:  Use a shift? this is for CEPA type 
- `precond`: use preconditioner? Only applied to Davidson and not yet working,
- `verbose`: print level
- `solver`: Which solver to use. Options = ["davidson", "krylovkit"]
"""
function ci_solve(ci_vector::BSstate{T,N,R}, cluster_ops, clustered_ham; 
                         conv_thresh = 1e-5,
                         max_ss_vecs = 12,
                         max_iter    = 40,
                         shift       = nothing,
                         precond     = false,
                         verbose     = 0,
                         solver      = "davidson") where {T,N,R}
#={{{=#

    @printf(" |== BS CI =========================================================\n")
    @printf(" Solve CI with # variables = %i\n", length(ci_vector))
    vec = deepcopy(ci_vector)
    orthonormalize!(vec)
    #flush term cache
    flush_cache(clustered_ham)
    dim = length(ci_vector)
    iters = 0
    cache=true
   
    function matvec(v::Matrix{T}) where T
        iters += 1
        
        in = BSstate(ci_vector, R=size(v,2))
        
        unfold!(in)
        set_vector!(in, v)
        
        out = deepcopy(in)
        zero!(out)
        build_sigma!(out, in, cluster_ops, clustered_ham)
        #build_sigma!(out, in, cluster_ops, clustered_ham, cache=cache)
        unfold!(out)

        flush(stdout)
        return get_vector(out)
    end
    function matvec(v::Vector{T}) where T
        iters += 1
        
        in = BSstate(ci_vector, R=1)
        
        unfold!(in)
        set_vector!(in, v)
        
        out = deepcopy(in)
        zero!(out)
        build_sigma!(out, in, cluster_ops, clustered_ham)
        #build_sigma!(out, in, cluster_ops, clustered_ham, cache=cache)
        unfold!(out)

        flush(stdout)
        return get_vector(out)[:,1]
    end

    

    v0 = get_vector(vec)
    nr = size(v0)[2]
    
    Hmap = FermiCG.LinOpMat{T}(matvec, dim, true)
    
    vguess = get_vector(vec, 1)[:,1]
    e = [] 
    v = [[]]
    
    if solver == "krylovkit"

        time = @elapsed e, v, info = KrylovKit.eigsolve(Hmap, vguess, R, :SR, 
                                                        verbosity   = verbose, 
                                                        maxiter     = max_iter, 
                                                        #krylovdim   = max_ss_vecs, 
                                                        issymmetric = true, 
                                                        ishermitian = true, 
                                                        eager       = true,
                                                        tol         = conv_thresh)

        @printf(" Number of matvecs performed: %5i\n", info.numops)
        @printf(" Number of subspace restarts: %5i\n", info.numiter)
        if info.converged >= R
            @printf(" CI Converged: %5i roots\n", info.converged)
        end
        println(" Residual Norms")
        for r in 1:R
            @printf(" State %5i %16.1e\n", r, info.normres[r])
        end
        
        println()
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
        set_vector!(vec,hcat(v[1:R]...))
    elseif solver == "davidson"

        davidson = Davidson(Hmap,v0=v0,max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, tol=conv_thresh)
        flush(stdout)
        time = @elapsed e,v = FermiCG.solve(davidson, iprint=verbose)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
        #println(" Memory used by cache: ", mem_used_by_cache(clustered_ham))
        set_vector!(vec,v)

    else
        error(" Bad value for `solver`")
    end
   

#    cache=true
#    if cache
#        #@timeit to "cache" cache_hamiltonian(vec, vec, cluster_ops, clustered_ham)
#        @printf(" Build and cache each hamiltonian term in the current basis:\n")
#        flush(stdout)
#        @time cache_hamiltonian(vec, vec, cluster_ops, clustered_ham)
#        @printf(" done.\n")
#        flush(stdout)
#    end

    #for (ftrans,terms) in clustered_ham
    #    for term in terms
    #        println("nick: ", length(term.cache))
    #    end
    #end

    #cache_hamiltonian(ci_vector, ci_vector, cluster_ops, clustered_ham)
    #println(" Memory used by cache: ", mem_used_by_cache(clustered_ham))

    #flush term cache
    flush_cache(clustered_ham)
    
    clustered_S2 = extract_S2(vec.clusters)
    @printf(" %-50s", "Compute <S^2>: ")
    flush(stdout)
    tmp = deepcopy(vec)
    zero!(tmp)
    @time build_sigma!(tmp, vec, cluster_ops, clustered_S2)
    s2 = dot(tmp,vec)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e[r], abs(s2[r]))
    end

    @printf(" ==================================================================|\n")
    return e,vec
end
#=}}}=#
















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

