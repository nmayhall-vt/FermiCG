#### Not tested, probably doesn't work!
"""
    get_map(ci_vector, cluster_ops, clustered_ham)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ci_vector::TuckerState, cluster_ops, clustered_ham; shift = nothing)
    #={{{=#
    iters = 0
   
    dim = length(ci_vector)
    function mymatvec(v)
        iters += 1
        
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v, length(v), nr)
        elseif length(size(v)) == 2
            nr = size(v)[2]
        else
            error(" is tensor not unfolded?")
        end
    
      
        set_vector!(ci_vector, v)
        
        fold!(ci_vector)
        sig = deepcopy(ci_vector)
        zero!(sig)
        build_sigma!(sig, ci_vector, cluster_ops, clustered_ham)

        unfold!(ci_vector)
        
        sig = get_vector(sig)

        if shift != nothing
            # this is how we do CEPA
            sig += shift * get_vector(ci_vector)
        end

        return sig 
    end
    return LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#

function tucker_ci_solve!(ci_vector::TuckerState, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#
    unfold!(ci_vector) 
    Hmap = get_map(ci_vector, cluster_ops, clustered_ham)

    v0 = get_vector(ci_vector)
    nr = size(v0)[2] 
    
    davidson = Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=1e-5)
    #Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    e,v = FermiCG.solve(davidson)
    set_vector!(ci_vector,v)
    return e,v
end
#=}}}=#


"""
0 = <x|H - E0|x'>v(x') + <x|H - E0|p>v(p) 
0 = <x|H - E0|x'>v(x') + <x|H|p>v(p) 
A(x,x')v(x') = -H(x,p)v(p)

here, x is outside the reference space, and p is inside

Ax=b

works for one root at a time
"""
function tucker_cepa_solve!(ref_vector::TuckerState, ci_vector::TuckerState, cluster_ops, clustered_ham; tol=1e-5)
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
    get_foi(v::TuckerState, clustered_ham, q_spaces; nroots=1, nbody=2)
Compute the first-order interacting space as defined by clustered_ham

e.g., 
 1(p') 3(r) 4(q's)  *  v[(1,1),(1,1),(1,1),(1,1)][1:1,1:1,1:1,1:1]  =>  v[(2,1), (1,1), (0,1), (1,1)][1:N, 1:1, 1:N, 1:N]
"""
function get_foi(v::TuckerState, clustered_ham, q_spaces; nroots=1, nbody=2)
    println(" Prepare empty TuckerState spanning the FOI of input")#={{{=#
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
