using TimerOutputs
using BlockDavidson
using BenchmarkTools

"""
    build_full_H(ci_vector::TPSCIstate, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H(ci_vector::TPSCIstate, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    zero_fock = TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            ket_idx = 0
            for (fock_ket, configs_ket) in ci_vector.data
                fock_trans = fock_bra - fock_ket

                # check if transition is connected by H
                if haskey(clustered_ham, fock_trans) == false
                    ket_idx += length(configs_ket)
                    continue
                end

                for (config_ket, coeff_ket) in configs_ket
                    ket_idx += 1
                    ket_idx <= bra_idx || continue


                    for term in clustered_ham[fock_trans]
                    
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                       
                        me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                        H[bra_idx, ket_idx] += me 
                    end

                    H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                end
            end
        end
    end
    return H
end
#=}}}=#


"""
    build_full_H_parallel(ci_vector::TPSCIstate, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H_parallel( ci_vector_l::TPSCIstate{T,N,R}, ci_vector_r::TPSCIstate{T,N,R}, 
                                cluster_ops, clustered_ham::ClusteredOperator;
                                sym=false) where {T,N,R}
#={{{=#
    dim_l = length(ci_vector_l)
    dim_r = length(ci_vector_r)
    H = zeros(T, dim_l, dim_r)

    dim_l == dim_r || sym == false || error(" dim_l!=dim_r yet sym==true")

    if (dim_l == dim_r) && sym == false
        @warn(" are you missing sym=true?")
    end
    jobs = []

    zero_fock = TransferConfig([(0,0) for i in 1:N])
    bra_idx = 0

    for (fock_bra, configs_bra) in ci_vector_l.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            #push!(jobs, (bra_idx, fock_bra, config_bra) )
            #push!(jobs, (bra_idx, fock_bra, config_bra, H[bra_idx,:]) )
            push!(jobs, (bra_idx, fock_bra, config_bra, zeros(dim_r)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        Hrow = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector_r.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                ket_idx += 1
                ket_idx <= job[1] || sym == false || continue

                for term in clustered_ham[fock_trans]
                       
                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    
                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end

            end

        end
    end

    # because @threads divides evenly the loop, let's distribute thework more fairly
    #mid = length(jobs) ÷ 2
    #r = collect(1:length(jobs))
    #perm = [r[1:mid] reverse(r[mid+1:end])]'[:]
    #jobs = jobs[perm]
    
    #for job in jobs
    Threads.@threads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        H[job[1],:] .= job[4]
    end

    if sym
        for i in 1:dim_l
            @simd for j in i+1:dim_l
                @inbounds H[i,j] = H[j,i]
            end
        end
    end


    return H
end
#=}}}=#


"""
    function tps_ci_direct( ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        H_old    = nothing,
                        v_old    = nothing,
                        verbose   = 0) where {T,N,R}

# Solve for eigenvectors/values in the basis defined by `ci_vector`. Use direct diagonalization. 

If updating existing matrix, pass in H_old/v_old to avoid rebuilding that block
# Arguments
- `solver`: Which solver to use. Options = ["davidson", "krylovkit"]
"""
function tps_ci_direct( ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        conv_thresh = 1e-5,
                        lindep_thresh = 1e-12,
                        max_ss_vecs = 12,
                        max_iter    = 40,
                        shift       = nothing,
                        precond     = false,
                        H_old    = nothing,
                        v_old    = nothing,
                        verbose   = 0,
                        solver = "davidson") where {T,N,R}
    #={{{=#
    println()
    @printf(" |== Tensor Product State CI =======================================\n")
    vec_out = deepcopy(ci_vector)
    e0 = zeros(T,R)
    @printf(" Hamiltonian matrix dimension = %5i: \n", length(ci_vector))
    dim = length(ci_vector)
    flush(stdout)
   
    precond == false || @warn("davidson preconditioning NYI")

    H = zeros(T, 1,1)

    if H_old !== nothing
        v_old !== nothing || error(" can't specify H_old w/out v_old")
        v_tot = deepcopy(ci_vector)
        v_new = deepcopy(ci_vector)
        
        project_out!(v_new, v_old)
        
        #v_tot = copy(v_old)
        #add!(v_tot, v_new)

        dim_old = length(v_old)
        dim_new = length(v_new)
            

        # create indexing to find old indices in new space
        indices = OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, Int}}()
   
        idx = 1
        for (fock,configs) in v_tot.data
            indices[fock] = OrderedDict{ClusterConfig{N}, Int}()
            for (config,coeff) in configs
                indices[fock][config] = idx
                idx += 1
            end
        end

        dim = dim_old + dim_new

        dim == length(v_tot) || error(" not adding up", dim_old, " ", dim_new, " ", length(v_tot))

        H = zeros(T, dim, dim)


        # add old H elements
        @printf(" %-50s", "Fill old/old Hamiltonian: ")
        flush(stdout)
        @time _fill_H_block!(H, H_old, v_old, v_old, indices)

        @printf(" %-50s", "Build old/new Hamiltonian matrix with dimension: ")
        flush(stdout)
        @time Htmp = build_full_H_parallel(v_old, v_new, cluster_ops, clustered_ham)
        _fill_H_block!(H, Htmp, v_old, v_new, indices)
        _fill_H_block!(H, Htmp', v_new, v_old, indices)

        @printf(" %-50s", "Build new/new Hamiltonian matrix with dimension: ")
        flush(stdout)
        @time Htmp = build_full_H_parallel(v_new, v_new, cluster_ops, clustered_ham, sym=true)
        _fill_H_block!(H, Htmp, v_new, v_new, indices)
        
        vec_out = deepcopy(v_tot)
    else
        @printf(" %-50s", "Build full Hamiltonian matrix with dimension: ")
        @time H = build_full_H_parallel(ci_vector, ci_vector, cluster_ops, clustered_ham, sym=true)
    end
        
        

    @printf(" Now diagonalize\n")
    flush(stdout)
    if length(vec_out) > 500
    
        if solver == "krylovkit"
            time = @elapsed e0,v, info = KrylovKit.eigsolve(H, R, :SR, 
                                                            verbosity=  verbose, 
                                                            maxiter=    max_iter, 
                                                            #krylovdim=20, 
                                                            issymmetric=true, 
                                                            ishermitian=true, 
                                                            tol=        conv_thresh)
            println()
            println(info)
            println()
            @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
            v = hcat(v[1:R]...)

        elseif solver == "arpack"
            time = @elapsed e0,v = Arpack.eigs(H, nev = R, which=:SR)
        
        elseif solver == "davidson"
            davidson = Davidson(H, v0=get_vector(ci_vector), 
                                        max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, tol=conv_thresh, lindep_thresh=lindep_thresh)
            # time = @elapsed e0,v = BlockDavidson.eigs(davidson);
            time = @elapsed e0,v = BlockDavidson.eigs(davidson, Adiag=diag(H), precond_start_thresh=1e-1);
        end
        @printf(" %-50s", "Diagonalization time: ")
        @printf("%10.6f seconds\n",time)
        if verbose > 0
            display(info)
        end
    else
        time = @elapsed F = eigen(H)
        e0 = F.values[1:R]
        v = F.vectors[:,1:R]
        @printf(" %-50s", "Diagonalization time: ")
        @printf("%10.6f seconds\n",time)
    end
    set_vector!(vec_out, v)

    clustered_S2 = extract_S2(ci_vector.clusters, T=T)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    #@timeit to "<S2>" s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e0[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e0, vec_out, H 
end
#=}}}=#

function _fill_H_block!(H_big, H_small, v_l,v_r, indices)
    #={{{=#
    # Fill H_big with elements from H_small
    idx_l = 1
    
    idx_l = zeros(Int,length(v_l))
    idx_r = zeros(Int,length(v_r))

    idx = 1
    for (fock,configs) in v_l.data
        for (config,coeff) in configs
            idx_l[idx] = indices[fock][config]
            idx += 1
        end
    end

    idx = 1
    for (fock,configs) in v_r.data
        for (config,coeff) in configs
            idx_r[idx] = indices[fock][config]
            idx += 1
        end
    end

    for (il,iil) in enumerate(idx_l)
        for (ir,iir) in enumerate(idx_r)
            H_big[iil,iir] = H_small[il,ir]
        end
    end
#    for (fock_l,configs_l) in v_l.data
#        for (config_l,coeff_l) in configs_l
#            idx_l_tot = indices[fock_l][config_l]
#
#            idx_r = 1
#            for (fock_r,configs_r) in v_r.data
#                for (config_r,coeff_r) in configs_r
#                    idx_r_tot = indices[fock_r][config_r]
#
#                    H_big[idx_l_tot, idx_r_tot] = H_small[idx_l, idx_r]
#
#                    idx_r += 1
#                end
#            end
#
#            idx_l += 1
#        end
#    end
end
#=}}}=#


"""
    tps_ci_davidson(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Solve for eigenvectors/values in the basis defined by `ci_vector`. Use iterative davidson solver. 
"""
function tps_ci_davidson(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        v0 = nothing,
                        conv_thresh = 1e-5,
                        max_ss_vecs = 12,
                        max_iter    = 40,
                        shift       = nothing,
                        precond     = false,
                        verbose     = 0) where {T,N,R}
    #={{{=#
    println()
    @printf(" |== Tensor Product State CI =======================================\n")
    vec_out = deepcopy(ci_vector)
    e0 = zeros(T,R) 
   
    dim = length(ci_vector)
    iters = 0

    
    function matvec(v::Vector) 
        iters += 1
        #in = deepcopy(ci_vector) 
        in = TPSCIstate(ci_vector, R=size(v,2))
        set_vector!(in, v)
        #sig = deepcopy(in)
        #zero!(sig)
        #build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)
        return tps_ci_matvec(in, cluster_ops, clustered_ham)[:,1]
    end
    function matvec(v::Matrix)
        iters += 1
        #in = deepcopy(ci_vector) 
        in = TPSCIstate(ci_vector, R=size(v,2))
        set_vector!(in, v)
        #sig = deepcopy(in)
        #zero!(sig)
        #build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)
        return tps_ci_matvec(in, cluster_ops, clustered_ham)
    end


    Hmap = LinOpMat{T}(matvec, dim, true)

    davidson = Davidson(Hmap, v0=get_vector(ci_vector), 
                                max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, tol=conv_thresh)

    #time = @elapsed e0,v = Arpack.eigs(Hmap, nev = R, which=:SR)
    #time = @elapsed e0,v, info = KrylovKit.eigsolve(Hmap, R, :SR, 
    #                                                verbosity=  verbose, 
    #                                                maxiter=    max_iter, 
    #                                                #krylovdim=20, 
    #                                                issymmetric=true, 
    #                                                ishermitian=true, 
    #                                                tol=        conv_thresh)

    e = nothing
    v = nothing
    if precond
        @printf(" %-50s", "Compute diagonal: ")
        clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = "Hcmf") 
        @time Hd = compute_diagonal(ci_vector, cluster_ops, clustered_ham_0)
        @printf(" %-50s", "Compute <0|H0|0>: ")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)[1]
        @time Eref = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)[1]
        Hd .+= Eref - E0
        @printf(" Now iterate: \n")
        flush(stdout)
        @time e,v = BlockDavidson.eigs(davidson, Adiag=Hd);
    else
        @time e,v = BlockDavidson.eigs(davidson);
    end
    set_vector!(vec_out, v)
    
    clustered_S2 = extract_S2(ci_vector.clusters)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e, vec_out 
end
#=}}}=#


"""
    tps_ci_matvec(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Compute the action of `clustered_ham` on `ci_vector`. 
"""
function tps_ci_matvec(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#

    jobs = []

    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra, coeff_bra, zeros(T,R)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        coeff_bra = job[4]
        sig_out = job[5]
    
        for (fock_trans, terms) in clustered_ham
            fock_ket = fock_bra - fock_trans

            haskey(ci_vector.data, fock_ket) || continue
            
            configs_ket = ci_vector[fock_ket]


            for (config_ket, coeff_ket) in configs_ket
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
    
                    #norm(term.ints)*maximum(abs.(coeff_ket)) > 1e-5 || continue
                    #@btime norm($term.ints)*maximum(abs.($coeff_ket)) > 1e-12 
                    

                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    @simd for r in 1:R
                        @inbounds sig_out[r] += me * coeff_ket[r]
                    end
                    #@btime $sig_out .+= $me .* $ci_vector[$fock_ket][$config_ket] 
                end

            end

        end
    end

    #for job in jobs
    Threads.@threads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    sigv = zeros(size(ci_vector))
    for job in jobs
        #for r in 1:R
        #    sigv[job[1],r] += job[5][r]
        #end
        sigv[job[1],:] .+= job[5]
    end

    return sigv
end
#=}}}=#



function print_tpsci_iter(ci_vector::TPSCIstate{T,N,R}, it, e0, converged) where {T,N,R}
#={{{=#
    if converged 
        @printf("*TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    else
        @printf(" TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    end
    @printf(" E(var): ")
    for i in 1:R
        @printf("%13.8f ", e0[i])
    end
#    @printf(" E(pt2): ")
#    for i in 1:R
#        @printf("%13.8f ", e2[i])
#    end
    println()
end
#=}}}=#

"""
    compute_expectation_value(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}

Compute expectation value of a `ClusteredOperator` (`clustered_ham`) for state `ci_vector`
"""
function compute_expectation_value(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}
    #={{{=#

    out = zeros(T,R)

    for (fock_bra, configs_bra) in ci_vector.data

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket

                    me = 0.0
                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                        me += contract_matrix_element(term, cluster_ops, 
                                                      fock_bra, config_bra, 
                                                      fock_ket, config_ket)
                    end

                    #out .+= coeff_bra .* coeff_ket .* me
                    for r in 1:R
                        out[r] += coeff_bra[r] * coeff_ket[r] * me
                    end

                end

            end
        end
    end

    return out 
end
#=}}}=#

"""
    function compute_expectation_value_parallel(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
"""
function compute_expectation_value_parallel(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#

    # 
    # This will be were we collect our results
    evals = zeros(T,R)

    jobs = []

    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            push!(jobs, (fock_bra, config_bra, coeff_bra, zeros(T,R)) )
        end
    end

    function _add_val!(eval_job, me, coeff_bra, coeff_ket)
        for ri in 1:R
            #for rj in ri:R
            #    @inbounds eval_job[ri,rj] += me * coeff_bra[ri] * coeff_ket[rj] 
            #    #eval_job[rj,ri] = eval_job[ri,rj]
            #end
            @inbounds eval_job[ri] += me * coeff_bra[ri] * coeff_ket[ri] 
        end
    end

    function do_job(job)
        fock_bra = job[1]
        config_bra = job[2]
        coeff_bra = job[3]
        eval_job = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                #ket_idx += 1
                #ket_idx <= job[1] || continue

                me = 0.0
                for term in clustered_ham[fock_trans]

                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                    me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    #Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end
                #
                # now add the results
                #@inbounds for ri in 1:R
                #    @simd for rj in ri:R
                _add_val!(eval_job, me, coeff_bra, coeff_ket)
                #for ri in 1:R
                #    for rj in ri:R
                #        eval_job[ri,rj] += me * coeff_bra[ri] * coeff_ket[rj] 
                #        #eval_job[rj,ri] = eval_job[ri,rj]
                #    end
                #end
            end
        end
    end

    #for job in jobs
    #Threads.@threads for job in jobs
    @qthreads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        evals .+= job[4]
    end

    return evals 
end
#=}}}=#

"""
    compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    Hd = zeros(size(vector)[1])
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
                try
                    Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
                catch
                    display(term)
                    display(fock_bra)
                    display(config_bra)
                    error()
                end

            end
        end
    end
    return Hd
end

"""
    compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}

Fast version, used for PT2
"""
function compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}
    Hd = zeros(T, size(vector)[1])
    compute_diagonal!(Hd, vector, cluster_ops, opstring)
    return Hd
end


"""
    compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}


Fast version, used for PT2, overwrites Hd data with diagonal.
"""
function compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}
    fill!(Hd,0.0)
    idx = 1
    for (fock, configs) in vector.data
        for c in vector.clusters
            mat = []
            try
                mat = diag(cluster_ops[c.idx][opstring][(fock[c.idx],fock[c.idx])])
            catch
                println(c, fock[c.idx])
                error()
            end
            
            idxc = idx + 0
            for (config, _) in configs
                Hd[idxc] += mat[config[c.idx]]
                idxc += 1
            end
        end
        idx += length(configs)
    end
    return Hd
end


"""
    compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
		    try
			    Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
		    catch
			    display(term)
			    display(fock_bra)
			    display(config_bra)
			    error()
		    end

            end
        end
    end
    return
end
#=}}}=#


"""
    expand_each_fock_space!(s::TPSCIstate{T,N,R}, bases::Vector{ClusterBasis}) where {T,N,R}

For each fock space sector defined, add all possible basis states
- `basis::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::TPSCIstate{T,N,R}, bases::Vector{ClusterBasis{A,T}}) where {T,N,R,A}
    # {{{
    println("\n Make each Fock-Block the full space")
    # create full space for each fock block defined
    for (fblock,configs) in s.data
        #println(fblock)
        dims::Vector{UnitRange{Int16}} = []
        #display(fblock)
        for c in s.clusters
            # get number of vectors for current fock space
            dim = size(bases[c.idx][fblock[c.idx]], 2)
            push!(dims, 1:dim)
        end
        for newconfig in Iterators.product(dims...)
            #display(newconfig)
            #println(typeof(newconfig))
            #
            # this is not ideal - need to find a way to directly create key
            config = ClusterConfig(collect(newconfig))
            s.data[fblock][config] = zeros(SVector{R,T}) 
            #s.data[fblock][[i for i in newconfig]] = 0
        end
    end
end
# }}}

"""
    expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis}, na, nb)

Define all possible fock space sectors and add all possible basis states
- `basis::Vector{ClusterBasis}` 
- `na`: Number of alpha electrons total
- `nb`: Number of alpha electrons total
"""
function expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis{A,T}}, na, nb) where {A,T}
    # {{{
    println("\n Expand to full space")
    ns = []

    for c in s.clusters
        nsi = []
        for (fspace,basis) in bases[c.idx]
            push!(nsi,fspace)
        end
        push!(ns,nsi)
    end
    for newfock in Iterators.product(ns...)
        nacurr = 0
        nbcurr = 0
        for c in newfock
            nacurr += c[1]
            nbcurr += c[2]
        end
        if (nacurr == na) && (nbcurr == nb)
            config = FockConfig(collect(newfock))
            add_fockconfig!(s,config) 
        end
    end
    expand_each_fock_space!(s,bases)

    return
end
# }}}




"""
    project_out!(v::TPSCIstate, w::TPSCIstate)

Project w out of v 
    |v'> = |v> - |w><w|v>
"""
function project_out!(v::TPSCIstate, w::TPSCIstate)
    for (fock,configs) in w.data 
        if haskey(v.data, fock)
            for (config, coeff) in configs
                if haskey(v.data[fock], config)
                    delete!(v.data[fock], config)
                end
            end
            if length(v[fock]) == 0
                delete!(v.data, fock)
            end
        end
    end
    # I'm not sure why this is necessary
    idx = 0
    for (fock,configs) in v.data
        for (config, coeffs) in v.data[fock]
            idx += 1
        end
    end
end



"""
    hosvd(ci_vector::TPSCIstate{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}

Peform HOSVD aka Tucker Decomposition of TPSCIstate
"""
function hosvd(ci_vector::TPSCIstate{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}
#={{{=#
   
    cluster_rotations = []
    for ci in ci_vector.clusters
        println()
        println(" --------------------------------------------------------")
        println(" Density matrix: Cluster ", ci.idx)
        println()
        println(" Compute BRDM")
        println(" Hshift = ",hshift)
        
        dims = Dict()
        for (fock, mat) in cluster_ops[ci.idx]["H"]
            fock[1] == fock[2] || error("?")
            dims[fock[1]] = size(mat,1)
        end
        
        rdms = build_brdm(ci_vector, ci, dims)
        norm = 0
        entropy = 0
        rotations = Dict{Tuple,Matrix{T}}() 
        for (fspace,rdm) in rdms
            fspace_norm = 0
            fspace_entropy = 0
            @printf(" Diagonalize RDM for Cluster %2i in Fock space: ",ci.idx)
            println(fspace)
            F = eigen(Symmetric(rdm))

            idx = sortperm(F.values, rev=true) 
            n = F.values[idx]
            U = F.vectors[:,idx]


            # Either truncate the unoccupied cluster states, or remix them with a hamiltonian to be unique
            if truncate < 0
                remix = []
                for ni in 1:length(n)
                    if n[ni] < 1e-8
                        push!(remix, ni)
                    end
                end
                U2 = U[:,remix]
                Hlocal = U2' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U2
                
                F = eigen(Symmetric(Hlocal))
                n2 = F.values
                U2 = U2 * F.vectors
                
                U[:,remix] .= U2[:,:]
            
            else
                keep = []
                for ni in 1:length(n) 
                    if abs(n[ni]) > truncate
                        push!(keep, ni)
                    end
                end
                @printf(" Truncated Tucker space. Starting: %5i Ending: %5i\n" ,length(n), length(keep))
                U = U[:,keep]
            end
        

           
            
            n = diag(U' * rdm * U)
            Elocal = diag(U' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U)
            
            norm += sum(n)
            fspace_norm = sum(n)
            @printf("                 %4s:    %12s    %12s\n", "","Population","Energy")
            for (ni_idx,ni) in enumerate(n)
                if abs(ni/norm) > 1e-16
                    fspace_entropy -= ni*log(ni/norm)/norm
                    entropy -=  ni*log(ni)
                    @printf("   Rotated State %4i:    %12.8f    %12.8f\n", ni_idx,ni,Elocal[ni_idx])
                end
           end
           @printf("   ----\n")
           @printf("   Entanglement entropy:  %12.8f\n" ,fspace_entropy) 
           @printf("   Norm:                  %12.8f\n" ,fspace_norm) 

           #
           # let's just be careful that our vectors remain orthogonal
           F = svd(U)
           U = F.U * F.Vt
           check_orthogonality(U) 
           rotations[fspace] = U
        end
        @printf(" Final entropy:.... %12.8f\n",entropy)
        @printf(" Final norm:....... %12.8f\n",norm)
        @printf(" --------------------------------------------------------\n")

        flush(stdout) 

        #ci.rotate_basis(rotations)
        #ci.check_basis_orthogonality()
        push!(cluster_rotations, rotations)
    end
    return cluster_rotations
end
#=}}}=#




"""
    build_brdm(ci_vector::TPSCIstate, ci, dims)
    
Build block reduced density matrix for `Cluster`,  `ci`
- `ci_vector::TPSCIstate` = input state
- `ci` = Cluster type for whihch we want the BRDM
- `dims` = list of dimensions for each fock sector
"""
function build_brdm(ci_vector::TPSCIstate, ci, dims)
    # {{{
    rdms = OrderedDict()
    for (fspace, configs) in ci_vector.data
        curr_dim = dims[fspace[ci.idx]]
        rdm = zeros(curr_dim,curr_dim)
        for (configi,coeffi) in configs
            for cj in 1:curr_dim

                configj = [configi...]
                configj[ci.idx] = cj
                configj = ClusterConfig(configj)

                if haskey(configs, configj)
                    rdm[configi[ci.idx],cj] += sum(coeffi.*configs[configj])
                end
            end
        end


        if haskey(rdms, fspace[ci.idx]) 
            rdms[fspace[ci.idx]] += rdm 
        else
            rdms[fspace[ci.idx]] = rdm 
        end

    end
    return rdms
end
# }}}



function dump_tpsci(filename::AbstractString, ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    @save filename ci_vector cluster_ops clustered_ham
end

#function load_tpsci(filename::AbstractString) 
#    a = @load filename
#    return eval.(a)
#end

