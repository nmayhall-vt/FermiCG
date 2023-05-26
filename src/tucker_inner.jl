
"""
    build_sigma!(sigma_vector::BSTstate, ci_vector::BSTstate, cluster_ops, clustered_ham)
"""
function build_sigma_serial!(sigma_vector::BSTstate{T,N,R}, ci_vector::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                             nbody=4, cache=false) where {T,N,R}
    #={{{=# 

    for (fock_bra, configs_bra) in sigma_vector
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket


                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue
                    
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                        #coeff_bra.core .= form_sigma_block!(term, cluster_ops, fock_bra, config_bra,
                        #                                    fock_ket, config_ket,
                        #                                    coeff_bra, coeff_ket,
                        #                                    cache=cache)
                        tmp = form_sigma_block!(term, cluster_ops, fock_bra, config_bra,
                                                            fock_ket, config_ket,
                                                            coeff_bra, coeff_ket,
                                                            cache=cache)
                        for r in 1:R
                            coeff_bra.core[r] .= tmp[r]                        
                        end


                    end
                end
            end
        end
    end
    return
    #=}}}=#
end


"""
    build_sigma!(sigma_vector::BSTstate, ci_vector::BSTstate, cluster_ops, clustered_ham)
"""
function cache_hamiltonian_old(sigma_vector::BSTstate, ci_vector::BSTstate, cluster_ops, clustered_ham; nbody=4)
    #={{{=#
    


    println(" Cache hamiltonian terms")
    
    for (fock_bra, configs_bra) in sigma_vector
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket


                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue
                    
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                        
                        cache_key = OperatorConfig((fock_bra, fock_ket, config_bra, config_ket))
                        term.cache[cache_key] = build_dense_H_term(term, cluster_ops, 
                                                                   fock_bra, config_bra, coeff_bra, 
                                                                   fock_ket, config_ket, coeff_ket)

                    end
                end
            end
        end
    end
    return
    #=}}}=#
end

function cache_hamiltonian(bra::BSTstate{T,N,R}, ket::BSTstate{T,N,R}, cluster_ops, clustered_ham; nbody=4, verbose=0, blas=false) where {T,N,R}
#={{{=#
    
    # it seems like this is quite a bit faster when turned off:
    #if blas
    #    TensorOperations.enable_blas()
    #else
    #    TensorOperations.disable_blas()
    #end

    keys_to_loop = [keys(clustered_ham.trans)...]
    
    # set up scratch arrays
    nscr = 10 
    scr_f = Vector{Vector{Vector{T}} }()
    for tid in 1:Threads.nthreads()
        tmp = Vector{Vector{T}}() 
        [push!(tmp, zeros(T,100000)) for i in 1:nscr]
        push!(scr_f, tmp)
    end
   
    
    if verbose>0
        @printf(" %-50s", " Number of threaded jobs:")
        println(length(keys_to_loop))
    end
    
    Threads.@threads for ftrans in keys_to_loop
        scr = scr_f[Threads.threadid()]
        terms = clustered_ham[ftrans]
        for term in terms
               
            length(term.clusters) <= nbody || continue

            for (fock_ket, configs_ket) in ket
                #fock_bra = [fock_ket.config...]
                #for (cii,ci) in enumerate(term.clusters)
                #    fock_bra[ci.idx] = (fock_ket[ci.idx][1] + ftrans[cii][1], fock_ket[ci.idx][2] + ftrans[cii][2])
                #end
            
                fock_bra = ftrans + fock_ket
                #fock_bra = FockConfig(fock_bra)

                haskey(bra.data, fock_bra) == true || continue

                for (config_ket, tuck_ket) in configs_ket
                    for (config_bra, tuck_bra) in bra[fock_bra]
                        
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                        cache_key = OperatorConfig((fock_bra, fock_ket, config_bra, config_ket))
                        #if term isa ClusteredTerm4B
                        #    @btime op = build_dense_H_term($term, $cluster_ops, $fock_bra, $config_bra, $tuck_bra, 
                        #                                   $fock_ket, $config_ket, $tuck_ket, $scr)
                        #    error("please stop")
                        #end

                        term.cache[cache_key] = build_dense_H_term(term, cluster_ops, 
                                                                   fock_bra, config_bra, tuck_bra, 
                                                                   fock_ket, config_ket, tuck_ket, 
                                                                   scr)
                        #term.cache[cache_key] = build_dense_H_term(term, cluster_ops, 
                        #                                           fock_bra, config_bra, tuck_bra, 
                        #                                           fock_ket, config_ket, tuck_ket)
                    end 
                end
            end
        end
    end
end
#=}}}=#

"""
    build_sigma_parallel!(sigma_vector::BSTstate, ci_vector::BSTstate, cluster_ops, clustered_ham)
"""
function build_sigma!(sigma_vector::BSTstate{T,N,R}, ci_vector::BSTstate{T,N,R}, cluster_ops, clustered_ham; nbody=4, cache=false, verbose=1) where {T,N,R}
    #={{{=#

    verbose < 2 || @printf(" in build_sigma!")
    verbose < 2 || println(" length of sigma vector: ", length(sigma_vector))
    flush(stdout)
    jobs = []
    output = [[] for i in 1:Threads.nthreads()]
    for (fock_bra, configs_bra) in sigma_vector
        for (config_bra, tuck_bra) in configs_bra
            push!(jobs, [fock_bra, config_bra])
        end
    end
    
    # set up scratch arrays
    nscr = 10 
    scr_f = Vector{Vector{Vector{T}} }()
    for tid in 1:Threads.nthreads()
        tmp = Vector{Vector{T}}() 
        [push!(tmp, zeros(T,1000)) for i in 1:nscr]
        push!(scr_f, tmp)
    end
   
    function do_job(job)
        
        fock_bra = job[1]
        config_bra = job[2]
        coeff_bra = sigma_vector[fock_bra][config_bra]
        
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_ket, coeff_ket) in configs_ket


                for term in clustered_ham[fock_trans]

                    length(term.clusters) <= nbody || continue

                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                    # these methods dispatched on type of term
                    #coeff_bra.core .= form_sigma_block!(term, cluster_ops, fock_bra, config_bra,
                    #                              fock_ket, config_ket,
                    #                              coeff_bra, coeff_ket,
                    #                              cache=cache)
                    out = form_sigma_block!(term, cluster_ops, fock_bra, config_bra,
                                                  fock_ket, config_ket,
                                                  coeff_bra, coeff_ket,
                                                  scr_f[Threads.threadid()],
                                                  cache=cache)

                    push!(output[Threads.threadid()], (fock_bra, config_bra, out))


                end
            end
        end
    end
   
    Threads.@threads for job in jobs
    #for job in jobs
        do_job(job)
    end

    flush(stdout)

    for tid in output
        for out in tid
            fock_bra = out[1]
            config_bra = out[2]
            core = out[3]
            add!(sigma_vector[fock_bra][config_bra].core, core)
        end
    end
    return
    #=}}}=#
end


#
# form_sigma_block computes the action of the term on a Tucker compressed state, 
# projected into the space defined by bra. This is used to work with H within a subspace defined by a compression
#
#
function form_sigma_block!(term::C,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R},
                            scr_f::Vector{Vector{T}};
                            cache=false ) where {T,N,R, C<:ClusteredTerm}
    #={{{=#
    check_term(term, fock_bra, bra, fock_ket, ket) || throw(Exception) 
    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    op = Array{T}[]
    cache_key = OperatorConfig((fock_bra, fock_ket, bra, ket))
    #if cache && haskey(term.cache, cache_key)
    if cache 
       

        #
        # read the dense H term
        op = term.cache[cache_key]
    
    else

        #cache == false || println(" couldn't find:", cache_key)

        #
        # build the dense H term
        op = build_dense_H_term(term, cluster_ops, fock_bra, bra, coeffs_bra, fock_ket, ket, coeffs_ket, scr_f)
        #if term isa ClusteredTerm4B
        #    @btime op = build_dense_H_term($term, $cluster_ops, $fock_bra, $bra, $coeffs_bra, $fock_ket, $ket, $coeffs_ket, $scr_f)
        #    error("please stop")
        #end
        #if cache
        #    term.cache[cache_key] = op
        #end
    end

    #if term isa ClusteredTerm2B
    #    display(term)
    #    @btime contract_dense_H_with_state($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    @btime contract_dense_H_with_state_tensor($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    @btime contract_dense_H_with_state_ncon($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    #error("please stop")
    #end
    return contract_dense_H_with_state(term, op, state_sign, coeffs_bra, coeffs_ket)
    #return contract_dense_H_with_state_tensor(term, op, state_sign, coeffs_bra, coeffs_ket)
    #return contract_dense_H_with_state_ncon(term, op, state_sign, coeffs_bra, coeffs_ket)
end
#=}}}=#


#
# form_sigma_block computes the action of the term on a Tucker compressed state, 
# projected into the space defined by bra. This is used to work with H within a subspace defined by a compression
#
#
function form_sigma_block!(term::C,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R};
                            cache=false ) where {T,N,R, C<:ClusteredTerm}
    #={{{=#
    check_term(term, fock_bra, bra, fock_ket, ket) || throw(Exception) 
    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    op = Array{T}[]
    cache_key = OperatorConfig((fock_bra, fock_ket, bra, ket))
    #if cache && haskey(term.cache, cache_key)
    if cache 
       

        #
        # read the dense H term
        op = term.cache[cache_key]
    
    else

        #cache == false || println(" couldn't find:", cache_key)

        #
        # build the dense H term
        op = build_dense_H_term(term, cluster_ops, fock_bra, bra, coeffs_bra, fock_ket, ket, coeffs_ket)
        #if term isa ClusteredTerm4B
        #    @btime op = build_dense_H_term($term, $cluster_ops, $fock_bra, $bra, $coeffs_bra, $fock_ket, $ket, $coeffs_ket, $scr_f)
        #    error("please stop")
        #end
        #if cache
        #    term.cache[cache_key] = op
        #end
    end

    #if term isa ClusteredTerm2B
    #    display(term)
    #    @btime contract_dense_H_with_state($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    @btime contract_dense_H_with_state_tensor($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    @btime contract_dense_H_with_state_ncon($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    #error("please stop")
    #end
    return contract_dense_H_with_state(term, op, state_sign, coeffs_bra, coeffs_ket)
    #return contract_dense_H_with_state_tensor(term, op, state_sign, coeffs_bra, coeffs_ket)
    #return contract_dense_H_with_state_ncon(term, op, state_sign, coeffs_bra, coeffs_ket)
end
#=}}}=#









function _compress_local_operator(gamma, Ul::Matrix{T}, Ur::Matrix{T}) where T
# this is way slower than @tensor
#={{{=#
    # gamma has 3 indices (orbital indices, cluster indices (left), cluster indices (right)

    #
    # out(i,jp) = gamma(p,I,J) Ul(I,i)
    out = Ul' * reshape(permutedims(gamma, [2,3,1]), size(gamma,2), size(gamma,3)*size(gamma,1))

    #
    # out(j,pi) = out(J,pi) Ur(J,j)
    out = Ur' * reshape(out', size(gamma,3), size(gamma,1)*size(Ul,2)) 
    
    # out(j,pi) -> out(p,i,j)
    return reshape(out', size(gamma,1), size(Ul,2), size(Ur,2))


#    # out(i,pJ) = gamma(I,pJ) U(I,i)
#    out = Ul' * unfold(gamma, 2) 
#    # out(ip,J) 
#    out = reshape(out, size(out,1) * size(gamma,1), size(gamma,3))
#    
#    # out(ip,j) = gamma(ip,J) U(J,j)
#    out = out * Ur
#    # out(i,p,j) 
#    out = reshape(out, size(Ul,2), size(gamma,1), size(Ur,2))
#
#    # out(p,i,j)
#    return permutedims(out, [2,1,3])
end
#=}}}=#

function calc_bound(term::ClusteredTerm1B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            prescreen=1e-4) where {T,N,R}
    c1 = term.clusters[1]
    
    bound1 = norm(term.ints)*norm(coeffs_ket.core)
    if bound1 < prescreen
        return false 
    end
    return true
end
   
function calc_bound(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            prescreen=1e-4) where {T,N,R}
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    #@views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    bound1 = norm(term.ints)*norm(coeffs_ket.core)
    #bound1 *= norm(gamma1)*norm(gamma2)
    bound1 *= norm(coeffs_ket.factors[c1.idx])*norm(coeffs_ket.factors[c2.idx])
    if bound1 < prescreen
        return false 
    end
    return true
end
   
function calc_bound(term::ClusteredTerm3B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            prescreen=1e-4) where {T,N,R}
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    bound1 = norm(term.ints)*norm(coeffs_ket.core)
    #bound1 *= norm(gamma1)*norm(gamma2)*norm(gamma3)
    bound1 *= norm(coeffs_ket.factors[c1.idx])*norm(coeffs_ket.factors[c2.idx])*norm(coeffs_ket.factors[c3.idx])
    if bound1 < prescreen
        return false 
    end
    return true
end
   
   
function calc_bound(term::ClusteredTerm4B,
                            cluster_ops::Vector{ClusterOps{T}},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            prescreen=1e-4) where {T,N,R}
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]
    
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
    bound1 = norm(term.ints)*norm(coeffs_ket.core)
    #bound1 *= norm(gamma1)*norm(gamma2)*norm(gamma3)*norm(gamma4)
    bound1 *= norm(coeffs_ket.factors[c1.idx])*norm(coeffs_ket.factors[c2.idx])*norm(coeffs_ket.factors[c3.idx])*norm(coeffs_ket.factors[c4.idx])
    if bound1 < prescreen
        return false 
    end
    return true
end
  
function Base.copyto!(a::Tuple{Array{T,N}}, b) where {T,N}
    length(a) == length(b) || throw(DimensionMismatch)
    for i in 1:length(a)
        a[i] .= b[i]
    end
end
