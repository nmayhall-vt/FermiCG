using Profile
using BenchmarkTools

"""
    build_sigma!(sigma_vector::BSTstate, ci_vector::BSTstate, cluster_ops, clustered_ham)
"""
function build_sigma_serial!(sigma_vector::BSTstate, ci_vector::BSTstate, cluster_ops, clustered_ham; nbody=4, cache=false)
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

                        coeff_bra.core .= form_sigma_block!(term, cluster_ops, fock_bra, config_bra,
                                                            fock_ket, config_ket,
                                                            coeff_bra, coeff_ket,
                                                            cache=cache)


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

function cache_hamiltonian(bra::BSTstate{T,N,R}, ket::BSTstate{T,N,R}, cluster_ops, clustered_ham; nbody=4, verbose=0) where {T,N,R}
#={{{=#
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
                                                                   scr_f[Threads.threadid()])
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
function build_sigma!(sigma_vector::BSTstate{T,N,R}, ci_vector::BSTstate{T,N,R}, cluster_ops, clustered_ham; nbody=4, cache=false) where {T,N,R}
    #={{{=#

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
        [push!(tmp, zeros(T,100000)) for i in 1:nscr]
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
                            cluster_ops::Vector{ClusterOps},
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

    op = Array{Float64}[]
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
    #    @btime contract_dense_H_with_state($term, $op, $state_sign, $coeffs_bra, $coeffs_ket)
    #    error("please stop")
    #end
    return contract_dense_H_with_state(term, op, state_sign, coeffs_bra, coeffs_ket)
end
#=}}}=#





"""
    Contract integrals and ClusterOps to form dense 4-body Hamiltonian matrix (tensor) in Tucker basis
"""
function build_dense_H_term(term::ClusteredTerm1B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker,
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    op = Array{Float64,2}[]
        
    op1::Array{Float64,2} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]

    #
    # Get 1body operator and compress it using the cluster's Tucker factors
    op = coeffs_bra.factors[c1.idx]' * (op1[bra[c1.idx],ket[c1.idx]] * coeffs_ket.factors[c1.idx])

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm2B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker, 
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1m::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2m::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    @views gamma1 = gamma1m[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = gamma2m[:,bra[c2.idx],ket[c2.idx]]
    #@btime @views gamma1::Array{Float64,3} = $cluster_ops[$c1.idx][$term.ops[1]][($fock_bra[$c1.idx],$fock_ket[$c1.idx])][:,$bra[$c1.idx],$ket[$c1.idx]]
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]

    tmp = scr_f[1]
    g1  = scr_f[2]
    g2  = scr_f[3]

    resize!(tmp, size(Ul,2) * size(gamma1,1) * size(gamma1,3))
    resize!(g1, size(Ul,2) * size(Ur,2) * size(gamma1,1))
    
    tmp = reshape2(tmp, (size(gamma1,1), size(Ul,2), size(gamma1,3)))
    g1 = reshape2(g1, (size(gamma1,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp[p,k,J] = Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] = Ur[J,l] * tmp[p,k,J]
    end

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
    
    tmp = scr_f[1]
    resize!(tmp, size(Ul,2) * size(gamma2,1) * size(gamma2,3))
    resize!(g2, size(Ul,2) * size(Ur,2) * size(gamma2,1))
    
    tmp = reshape2(tmp, (size(gamma2,1), size(Ul,2), size(gamma2,3)))
    g2 = reshape2(g2, (size(gamma2,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp[p,k,J] = Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] = Ur[J,l] * tmp[p,k,J]
    end

    tmp2 = scr_f[4]
    resize!(tmp2, size(term.ints,2) * size(g1,3) * size(g1,2) )
    tmp2 = reshape2(tmp2, (size(term.ints,2), size(g1,3), size(g1,2)))
    
    @tensor begin
        tmp2[q,J,I] = term.ints[p,q] * g1[p,I,J]
        op[J,L,I,K] := tmp2[q,J,I] * g2[q,K,L]
    end

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm3B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker,
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1m::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2m::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3m::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    @views gamma1 = gamma1m[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = gamma2m[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = gamma3m[:,bra[c3.idx],ket[c3.idx]]
    
    Ul = coeffs_bra.factors[c1.idx]
    Ur = coeffs_ket.factors[c1.idx]
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    g1  = scr_f[3]
    g2  = scr_f[4]
    g3  = scr_f[5]

    resize!(tmp1, size(Ul,2) * size(gamma1,1) * size(gamma1,3))
    resize!(g1, size(Ul,2) * size(Ur,2) * size(gamma1,1))
    
    tmp1 = reshape2(tmp1, (size(gamma1,1), size(Ul,2), size(gamma1,3)))
    g1 = reshape2(g1, (size(gamma1,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp1[p,k,J] = Ul[I,k] * gamma1[p,I,J]
        g1[p,k,l] = Ur[J,l] * tmp1[p,k,J]
    end

    Ul = coeffs_bra.factors[c2.idx]
    Ur = coeffs_ket.factors[c2.idx]
    
    tmp1 = scr_f[1]
    resize!(tmp1, size(gamma2,1) * size(Ul,2) * size(gamma2,3))
    resize!(g2, size(Ul,2) * size(Ur,2) * size(gamma2,1))
    
    tmp1 = reshape2(tmp1, (size(gamma2,1), size(Ul,2), size(gamma2,3)))
    g2 = reshape2(g2, (size(gamma2,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp1[p,k,J] := Ul[I,k] * gamma2[p,I,J]
        g2[p,k,l] := Ur[J,l] * tmp1[p,k,J]
    end
    #display(("g1/2", size(g1), size(g2)))

    Ul = coeffs_bra.factors[c3.idx]
    Ur = coeffs_ket.factors[c3.idx]
    
    tmp1 = scr_f[1]
    resize!(tmp1, size(Ul,2) * size(gamma3,1) * size(gamma3,3))
    resize!(g3, size(Ul,2) * size(Ur,2) * size(gamma3,1))
    
    tmp1 = reshape2(tmp1, (size(gamma3,1), size(Ul,2), size(gamma3,3)))
    g3 = reshape2(g3, (size(gamma3,1), size(Ul,2), size(Ur,2)))
    
    @tensor begin
        tmp1[p,k,J] = Ul[I,k] * gamma3[p,I,J]
        g3[p,k,l] = Ur[J,l] * tmp1[p,k,J]
    end

    #
    # Now contract into 3body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    resize!(tmp1, size(term.ints,2) * size(term.ints,3) * size(g1,2) * size(g1,3))
    resize!(tmp2, size(term.ints,3) * size(g1,2) * size(g1,3) * size(g2,2) * size(g2,3))
    
    tmp1 = reshape2(tmp1, (size(term.ints,2), size(term.ints,3), size(g1,2), size(g1,3)))
    tmp2 = reshape2(tmp2, (size(term.ints,3), size(g1,2), size(g1,3), size(g2,2), size(g2,3)))
    @tensor begin
        tmp1[q,r,I,J] = term.ints[p,q,r] * g1[p,I,J]
        tmp2[r,I,J,K,L] = tmp1[q,r,I,J] * g2[q,K,L]
        op[J,L,N,I,K,M] := tmp2[r,I,J,K,L] * g3[r,M,N]
    end

    return op
end
#=}}}=#
function build_dense_H_term(term::ClusteredTerm4B, cluster_ops, fock_bra, bra, coeffs_bra::Tucker, fock_ket, ket, coeffs_ket::Tucker,
                            scr_f::Vector{Vector{T}}) where T
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    #
    # Compress Gammas using the cluster's Tucker factors
    # e.g.,
    #   Gamma(pqr, I, J) Ul(I,k) Ur(J,l) = Gamma(pqr, k, l) where k and l are compressed indices
    gamma1m::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    gamma2m::Array{Float64,3} = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])]
    gamma3m::Array{Float64,3} = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])]
    gamma4m::Array{Float64,3} = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])]
    @views gamma1 = gamma1m[:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = gamma2m[:,bra[c2.idx],ket[c2.idx]]
    @views gamma3 = gamma3m[:,bra[c3.idx],ket[c3.idx]]
    @views gamma4 = gamma4m[:,bra[c4.idx],ket[c4.idx]]



    Ul1 = coeffs_bra.factors[c1.idx]
    Ul2 = coeffs_bra.factors[c2.idx]
    Ul3 = coeffs_bra.factors[c3.idx]
    Ul4 = coeffs_bra.factors[c4.idx]
    
    Ur1 = coeffs_ket.factors[c1.idx]
    Ur2 = coeffs_ket.factors[c2.idx]
    Ur3 = coeffs_ket.factors[c3.idx]
    Ur4 = coeffs_ket.factors[c4.idx]
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    tmp3 = scr_f[3]
    tmp4 = scr_f[4]
    g1  = scr_f[5]
    g2  = scr_f[6]
    g3  = scr_f[7]
    g4  = scr_f[8]

    resize!(tmp1, size(Ul1,2) * size(gamma1,1) * size(gamma1,3))
    resize!(tmp2, size(Ul2,2) * size(gamma2,1) * size(gamma2,3))
    resize!(tmp3, size(Ul3,2) * size(gamma3,1) * size(gamma3,3))
    resize!(tmp4, size(Ul4,2) * size(gamma4,1) * size(gamma4,3))

    resize!(g1, size(Ul1,2) * size(Ur1,2) * size(gamma1,1))
    resize!(g2, size(Ul2,2) * size(Ur2,2) * size(gamma2,1))
    resize!(g3, size(Ul3,2) * size(Ur3,2) * size(gamma3,1))
    resize!(g4, size(Ul4,2) * size(Ur4,2) * size(gamma4,1))
    
    tmp1 = reshape2(tmp1, (size(gamma1,1), size(Ul1,2), size(gamma1,3)))
    tmp2 = reshape2(tmp2, (size(gamma2,1), size(Ul2,2), size(gamma2,3)))
    tmp3 = reshape2(tmp3, (size(gamma3,1), size(Ul3,2), size(gamma3,3)))
    tmp4 = reshape2(tmp4, (size(gamma4,1), size(Ul4,2), size(gamma4,3)))
    
    g1 = reshape2(g1, (size(gamma1,1), size(Ul1,2), size(Ur1,2)))
    g2 = reshape2(g2, (size(gamma2,1), size(Ul2,2), size(Ur2,2)))
    g3 = reshape2(g3, (size(gamma3,1), size(Ul3,2), size(Ur3,2)))
    g4 = reshape2(g4, (size(gamma4,1), size(Ul4,2), size(Ur4,2)))


    @tensor begin
        tmp1[p,k,J] = Ul1[I,k] * gamma1[p,I,J]
        g1[p,k,l]   = Ur1[J,l] * tmp1[p,k,J]
    end
    @tensor begin
        tmp2[p,k,J] = Ul2[I,k] * gamma2[p,I,J]
        g2[p,k,l]   = Ur2[J,l] * tmp2[p,k,J]
    end
    @tensor begin
        tmp3[p,k,J] = Ul3[I,k] * gamma3[p,I,J]
        g3[p,k,l]   = Ur3[J,l] * tmp3[p,k,J]
    end
    @tensor begin
        tmp4[p,k,J] = Ul4[I,k] * gamma4[p,I,J]
        g4[p,k,l]   = Ur4[J,l] * tmp4[p,k,J]
    end


    #
    # Now contract into 4body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    
    tmp1 = scr_f[1]
    tmp2 = scr_f[2]
    tmp3 = scr_f[3]
    
    resize!(tmp1, size(term.ints,2) * size(term.ints,3) * size(term.ints,4) * size(g1,3) * size(g1,2))
    resize!(tmp2, size(term.ints,3) * size(term.ints,4) * size(g1,3) * size(g2,3) * size(g1,2) * size(g2,2))
    resize!(tmp3, size(term.ints,4) * size(g1,3) * size(g2,3) * size(g3,3) * size(g1,2) * size(g2,2) * size(g3,2))
    
    tmp1 = reshape2(tmp1, (size(term.ints,2), size(term.ints,3), size(term.ints,4), size(g1,3), size(g1,2)))
    tmp2 = reshape2(tmp2, (size(term.ints,3), size(term.ints,4), size(g1,3), size(g2,3), size(g1,2), size(g2,2)))
    tmp3 = reshape2(tmp3, (size(term.ints,4), size(g1,3), size(g2,3), size(g3,3), size(g1,2), size(g2,2), size(g3,2)))
    
    @tensor begin
        tmp1[q,r,s,J,I] = term.ints[p,q,r,s] * g1[p,I,J]
        tmp2[r,s,J,L,I,K] = tmp1[q,r,s,J,I] * g2[q,K,L]
        tmp3[s,J,L,N,I,K,M] = tmp2[r,s,J,L,I,K] * g3[r,M,N]
        op[J,L,N,P,I,K,M,O] := tmp3[s,J,L,N,I,K,M] * g4[s,O,P]
    end

    return op
end
#=}}}=#

"""
    Contract  Hamiltonian matrix (tensor) in Tucker basis with trial vector (Tucker)
"""
function contract_dense_H_with_state(term::ClusteredTerm1B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]

    n_clusters = N 
    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    #
    # todo: PRECOMPUTE THIS!
    overlaps = Dict{Int,Matrix{T}}()
    s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        # if it is a scalar, then just update the sign with the value
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    perm,_ = bubble_sort(indices)

    #coeffs_ket2 = copy(coeffs_ket.core)

    #
    # multiply by overlaps first if the bra side is smaller,
    # otherwise multiply by Hamiltonian term first

    coeffs_bra2_out = [coeffs_bra.core...]
    #
    # Transpose to get contig data for blas (explicit copy?)
    for r in 1:R
        coeffs_ket2  = transform_basis(coeffs_ket.core[r], overlaps, trans=true)

        coeffs_ket2 = permutedims(coeffs_ket2 , perm)
        coeffs_bra2 = permutedims(coeffs_bra.core[r], perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)
       
        coeffs_ket2 = reshape(coeffs_ket2, dim1[1], prod(dim1[2:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1], prod(dim2[2:end]))

        #
        # Reshape Hamiltonian term operator
        # ... not needed for 1b term

        #
        # Multiply
        coeffs_bra2 += s .* (op * coeffs_ket2)

        # now untranspose
        perm2,_ = bubble_sort(perm)
        coeffs_bra2 = reshape(coeffs_bra2, dim2)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

#        #
#        # multiply by overlaps now if the bra side is larger,
#        if length(coeffs_bra2[r]) >= length(coeffs_ket2[r])
#            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
#        end
        coeffs_bra2_out[r] = coeffs_bra2
    end

    return ntuple(r->coeffs_bra2_out[r], R)
end
#=}}}=#
function contract_dense_H_with_state(term::ClusteredTerm2B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    n_clusters = N 
    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    overlaps = Dict{Int,Matrix{T}}()
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    indices[c2.idx] = 0
    perm,_ = bubble_sort(indices)

    coeffs_bra2_out = [coeffs_bra.core...]

    #
    # Reshape Hamiltonian term operator
    op2 = reshape(op, prod(size(op)[1:2]), prod(size(op)[3:4]))

    for r in 1:R
        coeffs_bra2 = deepcopy(coeffs_bra.core[r])
        coeffs_ket2 = deepcopy(coeffs_ket.core[r])

        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(coeffs_bra2) < length(coeffs_ket2)
            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
        end
        coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        coeffs_ket2 = permutedims(coeffs_ket2, perm)
        coeffs_bra2 = permutedims(coeffs_bra2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)
        
        coeffs_ket2 = reshape(coeffs_ket2, dim1[1]*dim1[2], prod(dim1[3:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1]*dim2[2], prod(dim2[3:end]))

        coeffs_bra2 .+= s .* (op2' * coeffs_ket2)

        coeffs_bra2 = reshape(coeffs_bra2, dim2)

        # now untranspose
        perm2,_ = bubble_sort(perm)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

        coeffs_bra2_out[r] = coeffs_bra2
    end
    return ntuple(r->coeffs_bra2_out[r], R)
end
#=}}}=#
function contract_dense_H_with_state(term::ClusteredTerm3B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    n_clusters = N 
    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    overlaps = Dict{Int,Matrix{T}}()
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    indices[c2.idx] = 0
    indices[c3.idx] = 0
    perm,_ = bubble_sort(indices)

    coeffs_bra2_out = [coeffs_bra.core...]

    #
    # Reshape Hamiltonian term operator
    op2 = reshape(op, prod(size(op)[1:3]), prod(size(op)[4:6]))

    for r in 1:R
        coeffs_bra2 = deepcopy(coeffs_bra.core[r])
        coeffs_ket2 = deepcopy(coeffs_ket.core[r])

        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(coeffs_bra2) < length(coeffs_ket2)
            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
        end
        coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        coeffs_ket2 = permutedims(coeffs_ket2, perm)
        coeffs_bra2 = permutedims(coeffs_bra2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)

        coeffs_ket2 = reshape(coeffs_ket2, dim1[1]*dim1[2]*dim1[3], prod(dim1[4:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1]*dim2[2]*dim2[3], prod(dim2[4:end]))

        #
        # Multiply
        coeffs_bra2 .+= s .* (op2' * coeffs_ket2)

        # now untranspose
        perm2,_ = bubble_sort(perm)
        coeffs_bra2 = reshape(coeffs_bra2, dim2)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

        coeffs_bra2_out[r] = coeffs_bra2
    end

    return ntuple(r->coeffs_bra2_out[r], R)
end
#=}}}=#
function contract_dense_H_with_state(term::ClusteredTerm4B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
    #={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    n_clusters = N 

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    overlaps = Dict{Int,Matrix{T}}()
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue
        ci != c4.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    indices[c2.idx] = 0
    indices[c3.idx] = 0
    indices[c4.idx] = 0
    perm,_ = bubble_sort(indices)

    coeffs_bra2_out = [coeffs_bra.core...]

    #
    # Reshape Hamiltonian term operator
    op2 = reshape(op, prod(size(op)[1:4]), prod(size(op)[5:8]))

    for r in 1:R
        coeffs_bra2 = deepcopy(coeffs_bra.core[r])
        coeffs_ket2 = deepcopy(coeffs_ket.core[r])


        #
        # multiply by overlaps first if the bra side is smaller,
        # otherwise multiply by Hamiltonian term first
        if length(coeffs_bra2) < length(coeffs_ket2)
            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
        end
        coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)

        #
        # Transpose to get contig data for blas (explicit copy?)
        coeffs_ket2 = permutedims(coeffs_ket2, perm)
        coeffs_bra2 = permutedims(coeffs_bra2, perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)

        coeffs_ket2 = reshape(coeffs_ket2, dim1[1]*dim1[2]*dim1[3]*dim1[4], prod(dim1[5:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1]*dim2[2]*dim2[3]*dim2[4], prod(dim2[5:end]))

        #
        # Multiply
        coeffs_bra2 .+= s .* (op2' * coeffs_ket2)

        # now untranspose
        perm2,_ = bubble_sort(perm)
        coeffs_bra2 = reshape(coeffs_bra2, dim2)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

        coeffs_bra2_out[r] = coeffs_bra2
    end
    return ntuple(r->coeffs_bra2_out[r], R)
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

#
# form_sigma_block_expand computes the action of the term on a Tucker compressed state, 
# NOT projected into the space defined by bra, but rather prescreen each operator defined by prescreen. 
# This is used to find the first order interactiong space (FOIS) from a Tucker compressed state.  
#

function form_sigma_block_expand(term::ClusteredTerm1B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
#={{{=#
    #display(term)
    c1 = term.clusters[1]
    n_clusters = length(bra)

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    #
    # Get 1body operator and compress it using the cluster's Tucker factors,
    # but since we are expanding the compression space
    # only compress the right hand side
    op1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    op =  (op1[bra[c1.idx],ket[c1.idx]] * coeffs_ket.factors[c1.idx])

    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [-c1.idx, c1.idx]
        state_indices[c1.idx] = c1.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
    push!(indices, state_indices)

    output_size = [size(coeffs_ket.core[1])...]
    output_size[c1.idx] = size(op,1)
    
    # loop over global states
    bra_cores = ntuple(r->zeros(T,output_size...), R)
    for r in 1:R

        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        if length(tensors) == 1
            # this means that all the overlaps and the operator is a scalar
            bra_cores[r] .= coeffs_ket.core[r] .* s
        else
            bra_cores[r] .= @ncon(tensors, indices)
            bra_cores[r] .= bra_cores[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    new_factors[c1.idx] = Matrix(1.0I, size(bra_cores[1],c1.idx), size(bra_cores[1],c1.idx)) 
    return Tucker(bra_cores, NTuple{N}(new_factors))
end
#=}}}=#

function form_sigma_block_expand(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
    #={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    n_clusters = length(bra)
    #bound1 = norm(term.ints)*norm(coeffs_ket.core)
    ##bound1 *= norm(gamma1)*norm(gamma2)
    ##bound1 *= norm(coeffs_ket.factors[c1.idx])*norm(coeffs_ket.factors[c2.idx])
    #if bound1 < sqrt(prescreen)
    #    return FermiCG.Tucker(Array{Float64}(undef,[0 for i in 1:N]...), ntuple(i->Array{Float64}(undef,0,0), N))
    #end

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
   


    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end
    #@printf("Norm: %12.8f %12.8f %12.8f\n",norm(gamma1),norm(coeffs_ket.core),norm(gamma1)*norm(coeffs_ket.core))

    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over

    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))



    D = permutedims(g1, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > 1e-12 
            nkeep += 1
        end
    end
    new_factor1 = F.U[:,1:nkeep]
    g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
    g1 = permutedims(g1, [2,1,3])


    D = permutedims(g2, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > 1e-12
            nkeep += 1
        end
    end
    new_factor2 = F.U[:,1:nkeep]
    g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
    g2 = permutedims(g2, [2,1,3])


    #
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,J,I] := term.ints[p,q] * g1[p,I,J]
        op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
    end

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, -c1.idx, -c2.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
    push!(indices, state_indices)

    output_size = [size(coeffs_ket.core[1])...]
    output_size[c1.idx] = size(op,3)
    output_size[c2.idx] = size(op,4)
    
    # loop over global states
    bra_cores = ntuple(r->zeros(T,output_size...), R)
    for r in 1:R

        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        #bra_core = zeros(1,1)
        #bra_core = zeros(1,1)
        if length(tensors) == 1
            # this means that all the overlaps and the operator is a scalar
            bra_cores[r] .= coeffs_ket.core[r] .* s
        else
            bra_cores[r] .= @ncon(tensors, indices)
            bra_cores[r] .= bra_cores[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    #new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
    #new_factors[c2.idx] = Matrix(1.0I, size(bra_core,c2.idx), size(bra_core,c2.idx))
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    return Tucker(bra_cores, NTuple{N}(new_factors))
end
#=}}}=#

function form_sigma_block_expand(term::ClusteredTerm3B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
    #={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    n_clusters = length(bra)

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    Ur = coeffs_ket.factors[c3.idx]
    @tensor begin
        g3[p,I,l] := Ur[J,l] * gamma3[p,I,J]
    end

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over

    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))
    new_factor3 = Matrix(1.0I, size(g3,2), size(g3,2))



    D = permutedims(g1, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor1 = F.U[:,1:nkeep]
    g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
    g1 = permutedims(g1, [2,1,3])


    D = permutedims(g2, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor2 = F.U[:,1:nkeep]
    g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
    g2 = permutedims(g2, [2,1,3])


    D = permutedims(g3, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    # 
    # for now, let's just keep the full space, then maybe later start threshing
    new_factor3 = F.U[:,1:nkeep]
    g3 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g3 = reshape(g3, size(g3,1), size(D,2), size(D,3))
    g3 = permutedims(g3, [2,1,3])

    #
    # Now contract into 3body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,r,I,J] := term.ints[p,q,r] * g1[p,I,J]
    end
    @tensor begin
        op[r,I,J,K,L] := op[q,r,I,J] * g2[q,K,L]
    end
    @tensor begin
        op[J,L,N,I,K,M] := op[r,I,J,K,L] * g3[r,M,N]
    end

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    #tensors = Vector{NTuple{R,Array{T}} }()
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, c3.idx, -c1.idx, -c2.idx, -c3.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
    push!(indices, state_indices)

    output_size = [size(coeffs_ket.core[1])...]
    output_size[c1.idx] = size(op,4)
    output_size[c2.idx] = size(op,5)
    output_size[c3.idx] = size(op,6)
    
    # loop over global states
    bra_cores = ntuple(r->zeros(T,output_size...), R)
    for r in 1:R

        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        #bra_core = zeros(1,1)
        #bra_core = zeros(1,1)
        if length(tensors) == 1
            # this means that all the overlaps and the operator is a scalar
            bra_cores[r] .= coeffs_ket.core[r] .* s
        else
            #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", size.(overlaps), indices))
            #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", overlaps, indices))
            bra_cores[r] .= @ncon(tensors, indices)
            bra_cores[r] .= bra_cores[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    new_factors[c3.idx] = new_factor3 
    return Tucker(bra_cores, NTuple{N}(new_factors))

end
#=}}}=#

function form_sigma_block_expand(term::ClusteredTerm4B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
#={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]
    n_clusters = length(bra)

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
    fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)
    fock_bra[c4.idx] == fock_ket[c4.idx] .+ term.delta[4] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)

    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>
    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    @views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    Ur = coeffs_ket.factors[c1.idx]
    @tensor begin
        g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    @views gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
    Ur = coeffs_ket.factors[c3.idx]
    @tensor begin
        g3[p,I,l] := Ur[J,l] * gamma3[p,I,J]
    end

    @views gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]
    Ur = coeffs_ket.factors[c4.idx]
    @tensor begin
        g4[p,I,l] := Ur[J,l] * gamma4[p,I,J]
    end
    
    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over
    
    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))
    new_factor3 = Matrix(1.0I, size(g3,2), size(g3,2))
    new_factor4 = Matrix(1.0I, size(g4,2), size(g4,2))

   
    D = permutedims(g1, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor1 = F.U[:,1:nkeep]
    g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
    g1 = permutedims(g1, [2,1,3])


    D = permutedims(g2, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor2 = F.U[:,1:nkeep]
    g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
    g2 = permutedims(g2, [2,1,3])


    D = permutedims(g3, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor3 = F.U[:,1:nkeep]
    g3 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g3 = reshape(g3, size(g3,1), size(D,2), size(D,3))
    g3 = permutedims(g3, [2,1,3])


    D = permutedims(g4, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor4 = F.U[:,1:nkeep]
    g4 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g4 = reshape(g4, size(g4,1), size(D,2), size(D,3))
    g4 = permutedims(g4, [2,1,3])

    #
    # Now contract into 4body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,r,s,J,I] := term.ints[p,q,r,s] * g1[p,I,J]
        op[r,s,J,L,I,K] := op[q,r,s,J,I] * g2[q,K,L]
        op[s,J,L,N,I,K,M] := op[r,s,J,L,I,K] * g3[r,M,N]
        op[J,L,N,P,I,K,M,O] := op[s,J,L,N,I,K,M] * g4[s,O,P]
    end

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, c3.idx, c4.idx, -c1.idx, -c2.idx, -c3.idx, -c4.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        state_indices[c4.idx] = c4.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end

    push!(indices, state_indices)

    output_size = [size(coeffs_ket.core[1])...]
    output_size[c1.idx] = size(op,5)
    output_size[c2.idx] = size(op,6)
    output_size[c3.idx] = size(op,7)
    output_size[c4.idx] = size(op,8)
    
    # loop over global states
    bra_cores = ntuple(r->zeros(T,output_size...), R)
    for r in 1:R

        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        if length(tensors) == 1
            # this means that all the overlaps and the operator is a scalar
            bra_cores[r] .= coeffs_ket.core[r] .* s
        else
            bra_cores[r] .= @ncon(tensors, indices)
            bra_cores[r] .= bra_cores[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    new_factors[c3.idx] = new_factor3 
    new_factors[c4.idx] = new_factor4 
    return Tucker(bra_cores, NTuple{N}(new_factors))
    
end
#=}}}=#

function form_sigma_block_expand2(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R},
                            scr::Vector{Vector{Float64}};  
                            max_number=nothing, prescreen=1e-4) where {T,N,R}
    #={{{=#
    #display(term)
    #display.((fock_bra, fock_ket))
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    n_clusters = length(bra)

    #
    # make sure active clusters are correct transitions
    fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
    fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)

    #
    # determine sign from rearranging clusters if odd number of operators
    state_sign = compute_terms_state_sign(term, fock_ket)
    
    scr1 = scr[1]
    scr2 = scr[2]
    scr3 = scr[3]
    scr4 = scr[4]

    np = size(term.ints,1)
    nq = size(term.ints,2)
    
    #
    # op[IK,JL] = <I|p'|J> h(pq) <K|q|L>

    # todo: add in 2e integral tucker decomposition and compress gamma along 1st index first

    #
    # Compress Gammas using the cluster's Tucker factors, but since we are expanding the compression space
    # only compress the right hand side
    # e.g.,
    #   Gamma(pqr, I, J) Ur(J,l) = Gamma(pqr, I, l) where k and l are compressed indices
    g1a::Array{Float64,3} = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])]
    #@btime @views gamma1 = $g1a[:,$bra[$c1.idx],$ket[$c1.idx]]
    #@views gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
    @views gamma1 = g1a[:,bra[c1.idx],ket[c1.idx]]
    Ur = coeffs_ket.factors[c1.idx]

    # gamma1(p,I,J) U(J,j)  = g1(p,I,j)
    nI = size(gamma1, 2) 
    nJ = size(gamma1, 3) 
    nj = size(Ur, 2) 

    #resize!(scr1, np*nI*nj)
    #scr1 = reshape2(scr1, (np*nI,nj))
    #mul!(scr1, reshape2(gamma1,(np*nI,nJ)), Ur)
    #g1 = reshape2(scr1, (np,nI,nj))

    g1 = scr[1]
    resize!(g1, np*nI*nj)
    g1 = reshape2(g1, (np,nI,nj))
    @tensor begin
        g1[p,I,l] = Ur[J,l] * gamma1[p,I,J]
        #g1[p,I,l] := Ur[J,l] * gamma1[p,I,J]
    end

    @views gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
    Ur = coeffs_ket.factors[c2.idx]
    @tensor begin
        g2[p,I,l] := Ur[J,l] * gamma2[p,I,J]
    end

    #
    # Decompose the local operators. Since gamma[p,I,l] has indices (small, large, small),
    # we only need at most p*l number of new vectors for the index we are searching over

    new_factor1 = Matrix(1.0I, size(g1,2), size(g1,2))
    new_factor2 = Matrix(1.0I, size(g2,2), size(g2,2))



    D = permutedims(g1, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor1 = F.U[:,1:nkeep]
    g1 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g1 = reshape(g1, size(g1,1), size(D,2), size(D,3))
    g1 = permutedims(g1, [2,1,3])


    D = permutedims(g2, [2,1,3])
    F = svd(reshape(D, size(D,1), size(D,2)*size(D,3)))
    nkeep = 0
    for si in F.S
        if si > prescreen
            nkeep += 1
        end
    end
    new_factor2 = F.U[:,1:nkeep]
    g2 = Diagonal(F.S[1:nkeep]) * F.Vt[1:nkeep,:] 
    g2 = reshape(g2, size(g2,1), size(D,2), size(D,3))
    g2 = permutedims(g2, [2,1,3])

    #
    # Now contract into 2body term
    #
    # h(p,q) * g1(p,I,J) * g2(q,K,L) = op(J,L,I,K)
    op = Array{Float64}[]
    @tensor begin
        op[q,J,I] := term.ints[p,q] * g1[p,I,J]
        op[J,L,I,K] := op[q,J,I] * g2[q,K,L]
    end

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int16}}()
    state_indices = -collect(1:n_clusters)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, -c1.idx, -c2.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end

    push!(tensors, coeffs_ket.core)
    push!(indices, state_indices)

    length(tensors) == length(indices) || error(" mismatch between operators and indices")

    bra_core = zeros(1,1)
    if length(tensors) == 1
        # this means that all the overlaps and the operator is a scalar
        bra_core = coeffs_ket.core .* s
    else
        #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", size.(overlaps), indices))
        #display.(("a", size(coeffs_bra), size(coeffs_ket), "sizes: ", overlaps, indices))
        bra_core = @ncon(tensors, indices)
        bra_core .= bra_core .* s
    end

    # first decompose the already partially decomposed core tensor
    #
    # Vket ~ Aket x U1 x U2 x ...
    #
    # then we modified the compressed coefficients in the ket, Aket
    #
    # to get Abra, which we then compress again.
    #
    # The active cluster tucker factors become identity matrices
    #
    # Abra ~ Bbra x UU1 x UU2 x ....
    #
    #
    # e.g, V(IJK) = C(ijk) * U1(Ii) * U2(Jj) * U3(Kk)
    #
    # then C get's modified and furhter compressed
    #
    # V(IJK) = C(abc) * U1(ia) * U2(jb) * U3(kc) * U1(Ii) * U2(Jj) * U3(Kk)
    # V(IJK) = C(abc) * (U1(ia) * U1(Ii)) * (U2(jb) * U2(Jj)) * (U3(kc) * U3(Kk))
    # V(IJK) = C(abc) * U1(Ia)  * U2(Jb) * U3(Kc)
    #

    new_factors = [coeffs_ket.factors[i] for i in 1:N]
    #new_factors[c1.idx] = Matrix(1.0I, size(bra_core,c1.idx), size(bra_core,c1.idx))
    #new_factors[c2.idx] = Matrix(1.0I, size(bra_core,c2.idx), size(bra_core,c2.idx))
    new_factors[c1.idx] = new_factor1
    new_factors[c2.idx] = new_factor2 
    return Tucker(bra_core, NTuple{N}(new_factors))
end
#=}}}=#

function calc_bound(term::ClusteredTerm1B,
                            cluster_ops::Vector{ClusterOps},
                            fock_bra::FockConfig, bra::TuckerConfig,
                            fock_ket::FockConfig, ket::TuckerConfig,
                            coeffs_ket::Tucker{T,N,R};
                            prescreen=1e-4) where {T,N,R}
    c1 = term.clusters[1]
    
    bound1 = norm(term.ints)*norm(coeffs_ket.core)
    if bound1 < sqrt(prescreen)
        return false 
    end
    return true
end
   
function calc_bound(term::ClusteredTerm2B,
                            cluster_ops::Vector{ClusterOps},
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
    if bound1 < sqrt(prescreen)
        return false 
    end
    return true
end
   
function calc_bound(term::ClusteredTerm3B,
                            cluster_ops::Vector{ClusterOps},
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
    if bound1 < sqrt(prescreen)
        return false 
    end
    return true
end
   
   
function calc_bound(term::ClusteredTerm4B,
                            cluster_ops::Vector{ClusterOps},
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
    if bound1 < sqrt(prescreen)
        return false 
    end
    return true
end
   
