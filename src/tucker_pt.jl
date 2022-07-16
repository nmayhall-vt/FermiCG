using StatProfilerHTML
"""

 | PHP  PHQ ||PC| = |PC| E 
 | QHP  QHQ ||QC| = |QC| 

 PHPC + PHQC = PCe
 QHPC + QHQC = QCe

 (QHQ-e)*QC = -QHP*PC
 (X-P)H(X-P)C - e(X-P)C = -(X-P)HP*PC
 
 QC = XC - PC

 XHX*XC - PHX*XC - XHP*PC + PHP*PC - e*XP + e*PC = -XHP*PC + PHP*PC
 
 
 XHX*XC - PHX*XC - e*XP -e*PC =

 
 (QFQ-e0)*QC = -QHP*PC
 (X-P)F(X-P)C - e0*XC + e0*PC = -XHP*PC + e0*PC

 (X-P)F(X-P)C - e0*XC = -XHP*PC
 XFX*XC - PFX*XC - XFP*PC + PFP*PC - e0*XC = -XHP*PC

 XFX*XC - PFX*XC - e0*XC = -XHP*PC - PFP*PC + XFP*PC 

 (XFX - PFX - e0)*XC = -XHP*PC - PFP*PC + XFP*PC 


 """




 """
    function hylleraas_compressed_mp2(sig_in::BSTstate{T,N,R}, ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                                  H0 = "Hcmf", 
                                  tol=1e-6,   
                                  nbody=4, 
                                  max_iter=100, 
                                  verbose=1, 
                                  thresh=1e-8) where {T,N,R}

- `H0`: ["H", "Hcmf"] 

Compute compressed PT2.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |1 | = |I   + Sax| |1 | E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa + Hax*Cx = (1 + Sax*Cx)E
Hxa + HxxCx = SxaE + CxE

(Fxx-Eref-<0|F|0>)*Cx = Sxa*Eref - Hxa

Ax=b

After solving, the Energy can be obtained as:
E = (Eref + Hax*Cx) / (1 + Sax*Cx)
"""
function hylleraas_compressed_mp2(sig_in::BSTstate{T,N,R}, ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                                  H0 = "Hcmf", 
                                  tol=1e-6,   
                                  nbody=4, 
                                  max_iter=100, 
                                  verbose=1, 
                                  thresh=1e-8) where {T,N,R}
#={{{=#
    
#
            

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    # 
    # get <X|H|0>
    #sig = compress(sig_in, thresh=thresh)
    sig = deepcopy(sig_in)
    @printf(" %-50s%10i\n", "Length of input      FOIS: ", length(sig_in))
    #@printf(" %-50s%10i\n", "Length of compressed FOIS: ", length(sig))
    #project_out!(sig, ref)
    zero!(sig)
            
    @printf(" %-50s", "Build exact <X|V|0>: ")
    @time build_sigma!(sig, ref, cluster_ops, clustered_ham)
    
    # b = <X|H|0> 
    b = -get_vector(sig)
    
    
    # (H0 - E0) |1> = X H |0>

    e2 =zeros(T,R) 
   
    # 
    # get E_ref = <0|H|0>
    tmp = deepcopy(ref)
    zero!(tmp)
    build_sigma!(tmp, ref, cluster_ops, clustered_ham)
    e_ref = orth_dot(ref, tmp)

    # 
    # get E0 = <0|H0|0>
    tmp = deepcopy(ref)
    zero!(tmp)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time build_sigma!(tmp, ref, cluster_ops, clustered_ham_0)
    e0 = orth_dot(ref,tmp)
    
    if verbose > 0 
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, e_ref[r], e0[r])
        end
    end
  
    
    # 
    # get <X|F|0>
    tmp = deepcopy(sig)
    zero!(tmp)
    @printf(" %-50s", "Compute <X|F|0>: ")
    @time build_sigma!(tmp, ref, cluster_ops, clustered_ham_0)

    # b = - <X|H|0> + <X|F|0> = -<X|V|0>
    b .+= get_vector(tmp)
    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(sig)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        if haskey(ref, fock)
            for (tconfig, tuck) in tconfigs
                if haskey(ref[fock], tconfig)
                    ref_tuck = ref[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = Vector{Matrix{T}}() 
                    for i in 1:N
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    for r in 1:R
                        Sx[fock][tconfig].core[r] .= transform_basis(ref_tuck.core[r], overlaps)
                    end
                end
            end
        end
    end

    #@printf(" Norm of b         : %18.12f\n", sum(b.*b))
    flush_cache(clustered_ham_0)
    @printf(" %-50s", "Cache zeroth-order Hamiltonian: ")
    @time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0)
    psi1 = deepcopy(sig)

    #
    # Currently, we need to solve each root separately, this should be fixed
    # by writing our own CG solver
    for r in 1:R
        
        function mymatvec(x)

            xr = BSTstate(sig, R=1)
            xl = BSTstate(sig, R=1)

            #display(size(xr))
            #display(size(x))
            length(xr) .== length(x) || throw(DimensionMismatch)
            set_vector!(xr,x, root=1)
            zero!(xl)
            build_sigma!(xl, xr, cluster_ops, clustered_ham_0, cache=true)

            # subtract off -E0|1>
            #
            
            scale!(xr,-e0[1])
            #scale!(xr,-e0[r])  # pretty sure this should be uncommented - but it diverges, not sure why
            orth_add!(xl,xr)
            flush(stdout)

            return get_vector(xl)
        end
        br = b[:,r] .+ get_vector(Sx)[:,r] .* (e_ref[r] - e0[r])


        dim = length(br)
        Axx = LinearMap(mymatvec, dim, dim)


        #@time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0, nbody=1)

        #todo:  setting initial value to zero only makes sense when our reference space is projected out. 
        #       if it's not, then we want to add the reference state components |guess> += |ref><ref|guess>
        #
        x_vector = zeros(T,dim)
        x_vector = get_vector(sig)[:,r]*.1
        time = @elapsed x, solver = cg!(x_vector, Axx, br, log=true, maxiter=max_iter, verbose=false, abstol=tol)
        @printf(" %-50s%10.6f seconds\n", "Time to solve for PT1 with conjugate gradient: ", time)
    
        set_vector!(psi1, x_vector, root=r)
    end
        
    flush_cache(clustered_ham_0)
    
    SxC = orth_dot(Sx,psi1)
    #@printf(" %-50s%10.2f\n", "<A|X>C(X): ", SxC)
    #@printf(" <A|X>C(X) = %12.8f\n", SxC)
   
    tmp = deepcopy(ref)
    zero!(tmp)
    @printf(" %-50s", "Compute <0|H|1>: ")
    @time build_sigma!(tmp,psi1, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(tmp,ref)
    #@printf(" <1|1> = %12.8f\n", orth_dot(psi1,psi1))
    #@printf(" <0|H|1> = %12.8f\n", ecorr)
   
    e_pt2 = zeros(T,R)
    for r in 1:R
        e_pt2[r] = (e_ref[r] + ecorr[r])/(1+SxC[r])
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", e_pt2[r]-e_ref[r])
    end
    for r in 1:R
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2): ", e_pt2[r])
    end

    return psi1, e_pt2 

end#=}}}=#





"""
    function do_fois_pt2(ref::BSTstate, cluster_ops, clustered_ham;
            H0          = "Hcmf",
            max_iter    = 50,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            opt_ref     = true,
            verbose     = true)

Do PT2
"""
function do_fois_pt2(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
            H0          = "Hcmf",
            max_iter    = 50,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            opt_ref     = true,
            verbose     = true) where {T,N,R}
    #={{{=#
    @printf(" |== Solve for BST PT1 Wavefunction ================================\n")
    println(" H0          : ", H0          ) 
    println(" max_iter    : ", max_iter    ) 
    println(" nbody       : ", nbody       ) 
    println(" thresh_foi  : ", thresh_foi  ) 
    println(" tol         : ", tol         ) 
    println(" opt_ref     : ", opt_ref     ) 
    println(" verbose     : ", verbose     ) 
    @printf("\n")
    @printf(" %-50s", "Length of Reference: ")
    @printf("%10i\n", length(ref))

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    
    if opt_ref 
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    end

    #
    # Get First order wavefunction
    println()
    @printf(" %-50s\n", "Compute compressed FOIS: ")
    time = @elapsed pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    @printf(" %-50s%10.6f seconds\n", "Time spent building compressed FOIS: ",time)
    #display(orth_overlap(pt1_vec, pt1_vec))
    #display(eigen(get_vector(pt1_vec)'*get_vector(pt1_vec)))
    project_out!(pt1_vec, ref)
    
    # 
    # Compress FOIS
    norm1 = sqrt.(orth_dot(pt1_vec, pt1_vec))
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    norm2 = sqrt.(orth_dot(pt1_vec, pt1_vec))
    dim2 = length(pt1_vec)
    @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
    #@printf(" %-50s%10.2e → %-10.2e (thresh = %8.1e)\n", "Norm of |1>: ",norm1, norm2, thresh_foi)
    @printf(" %-50s", "Overlap between <1|0>: ")
    ovlp = nonorth_dot(pt1_vec, ref_vec, verbose=0)
    [@printf("%10.6f ", ovlp[r]) for r in 1:R]
    println()

    # 
    # Solve for first order wavefunction 
    @printf(" %-50s%10i\n", "Compute PT vector. Reference space dim: ", length(ref_vec))
    pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, max_iter=max_iter, H0=H0)
    #@printf(" E(Ref)      = %12.8f\n", e0[1])
    #@printf(" E(PT2) tot  = %12.8f\n", e_pt2)
    @printf(" ==================================================================|\n")
    return e_pt2, pt1_vec 
end
#=}}}=#


"""
    compute_pt2_energy(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            max_iter    = 50,
                            nbody       = 4,
                            thresh      = 1e-6,
                            max_number  = nothing,
                            tol         = 1e-5,
                            opt_ref     = true,
                            verbose     = true) where {T,N,R}

Compute the PT2 energy for the `ref` BST state
"""
function compute_pt2_energy(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            max_iter    = 50,
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            tol         = 1e-5,
                            opt_ref     = true,
                            verbose     = true) where {T,N,R}
#={{{=#
    @printf(" |== Compute PT2 Energy ============================================\n")
    println(" H0          : ", H0          ) 
    println(" max_iter    : ", max_iter    ) 
    println(" nbody       : ", nbody       ) 
    println(" thresh_foi  : ", thresh_foi  ) 
    println(" max_number  : ", max_number  ) 
    println(" tol         : ", tol         ) 
    println(" opt_ref     : ", opt_ref     ) 
    println(" verbose     : ", verbose     ) 
    @printf("\n")
    @printf(" %-50s", "Length of Reference: ")
    @printf("%10i\n", length(ref))
    
    lk = ReentrantLock()

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    clusters = ref_vec.clusters
   
    e_ref = zeros(T,R)

    #e_ref[1] = -196.68470072
    if true 
        if opt_ref 
            @printf(" %-50s\n", "Solve zeroth-order problem: ")
            time = @elapsed e_ref, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
            @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
        else
            @printf(" %-50s", "Compute zeroth-order energy: ")
            flush(stdout)
            @time e_ref = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
        end
    end

    # 
    # get E0 = <0|H0|0>
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time e0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham_0)
    
    if verbose > 0 
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, e_ref[r], e0[r])
        end
    end

    # 
    # define batches (FockConfigs present in resolvant)
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ref_vec.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_x) || continue 
            all(f[2] >= 0 for f in fock_x) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
        end
    end


    jobs_vec = []
    for (fock_x, job) in jobs
        push!(jobs_vec, (fock_x, job))
    end

    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)
    
    #ham_0s = Vector{ClusteredOperator}()
    #for t in Threads.nthreads() 
    #    push!(ham_0s, extract_1body_operator(clustered_ham, op_string = H0) )
    #end


    e2_thread = Vector{Vector{Float64}}()
    for tid in 1:Threads.nthreads()
        push!(e2_thread, zeros(T,R))
    end

    tmp = ceil(length(jobs_vec)/100)
    verbose < 1 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 1 || println(" |0%                                                                                              100%|")
    verbose < 1 || print(" |")
    #@profilehtml @Threads.threads for job in jobs_vec
    nprinted = 0
    alloc = @allocated t = @elapsed begin
        
        @Threads.threads for (jobi,job) in collect(enumerate(jobs_vec))
        #for (jobi,job) in collect(enumerate(jobs_vec))
            fock_sig = job[1]
            tid = Threads.threadid()
            e2_thread[tid] .+= _pt2_job(fock_sig, job[2], ref_vec, cluster_ops, clustered_ham, clustered_ham_0, 
                          tol, nbody, max_iter, verbose, thresh_foi, max_number, e_ref, e0)
            #@btime  _pt2_job($fock_sig, $job[2], $ref_vec, $cluster_ops, $clustered_ham, $clustered_ham_0, 
            #              $tol, $nbody, $max_iter, $verbose, $thresh_foi, $max_number, $e_ref, $e0)
            if verbose > 0
                if  jobi%tmp == 0
                    begin
                        lock(lk)
                        try
                            print("-")
                            nprinted += 1
                            flush(stdout)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end
        end
    end
    flush(stdout)
    for i in nprinted+1:100
        print("-")
    end
    verbose < 1 || println("|")
    flush(stdout)
  
    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n", "Time spent computing E2: ",t,alloc*1e-9)
    e2 = sum(e2_thread) 
    
    for r in 1:R
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", e2[r])
    end
    for r in 1:R
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2): ", e2[r]+e_ref[r])
    end

    
    @printf(" ==================================================================|\n")
    return e2 
end
#=}}}=#


function _pt2_job(sig_fock, job, ket::BSTstate{T,N,R}, cluster_ops, clustered_ham, clustered_ham_0, 
                  tol, nbody, max_iter, verbose, thresh, max_number, e_ref, e0) where {T,N,R}
    #={{{=#

    sig = BSTstate(ket.clusters, ket.p_spaces, ket.q_spaces, T=T, R=R)
    add_fockconfig!(sig, sig_fock)

    data = OrderedDict{TuckerConfig{N}, Vector{Tucker{T,N,R}} }()

    for jobi in job 

        terms, ket_fock, ket_tconfigs = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (ket_tconfig, ket_tuck) in ket_tconfigs
                #
                # find the sig TuckerConfigs reached by applying current Hamiltonian term to ket_tconfig.
                #
                # For example:
                #
                #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                #
                #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                #   have both non-zero P and Q spaces within the current fock sector, "sig_fock".
                #
                # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                # available spaces, this list of which we will keep in "available".
                #

                available = [] # list of lists of index ranges, the cartesian product is the set needed
                #
                # for current term, expand index ranges for active clusters
                for ci in term.clusters
                    tmp = []
                    if haskey(ket.p_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.p_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    if haskey(ket.q_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.q_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    push!(available, tmp)
                end


                #
                # Now loop over cartesian product of available subspaces (those in X above) and
                # create the target TuckerConfig and then evaluate the associated terms
                for prod in Iterators.product(available...)
                    sig_tconfig = [ket_tconfig.config...]
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        sig_tconfig[ci.idx] = prod[cidx]
                    end
                    sig_tconfig = TuckerConfig(sig_tconfig)

                    #
                    # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                    # let's compute the matrix element block, then compress, then add it to any existing compressed
                    # coefficient tensor for that sig TuckerConfig.
                    #
                    # Both the Compression and addition takes a fair amount of work.


                    check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue


                    bound = calc_bound(term, cluster_ops,
                                       sig_fock, sig_tconfig,
                                       ket_fock, ket_tconfig, ket_tuck,
                                       prescreen=thresh)
                    if bound == false 
                        continue
                    end

                    sig_tuck = form_sigma_block_expand(term, cluster_ops,
                                                       sig_fock, sig_tconfig,
                                                       ket_fock, ket_tconfig, ket_tuck,
                                                       max_number=max_number,
                                                       prescreen=thresh)
                    if term isa ClusteredTerm3B && false
                                    
                        @profilehtml for ii in 1:1000
                            form_sigma_block_expand(term, cluster_ops,
                                                       sig_fock, sig_tconfig,
                                                       ket_fock, ket_tconfig, ket_tuck,
                                                       max_number=max_number,
                                                       prescreen=thresh)
                        end
                        @btime form_sigma_block_expand($term, $cluster_ops,
                                                       $sig_fock, $sig_tconfig,
                                                       $ket_fock, $ket_tconfig, $ket_tuck,
                                                       max_number=$max_number,
                                                       prescreen=$thresh)
                        error("stop")
                    end


                    if length(sig_tuck) == 0
                        continue
                    end
                    if norm(sig_tuck) < thresh 
                        continue
                    end
                

                    #compress new addition
                    sig_tuck = compress(sig_tuck, thresh=thresh)
                    
                    length(sig_tuck) > 0 || continue

                    #add to current sigma vector
                    if haskey(sig[sig_fock], sig_tconfig)
                        #sig[sig_fock][sig_tconfig] = nonorth_add(sig[sig_fock][sig_tconfig], sig_tuck)
                                       
                        if haskey(data, sig_tconfig)
                            push!(data[sig_tconfig], sig_tuck)
                        else
                            data[sig_tconfig] = [sig[sig_fock][sig_tconfig], sig_tuck]
                        end

                        #compress result
                        #sig[sig_fock][sig_tconfig] = compress(sig[sig_fock][sig_tconfig], thresh=thresh)
                    else
                        sig[sig_fock][sig_tconfig] = sig_tuck
                    end

                end
            end
        end
    end
    
    # Add results together
    for (tconfig, tucks) in data 
        sig[sig_fock][tconfig] = compress(nonorth_add(tucks), thresh=thresh)
    end
    
    project_out!(sig, ket)
    sig = compress(sig, thresh=thresh)
    #project_out!(sig, ket)

    # if length of sigma is zero get out
    length(sig) > 0 || return zeros(T,R)
    norms = sqrt.(orth_dot(sig,sig))
    for r in 1:R
        if norms[r] < thresh 
            return zeros(T,R)
        end
    end
    

    zero!(sig)
           
    ref = ket
    build_sigma_serial!(sig, ref, cluster_ops, clustered_ham)
    
    # b = <X|H|0> 
    b = -get_vector(sig)
    
    
    # (H0 - E0) |1> = X H |0>
    e2 =zeros(T,R) 
    
    # 
    # get <X|F|0>
    tmp = deepcopy(sig)
    zero!(tmp)
    build_sigma_serial!(tmp, ref, cluster_ops, clustered_ham_0)

    # b = - <X|H|0> + <X|F|0> = -<X|V|0>
    b .+= get_vector(tmp)
    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(sig)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        if haskey(ref, fock)
            for (tconfig, tuck) in tconfigs
                if haskey(ref[fock], tconfig)
                    ref_tuck = ref[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = Vector{Matrix{T}}() 
                    for i in 1:N
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    for r in 1:R
                        Sx[fock][tconfig].core[r] .= transform_basis(ref_tuck.core[r], overlaps)
                    end
                end
            end
        end
    end

    #flush_cache(clustered_ham_0)
    #verbose < 2 || @printf(" %-50s", "Cache zeroth-order Hamiltonian: ")
    #println(size(sig))
    #time = @elapsed cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0)
    #verbose < 2 || @printf(" %-10f", time)

    psi1 = deepcopy(sig)

    #
    # Currently, we need to solve each root separately, this should be fixed
    # by writing our own CG solver
    for r in 1:R
        
        function mymatvec(x)

            xr = BSTstate(sig, R=1)
            xl = BSTstate(sig, R=1)

            #display(size(xr))
            #display(size(x))
            length(xr) .== length(x) || throw(DimensionMismatch)
            set_vector!(xr, x, root=1)
            zero!(xl)
            build_sigma_serial!(xl, xr, cluster_ops, clustered_ham_0, cache=false)
            #build_sigma_serial!(xl, xr, cluster_ops, clustered_ham_0, cache=true)

            # subtract off -E0|1>
            #
            
            scale!(xr,-e0[1])
            #scale!(xr,-e0[r])  # pretty sure this should be uncommented - but it diverges, not sure why
            orth_add!(xl,xr)
            #flush(stdout)

            return get_vector(xl)
        end
        br = b[:,r] .+ get_vector(Sx)[:,r] .* (e_ref[r] - e0[r])


        dim = length(br)
        Axx = LinearMap(mymatvec, dim, dim)


        #@time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0, nbody=1)

        #todo:  setting initial value to zero only makes sense when our reference space is projected out. 
        #       if it's not, then we want to add the reference state components |guess> += |ref><ref|guess>
        #
        x_vector = zeros(T,dim)
        x_vector = get_vector(sig)[:,r]*.1
        #time = @elapsed x, solver = cg!(x_vector, Axx, br, log=true, maxiter=max_iter, verbose=false, abstol=1e-12)
        time = @elapsed x, solver = cg!(x_vector, Axx, br, log=true, maxiter=max_iter, verbose=false, abstol=tol)
        verbose < 2 || @printf(" %-50s%10.6f seconds\n", "Time to solve for PT1 with conjugate gradient: ", time)
    
        set_vector!(psi1, x_vector, root=r)
    end
       

    #flush_cache(clustered_ham_0)
    
    SxC = orth_dot(Sx,psi1)
    #@printf(" %-50s%10.2f\n", "<A|X>C(X): ", SxC)
    #@printf(" <A|X>C(X) = %12.8f\n", SxC)
  
    ecorr = zeros(T,R)
    if length(psi1) < length(ref)
        tmp = deepcopy(ref)
        zero!(tmp)
        build_sigma_serial!(tmp,psi1, cluster_ops, clustered_ham)
        ecorr = nonorth_dot(tmp,ref)
    else
        tmp = deepcopy(psi1)
        zero!(tmp)
        build_sigma_serial!(tmp,ref, cluster_ops, clustered_ham)
        ecorr = nonorth_dot(tmp,psi1)
    end
    e_pt2 = zeros(T,R)
    for r in 1:R
        e_pt2[r] = (e_ref[r] + ecorr[r])/(1+SxC[r])
    end


    return e_pt2 .- e_ref
end
#=}}}=#




