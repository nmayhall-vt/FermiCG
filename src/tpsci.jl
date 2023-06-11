using Printf

"""
    tpsci_ci(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
            thresh_cipsi = 1e-2,
            thresh_foi   = 1e-6,
            thresh_asci  = nothing,
            thresh_var   = nothing,
            thresh_spin  = nothing,
            max_iter     = 10,
            conv_thresh  = 1e-4,
            nbody        = 4,
            incremental  = true,
            ci_conv      = 1e-5,
            ci_max_iter  = 50,
            ci_max_ss_vecs = 12,
    	    ci_lindep_thresh= 1e-11,
            davidson     = false,
            max_mem_ci   = 20.0, 
            threaded     = true) where {T,N,R}

# Run TPSCI 
- `thresh_cipsi`: threshold for which configurations to include in the variational space. Add if |c^{(1)}| > `thresh_cipsi`
- `thresh_foi`  : threshold for which terms to keep in the H|0> vector used to form the first order wavefunction
- `thresh_asci` : threshold for determining from which variational configurations  ``|c^{(0)}_i|`` > `thresh_asci` 
- `thresh_var`  : threshold for clipping the result of the variational wavefunction. Not really needed default set to nothing 
- `thresh_spin` : threshold for clipping the result of the S2 residual vector for the spin extension. 
- `max_iter`    : maximum selected CI iterations
- `conv_thresh` : stop selected CI iterations when energy change is smaller than `conv_thresh`
- `nbody`       : only consider up to `nbody` terms when searching for new configurations
- `incremental` : for the sigma vector incrementally between iterations
- `ci_conv`     : convergence threshold for the inner CI step (only needed when davidson is used)
- `ci_max_iter` : max iterations for inner CI step (only needed when davidson is used) 
- `ci_max_ss_vecs`: max subspace size for inner CI step (only needed when davidson is used) 
- `davidson`    : use davidson? changes to true after needing more than max_mem_ci
- `max_mem_ci`  : maximum memory (Gb) allowed for storing full H. If more is needed, do Davidson. 
- `threaded`    : Use multithreading? 
"""
function tpsci_ci(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
    thresh_cipsi    = 1e-2,
    thresh_foi      = 1e-6,
    thresh_asci     = nothing,
    thresh_var      = nothing,
    thresh_spin     = nothing,
    max_iter        = 10,
    conv_thresh     = 1e-4,
    nbody           = 4,
    incremental     = true,
    ci_conv         = 1e-5,
    ci_max_iter     = 50,
    ci_max_ss_vecs  = 12,
    ci_lindep_thresh= 1e-12,
    davidson        = false,
    max_mem_ci      = 20.0,
    threaded        = true) where {T,N,R}

    vec_var = deepcopy(ci_vector)
    vec_pt = deepcopy(ci_vector)
    length(ci_vector) > 0 || error(" input vector has zero length")
    zero!(vec_pt)
    e0 = zeros(T,R) 
    e2 = zeros(T,R) 
    e0_last = zeros(T,R)
   
    clustered_S2 = extract_S2(ci_vector.clusters)

    println(" ci_vector     : ", size(ci_vector) ) 
    println(" thresh_cipsi  : ", thresh_cipsi   ) 
    println(" thresh_foi    : ", thresh_foi     ) 
    println(" thresh_asci   : ", thresh_asci    ) 
    println(" thresh_var    : ", thresh_var     ) 
    println(" thresh_spin   : ", thresh_spin    ) 
    println(" max_iter      : ", max_iter       ) 
    println(" conv_thresh   : ", conv_thresh    ) 
    println(" nbody         : ", nbody          ) 
    println(" incremental   : ", incremental    ) 
    println(" ci_conv       : ", ci_conv        ) 
    println(" ci_max_iter   : ", ci_max_iter    ) 
    println(" ci_max_ss_vecs: ", ci_max_ss_vecs ) 
    println(" ci_lindep_thresh: ", ci_lindep_thresh ) 
    println(" davidson      : ", davidson       ) 
    println(" max_mem_ci    : ", max_mem_ci     ) 
    println(" threaded      : ", threaded       ) 
    
    vec_asci_old = TPSCIstate(ci_vector.clusters, R=R, T=T)
    sig = TPSCIstate(ci_vector.clusters, R=R, T=T)
    sig_old = TPSCIstate(ci_vector.clusters, R=R, T=T)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = "Hcmf") 
    
    H = zeros(T,size(ci_vector))
   
    vec_var_old = deepcopy(ci_vector)
    to = TimerOutput()

    for it in 1:max_iter

        println()
        println()
        println(" ===================================================================")
        @printf("     Selected CI Iteration: %4i epsilon: %12.8f\n", it,thresh_cipsi)
        println(" ===================================================================")

        if it > 1
            if thresh_var !== nothing 
                l1 = length(vec_var)
                clip!(vec_var, thresh=thresh_var)
                l2 = length(vec_var)
                @printf(" Clip values < %8.1e         %6i → %6i\n", thresh_var, l1, l2)
            end

            #project_out!(vec_pt, vec_var)
    
            vec_var_old = deepcopy(vec_var)

            l1 = length(vec_var)
            zero!(vec_pt)
            add!(vec_var, vec_pt)
            l2 = length(vec_var)
           
            @printf("%-50s%6i → %6i\n", " Add pt vector to current space", l1, l2)
        end

        @timeit to "s2 extension" if thresh_spin != nothing 
            # S2|ψs> - |ψs><ψs|S2|ψs> = |rs>
            # add |rs>
            spin_residual = deepcopy(vec_var)
            if threaded 
                spin_residual = open_matvec_thread(vec_var, cluster_ops, clustered_S2, nbody=nbody, thresh=thresh_spin)
            else
                spin_residual = open_matvec_serial(vec_var, cluster_ops, clustered_S2, nbody=nbody, thresh=thresh_spin)
            end

            spin_expval = FermiCG.overlap(vec_var,spin_residual)
            # println(" Expectation values of S2:")
            # for i in 1:size(spin_expval,1)
            #     for j in 1:size(spin_expval,2)
            #         @printf(" %6.3f",spin_expval[i,j])
            #     end
            #     println()
            # end
            spin_residual = spin_residual - (vec_var * spin_expval)
            for r in 1:R
                @printf(" S^2 Residual %12.8f\n", dot(spin_residual, spin_residual, r, r))
            end
            zero!(spin_residual)

            l1 = size(vec_var)[1]
            add!(vec_var, spin_residual)
            l2 = size(vec_var)[1]
            
            @printf("%-50s%6i → %6i\n", " Add spin completing states", l1, l2)
            flush(stdout)
        end
        e0 = nothing
        mem_needed = sizeof(T)*length(vec_var)*length(vec_var)*1e-9
        @printf(" Memory needed to hold full CI matrix: %12.8f (Gb) Max allowed: %12.8f (Gb)\n",mem_needed, max_mem_ci)
        flush(stdout)
        @timeit to "ci" begin
            if (mem_needed > max_mem_ci) || davidson == true
                orthonormalize!(vec_var)
                e0, vec_var = tps_ci_davidson(vec_var, cluster_ops, clustered_ham,
                                              conv_thresh = ci_conv,
                                              max_iter = ci_max_iter,
                                              max_ss_vecs = ci_max_ss_vecs)
            else
                if it > 1 
                    # just update matrix
                    e0, vec_var, H = tps_ci_direct(vec_var, cluster_ops, clustered_ham, 
                                                   H_old = H,
                                                   v_old = vec_var_old,
                                                   conv_thresh = ci_conv,
                                                   max_ss_vecs = ci_max_ss_vecs,
                                                   max_iter = ci_max_iter,
						   lindep_thresh = ci_lindep_thresh)
                else
                    e0, vec_var, H = tps_ci_direct(vec_var, cluster_ops, clustered_ham,
                                                   conv_thresh = ci_conv,
                                                   max_ss_vecs = ci_max_ss_vecs,
                                                   max_iter = ci_max_iter,
						   lindep_thresh = ci_lindep_thresh)
                end
            end
        end
        flush(stdout)
      
       
        # get barycentric energy <0|H0|0>
        Efock = compute_expectation_value_parallel(vec_var, cluster_ops, clustered_ham_0)
        #Efock = nothing
        flush(stdout)


        vec_asci = deepcopy(vec_var)
        if thresh_asci != nothing
            l1 = length(vec_asci)
            clip!(vec_asci, thresh=thresh_asci)
            l2 = length(vec_asci)
            @printf("%-50s%6i → %6i\n", " Length of ASCI vector", l1, l2)
        end
        
        #
        #   -- Incremental sigma vector build --
        #
        #   define projector onto previous tpsci space
        #   P^{i-1} = \sum_l |v_l^{i-1}><v_l^i{i-1}| 
        #
        #   and the orthogonal complement
        #   Q^{i-1} = I - P^{i-1}
        #
        #   Now use this to write our next iteration sigma vector in terms of a 
        #   linear combination of our previous sigma vectors and the application of
        #   the hamiltonian to the Q projection of the new variational state (which should 
        #   get smaller with each iteration)
        # 
        #   |sig_l^i> = H|v_l^i> 
        #             = H (P^{i-1} + 1 - P^{i-1}) |v_l^i> 
        #             = \sum_k H |v_k^{i-1}><v_k^{i-1}|v_l^i> + H(|v_l^i> - \sum_k |v_k^{i-1}><v_k^{i-1}|v_l^i>)
        #             = \sum_k |sig_k^{i-1}>s_kl^{i-1,i} + H(|v_l^i> - \sum_k |v_k^{i-1}> s_kl^{i-1,i})
        #   
        #
        #   Not done, but we could: Now rotate into the singular vector basis of the overlap: s_kl = U_ka S_a V_al
        #
        #   |sig_l^i> V'_la = |sig_k^{i-1}> U_ia S_a + H ( |v_l^i> V'_la - |v_l^{i-1}> U_ka S_a
        #
        #   Then rotate back by left multiplying by V_al
        #   
        #
        #

        if incremental 
            #
            # compute overlap of new variational vectors with old
   

            S = overlap(vec_asci_old, vec_asci)
  
            println(" Overlap between old and new eigenvectors:")
            for i in 1:size(S,1)
                for j in 1:size(S,2)
                    @printf(" %6.3f",S[i,j])
                end
                println()
            end
            println()

            F = svd(S)
            println(" Singular values of overlap:")
            for i in F.S 
                @printf(" %6.3f\n",i)
            end


            #tmp = deepcopy(sig)
            #@timeit to "sig rotate" sig = sig_old * F.U * Diagonal(F.S)
            #@timeit to "vec rotate" del_v0 = vec_asci*F.V - (vec_asci_old * F.U * Diagonal(F.S))
            @timeit to "sig rotate" sig = sig_old * S
            @timeit to "vec rotate" del_v0 = vec_asci - (vec_asci_old * S)

            println(" Norm of new projection:")
            [@printf(" %12.8f\n",i) for i in norm(del_v0)]
            println()


            @timeit to "copy" vec_asci_old = deepcopy(vec_asci)

            @timeit to "matvec" if threaded 
                del_sig_it = open_matvec_thread(del_v0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
            else
                del_sig_it = open_matvec_serial(del_v0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
            end
            flush(stdout)
            @timeit to "sig update" add!(sig, del_sig_it)

            #sig = sig * F.Vt


            #
            #  |1> = |x><x|H|0>/delx
            #  E2 = <0|H|1> = <0|H|x><
            #
            # | PHP  PHQ ||PC| = |PC| E 
            # | QHP  QHQ ||QC| = |QC| 
            #
            # PHPC + PHQC = PCe
            # QHPC + QHQC = QCe
            #
            # QC = (QHQ - e)^-1 QHPC
            #
            # C'PHPC + C'PHQC = e
            #
            #   X = P + Q
            #
            # (QHQ-e)*QC = -QHP*PC
            # (X-P)H(X-P)C - e(X-P)C = -(X-P)HP*PC
            # XHXC - PHXC - XHPC + PHPC - eXC + ePC = -(X-P)HP*PC
 
            #
            # QC = XC - PC
            #
            # XHX*XC - PHX*XC - XHP*PC + PHP*PC - e*XP + e*PC = -XHP*PC + PHP*PC

            @timeit to "copy" sig_old = deepcopy(sig)


            @timeit to "project out" project_out!(sig, vec_asci)
            println(" Length of FOIS vector: ", length(sig))
    
            @printf(" %-50s", "Compute diagonal: ")
            flush(stdout)
            @timeit to "diagonal" @time Hd = compute_diagonal(sig, cluster_ops, "Hcmf")
            println()
            flush(stdout)
    
    
            sig_v = get_vector(sig)
            v_pt  = zeros(T, size(sig_v))

            norms = norm(vec_asci);
            println()
            @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)") 
            for r in 1:R
                denom = 1.0 ./ (Efock[r]/(norms[r]*norms[r]) .- Hd)  
                v_pt[:,r] .= denom .* sig_v[:,r] 
                e2[r] = sum(sig_v[:,r] .* v_pt[:,r])

                @printf(" %5s %12.8f %12.8f\n",r, e0[r]/norms[r], e0[r]/(norms[r]*norms[r]) + e2[r])
            end


            @timeit to "copy" vec_pt = deepcopy(sig)
            set_vector!(vec_pt,Matrix{T}(v_pt))
        else
            @timeit to "pt1" e2, vec_pt = compute_pt1_wavefunction(vec_asci, cluster_ops, clustered_ham, E0=Efock, thresh_foi=thresh_foi, threaded=threaded, nbody=nbody)
        end
        flush(stdout)
        
        l1 = length(vec_pt)
        clip!(vec_pt, thresh=thresh_cipsi)
        l2 = length(vec_pt)
            
        #project_out!(sig, vec_asci)


        @printf("%-50s%6i → %6i\n", " Length of PT1  vector", l1, l2)
        #add!(vec_var, vec_pt)

        if maximum(abs.(e0_last .- e0)) < conv_thresh
            print_tpsci_iter(vec_var, it, e0, true)
            break
        else
            print_tpsci_iter(vec_var, it, e0, false)
            e0_last .= e0
        end
        flush(stdout)
    end
    
    println("")
    show(to)
    println("")
    flush(stdout)
    return e0, vec_var 
end




