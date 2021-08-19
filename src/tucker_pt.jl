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
    project_out!(sig, ref, thresh=thresh)
    zero!(sig)
            
    @printf(" %-50s", "Build exact <X|V|0>: ")
    @time build_sigma!(sig, ref, cluster_ops, clustered_ham)
    
    # b = <X|H|0> 
    b = -get_vectors(sig)
    
    
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
    b .+= get_vectors(tmp)
    
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
            set_vector!(xr,x,1)
            zero!(xl)
            build_sigma!(xl, xr, cluster_ops, clustered_ham_0, cache=true)

            # subtract off -E0|1>
            #
            
            scale!(xr,-e0[1])
            #scale!(xr,-e0[r])  # pretty sure this should be uncommented - but it diverges, not sure why
            orth_add!(xl,xr)
            flush(stdout)

            return get_vectors(xl)
        end
        br = b[:,r] .+ get_vectors(Sx)[:,r] .* (e_ref[r] - e0[r])


        dim = length(br)
        Axx = LinearMap(mymatvec, dim, dim)


        #@time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0, nbody=1)

        #todo:  setting initial value to zero only makes sense when our reference space is projected out. 
        #       if it's not, then we want to add the reference state components |guess> += |ref><ref|guess>
        #
        x_vector = zeros(T,dim)
        x_vector = get_vectors(sig)[:,r]*.1
        time = @elapsed x, solver = cg!(x_vector, Axx, br, log=true, maxiter=max_iter, verbose=true, abstol=tol)
        @printf(" %-50s%10.6f seconds\n", "Time to solve for PT1 with conjugate gradient: ", time)
    
        set_vector!(psi1,x_vector,r)
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
        @printf(" State %3i: %-37s%12.8f\n", r, "E(PT2) corr: ", e_pt2[r]-e_ref[r])
    end
    for r in 1:R
        @printf(" State %3i: %-37s%12.8f\n", r, "E(PT2): ", e_pt2[r])
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
        time = @elapsed e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    end

    #
    # Get First order wavefunction
    println()
    @printf(" %-50s\n", "Compute compressed FOIS: ")
    time = @elapsed pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    @printf(" %-50s%10.6f seconds\n", "Time spent building compressed FOIS: ",time)
    #display(orth_overlap(pt1_vec, pt1_vec))
    #display(eigen(get_vectors(pt1_vec)'*get_vectors(pt1_vec)))
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

