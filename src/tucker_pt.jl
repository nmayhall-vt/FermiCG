

 """
    hylleraas_compressed_mp2a(sig_in::BSTstate{T,N,R}, ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    tol=1e-6,
    nbody=4,
    max_iter=100,
    verbose=1,
    thresh=1e-8) where {T,N,R}


- `H0`: ["H", "Hcmf"] 

Compute compressed PT2.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |Ca| = |I   + Sax| |Ca| E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa*Ca + Hax*Cx = Ca*E + Sax*Cx*E
Hxa*Ca + Hxx*Cx = Sxa*Ca*E + Cx*E

(Hxx - Es)*Cx = Sxa*Ca*E - Hxa*Ca 

Perturbation partitioning: 
        H = F + <0|H-F|0> + λ(H - F - <0|H-F|0>)

Keeping only 1st order terms: 
    (Hxx^0 - E^0)*Cx^1 = Sxa*Ca^0*E^1 - Hxa^1*Ca^0 

Partitioning ensures zero 1st order energy correction
    E^1 = 0

    (Hxx^0 - E^0)*Cx^1 = - Hxa^1*Ca^0 


    (Fxx - <0|F|0>)*Cx = - <x|H - F - <0|V|0> |a> Ca

Leading to a linear system for `Cx`
    Ax=b

After solving, the energy can be obtained as:

    Ca'*Haa*Ca + Ca'*Hax*Cx = E + Ca'*Sax*Cx*E

    E = (Eref + Ca'*Hax*Cx) / (1 + Ca'*Sax*Cx)


"""




"""
    compute_pt1_wavefunction(σ_in::BSTstate{T,N,R}, ψ0::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    nbody=4,
    verbose=1) where {T,N,R}

H0 = F + <0|H - F|0>
V  = H - F - <0|H - F|0>

ψ1 = <X|H - F - E0 + F0|0> / (<0|F|0> - Fx)
   = <X|H|0> - <X|F|0> - (E0-F0)<X|0> / Dx

E2 = <0|V|X>ψ1

The local Tucker factors are first canonicalized to avoid having to solve the Hylleraas functional
"""
function compute_pt1_wavefunction(σ_in::BSTstate{T,N,R}, ψ0::BSTstate{T,N,R}, cluster_ops, clustered_ham, clustered_ham_0, E0, F0;
    H0="Hcmf",
    nbody=4,
    verbose=1) where {T,N,R}
    
    #
    # Copy data
    verbose < 2 || @printf(" copy sigma... \n")
    flush(stdout)
    σ = deepcopy(σ_in)
    zero!(σ)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)


    #
    # Rotate the local tucker factors to diagonalize the zeroth order hamiltonians and build F diagonal
    verbose < 2 || @printf(" form_1body_operator_diagonal... \n")
    flush(stdout)
    Fdiag = BSTstate(σ, R=1)
    form_1body_operator_diagonal!(σ, Fdiag, cluster_ops, pseudo_canon=true)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)
    
    
    #
    # Get Overlap <X|A>C(A)
    verbose < 2 || @printf(" get_overlap... \n")
    flush(stdout)
    Sx = deepcopy(σ)
    zero!(Sx)
    for (fock, tconfigs) in Sx
        if haskey(ψ0, fock)
            for (tconfig, tuck) in tconfigs
                if haskey(ψ0[fock], tconfig)
                    ref_tuck = ψ0[fock][tconfig]
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
    verbose < 2 || @printf(" done.\n")
    flush(stdout)


    # 
    # We need the numerator: <X|V|0> = <X|H|0> - <X|F|0> - (E0 - F0)<X|0>
    # 
    # Build exact Hamiltonian within FOIS defined by `sig`: <X|H|0>
    verbose < 2 || @printf(" build_sigma!... \n")
    flush(stdout)
    verbose < 1 || @printf(" %-50s%10i\n", "Length of input      FOIS: ", length(σ))
    time = @elapsed alloc = @allocated build_sigma!(σ, ψ0, cluster_ops, clustered_ham)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|H|0>: ", time, alloc/1e9)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)

    # subtract <X|F|0>
    verbose < 2 || @printf(" build_sigma!... \n")
    flush(stdout)
    XF0 = deepcopy(σ)
    zero!(XF0)
    time = @elapsed alloc = @allocated build_sigma!(XF0, ψ0, cluster_ops, clustered_ham_0)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|F|0>: ", time, alloc/1e9)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)

    verbose < 2 || @printf(" compute correction... \n")
    flush(stdout)
    
    σ = σ - XF0 - scale(Sx, E0 .- F0)
    
    ψ1 = deepcopy(σ)
    Fv = get_vector(Fdiag, 1)
    for r in 1:R
        ψ1r = get_vector(ψ1, r)
        sr = get_vector(Sx, r)
        
        denom = F0[r] .- Fv .+ 1e-12  # this shift just protects against exact/accidental zeros
        
        ψ1r ./= denom
        
        set_vector!(ψ1, ψ1r[:,1], root=r)
    end

    ecorr = orth_dot(σ, ψ1)
  
    E2 = zeros(R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
    end
    
    verbose < 2 || @printf(" done.\n")
    flush(stdout)
    
    return ψ1, E2, ecorr

end

"""
    compute_pt1_wavefunction(σ_in::BSTstate{T,N,R}, ψ0::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    nbody=4,
    verbose=1) where {T,N,R}

TBW
"""
function compute_pt1_wavefunction(σ_in::BSTstate{T,N,R}, ψ0::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    nbody=4,
    verbose=1) where {T,N,R}

    verbose < 1 || println()
    verbose < 1 || println(" |...................................BST-PT2............................................")
    flush(stdout)
    #
    # Copy data
    verbose < 2 || @printf(" copy sigma... \n")
    flush(stdout)
    σ = deepcopy(σ_in)
    zero!(σ)
    verbose < 2 || @printf("done.\n")
    flush(stdout)


    #
    # Extract zeroth-order hamiltonian
    verbose < 2 || @printf(" extract_1body_operator... \n")
    flush(stdout)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)
    verbose < 2 || @printf("done.\n")
    flush(stdout)
    

    # 
    # get E0 = <0|H|0>
    verbose < 2 || @printf(" compute_expectation_value... \n")
    flush(stdout)
    E0 = compute_expectation_value(ψ0, cluster_ops, clustered_ham)
    verbose < 2 || @printf("done.\n")
    flush(stdout)


    # 
    # get F0 = <0|F|0>
    verbose < 2 || @printf(" compute_expectation_value... \n")
    flush(stdout)
    F0 = compute_expectation_value(ψ0, cluster_ops, clustered_ham_0)
    verbose < 2 || @printf("done.\n")
    flush(stdout)


    if verbose > 1
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, E0[r], F0[r])
        end
    end

    verbose < 2 || @printf(" compute_pt1_wavefunction... \n")
    flush(stdout)
    ψ1, E2, ecorr = compute_pt1_wavefunction(σ, ψ0, cluster_ops, clustered_ham, clustered_ham_0,  E0, F0, H0=H0, verbose=verbose)
    verbose < 2 || @printf("done.\n")
    flush(stdout)

    verbose < 1 || for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end

    verbose < 1 || @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    verbose < 1 || for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    verbose < 1 || println(" ......................................................................................|")

    return ψ1, E2, ecorr 

end



"""
    compute_pt1_wavefunction(ψ0::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0 = "Hcmf",
    nbody = 4,
    thresh_foi = 1e-7
    verbose = 1,
    max_number = nothing) where {T,N,R}

TBW
"""
function compute_pt1_wavefunction(ψ0::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0 = "Hcmf",
    nbody = 4,
    thresh_foi = 1e-7,
    verbose = 1,
    max_number = nothing) where {T,N,R}

    #
    # Build target FOIS
    time = @elapsed alloc = @allocated σ = FermiCG.build_compressed_1st_order_state(ψ0, cluster_ops, clustered_ham, nbody=4, thresh=thresh_foi, max_number=max_number)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute Compressed FOIS: ", time, alloc/1e9)

    return compute_pt1_wavefunction(σ, ψ0, cluster_ops, clustered_ham, H0=H0, nbody=nbody, verbose=verbose)
end


"""
    form_1body_operator_diagonal!(sig::BSTstate{T,N,R}, Fdiag::BSTstate{T,N,1}, cluster_ops; H0="Hcmf") where {T,N,R}

TBW
"""
function form_1body_operator_diagonal!(sig::BSTstate{T,N,R}, Fdiag::BSTstate{T,N,1}, cluster_ops; H0="Hcmf", pseudo_canon=false) where {T,N,R}
    # Fdiag = BSTstate(v, R=1)
    clusters = sig.clusters

    zero!(Fdiag)
    for (fock, tconfigs) in sig
        for (tconfig, tuck) in tconfigs

            #
            # Initialize energy list of lists
            energies = Vector{Vector{T}}([])
            for ci in 1:N
                push!(energies, [])
            end

            # 
            # Rotate tucker fractors to diagonalize each cluster hamiltonian
            for ci in clusters
                Ui = tuck.factors[ci.idx]
                if size(Ui, 2) > 1

                    # build the local "fock" operator in the current tucker basis
                    Hi = cluster_ops[ci.idx][H0][(fock[ci.idx], fock[ci.idx])][tconfig[ci.idx], tconfig[ci.idx]]
                    Hi = Ui' * Hi * Ui

                    if pseudo_canon
                        # diagonalize and rotate tucker factors
                        F = eigen(Symmetric(Hi))
                        sig[fock][tconfig].factors[ci.idx] .= Ui * F.vectors
                        Fdiag[fock][tconfig].factors[ci.idx] .= sig[fock][tconfig].factors[ci.idx]

                        energies[ci.idx] = F.values
                    else
                        # just take diagonal
                        energies[ci.idx] = diag(Hi) 
                    end
                elseif size(Ui, 2) == 1
                    Hi = cluster_ops[ci.idx][H0][(fock[ci.idx], fock[ci.idx])][tconfig[ci.idx], tconfig[ci.idx]]
                    e = Ui' * Hi * Ui

                    length(e) == 1 || throw(DimensionMismatch)

                    energies[ci.idx] = [e[1]]
                end
            end

            #
            # Add the local energies together to form zeroth order energies for each element
            fcore = Fdiag[fock][tconfig].core
            length(fcore) == 1 || throw(DimensionMismatch)
            for i in CartesianIndices(fcore[1])
                for ci in 1:N
                    fcore[1][i] += energies[ci][i[ci]]
                end
            end
        end
    end
end


function hylleraas_compressed_mp2(sig_in::BSTstate{T,N,R}, ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    tol=1e-6,
    nbody=4,
    max_iter=100,
    verbose=1,
    thresh=1e-8) where {T,N,R}
    
    #
    # Extract zeroth-order hamiltonian
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)

    # 
    # Build exact Hamiltonian within FOIS defined by `sig_in`: <X|H|0>
    sig = deepcopy(sig_in)
    verbose < 1 || @printf(" %-50s%10i\n", "Length of input      FOIS: ", length(sig_in))
    #@printf(" %-50s%10i\n", "Length of compressed FOIS: ", length(sig))
    #project_out!(sig, ref)
    zero!(sig)

    time = @elapsed alloc = @allocated build_sigma!(sig, ref, cluster_ops, clustered_ham)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|V|0>: ", time, alloc/1e9)


    e2 = zeros(T, R)

    # 
    # get E_ref = <0|H|0>
    e_ref = compute_expectation_value(ref, cluster_ops, clustered_ham)

    # 
    # get E0 = <0|H0|0>
    e0 = compute_expectation_value(ref, cluster_ops, clustered_ham_0)


    if verbose > 0
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, e_ref[r], e0[r])
        end
    end



    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(sig)
    zero!(Sx)
    for (fock, tconfigs) in Sx
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


    # 
    # Build b
    #   b = <X|F|a>Ca - <X|H|a>Ca + <X|a>Ca (<0|H|0> - <0|F|0>)
    
    # b += - <X|H|0>
    b = -get_vector(sig)

    # b += <X|F|0>
    tmp = deepcopy(sig)
    zero!(tmp)
    time = @elapsed alloc = @allocated build_sigma!(tmp, ref, cluster_ops, clustered_ham_0)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|F|0>: ", time, alloc/1e9)
    b .+= get_vector(tmp)

    # b += Sx * (<0|H-F|0>)
    #   taken care off in loop over states


    #@printf(" Norm of b         : %18.12f\n", sum(b.*b))
    flush_cache(clustered_ham_0)
    time = @elapsed alloc = @allocated cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0)
    psi1 = deepcopy(sig)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Cache zeroth-order Hamiltonian: ", time, alloc/1e9)


    #  (Fxx + Eref - <0|F|0> - Es)*Cxs = Sxa*Cas*Eref - Hxa
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
            build_sigma!(xl, xr, cluster_ops, clustered_ham_0, cache=true)

            # subtract off -E0|1>
            #

            scale!(xr, -e0[1])
            #scale!(xr,-e0[r])  # pretty sure this should be uncommented - but it diverges, not sure why
            orth_add!(xl, xr)
            flush(stdout)

            return get_vector(xl)
        end
        
        # b += Sx * (<0|H-F|0>)
        br = b[:, r] .+ get_vector(Sx)[:, r] .* (e_ref[r] - e0[r])


        dim = length(br)
        Axx = LinearMap(mymatvec, dim, dim)


        #@time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0, nbody=1)

        #todo:  setting initial value to zero only makes sense when our reference space is projected out. 
        #       if it's not, then we want to add the reference state components |guess> += |ref><ref|guess>
        #
        x_vector = zeros(T, dim)
        # x_vector = get_vector(Sx)[:,r] + get_vector(sig)[:, r] * 0.1
        # x_vector = get_vector(sig)[:, r] * 0.1
        time = @elapsed alloc = @allocated x, solver = cg!(x_vector, Axx, br, log=true, maxiter=max_iter, verbose=true, abstol=tol)
        verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Time to solve for PT1 with conjugate gradient: ", time, alloc/1e9)

        set_vector!(psi1, x_vector, root=r)
    end

    flush_cache(clustered_ham_0)

    SxC = orth_dot(Sx, psi1)
    #@printf(" %-50s%10.2f\n", "<A|X>C(X): ", SxC)
    #@printf(" <A|X>C(X) = %12.8f\n", SxC)

    tmp = deepcopy(ref)
    zero!(tmp)
    time = @elapsed alloc = @allocated build_sigma!(tmp, psi1, cluster_ops, clustered_ham)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <0|H|1>: ", time, alloc/1e9)

    ecorr = nonorth_dot(tmp, ref)
    
    for r in 1:R
        SS = orth_dot(Sx, Sx)
        @printf(" SxC[r] %12.8f SxSx %12.8f\n", SxC[r], SS[r])
    end
    
    e_pt2 = zeros(T, R)
    for r in 1:R
        e_pt2[r] = (e_ref[r] + ecorr[r]) / (1 + SxC[r])
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", e_pt2[r] - e_ref[r])
    end
    for r in 1:R
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2): ", e_pt2[r])
    end

    return psi1, e_pt2

end





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
        time = @elapsed e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    end

    #
    # Get First order wavefunction
    println()
    @printf(" %-50s\n", "Compute compressed FOIS: ")
    time = @elapsed pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    @printf(" %-50s%10.6f seconds\n", "Time spent building compressed FOIS: ",time)
    # display(orth_overlap(pt1_vec, pt1_vec))
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
    # display(orth_overlap(pt1_vec, pt1_vec))
    pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham)
    # pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, max_iter=max_iter, H0=H0)
    #@printf(" E(Ref)      = %12.8f\n", e0[1])
    #@printf(" E(PT2) tot  = %12.8f\n", e_pt2)
    @printf(" ==================================================================|\n")
    return e_pt2, pt1_vec 
end


"""
    compute_pt2_energy(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            opt_ref     = true,
                            ci_tol      = 1e-6,
                            verbose     = true) where {T,N,R}

Directly compute the PT2 energy, in parallel, with each thread handling a single `FockConfig` at a time.
"""
function compute_pt2_energy(ref::BSTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            opt_ref     = true,
                            ci_tol      = 1e-6,
                            verbose     = 1,
                            prescreen   = false,
                            compress_twice = false) where {T,N,R}
    println()
    println(" |...................................BST-PT2............................................")
    verbose < 1 || println(" H0          : ", H0          ) 
    verbose < 1 || println(" nbody       : ", nbody       ) 
    verbose < 1 || println(" thresh_foi  : ", thresh_foi  ) 
    verbose < 1 || println(" max_number  : ", max_number  ) 
    verbose < 1 || println(" opt_ref     : ", opt_ref     ) 
    verbose < 1 || println(" ci_tol      : ", ci_tol       ) 
    verbose < 1 || println(" verbose     : ", verbose     ) 
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s", "Length of Reference: ")
    verbose < 1 || @printf("%10i\n", length(ref))
    
    lk = ReentrantLock()

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    clusters = ref_vec.clusters
   
    E0 = zeros(T,R)

    if opt_ref 
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # 
    # get E0 = <0|H0|0>
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time F0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham_0)
    
    if verbose > 0 
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, E0[r], F0[r])
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
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")
    #@profilehtml @Threads.threads for job in jobs_vec
    nprinted = 0
    alloc = @allocated t = @elapsed begin
        
        @Threads.threads for (jobi,job) in collect(enumerate(jobs_vec))
        #for (jobi,job) in collect(enumerate(jobs_vec))
            fock_sig = job[1]
            tid = Threads.threadid()
            e2_thread[tid] .+= _pt2_job(fock_sig, job[2], ref_vec, cluster_ops, clustered_ham, clustered_ham_0, 
                          nbody, verbose, thresh_foi, max_number, E0, F0, prescreen, compress_twice)
            if verbose > 1
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
    verbose < 2 || for i in nprinted+1:100
        print("-")
    end
    verbose < 2 || println("|")
    flush(stdout)
  
    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n", "Time spent computing E2: ",t,alloc*1e-9)
    ecorr = sum(e2_thread) 
   
    E2 = zeros(R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end

    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    println(" ......................................................................................|")
    
    return E2 
end


function _pt2_job(sig_fock, job, ket::BSTstate{T,N,R}, cluster_ops, clustered_ham, clustered_ham_0,
    nbody, verbose, thresh, max_number, E0, F0, prescreen, compress_twice) where {T,N,R}

    sig = BSTstate(ket.clusters, ket.p_spaces, ket.q_spaces, T=T, R=R)
    add_fockconfig!(sig, sig_fock)

    data = OrderedDict{TuckerConfig{N},Vector{Tucker{T,N,R}}}()

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


                    if prescreen
                        bound = calc_bound(term, cluster_ops,
                            sig_fock, sig_tconfig,
                            ket_fock, ket_tconfig, ket_tuck,
                            prescreen=thresh)
                        bound == true || continue
                    end

                    sig_tuck = form_sigma_block_expand(term, cluster_ops,
                        sig_fock, sig_tconfig,
                        ket_fock, ket_tconfig, ket_tuck,
                        max_number=max_number,
                        prescreen=thresh)
                    #                    if term isa ClusteredTerm2B && false 
                    #                                    
                    #                        #@profilehtml for ii in 1:3
                    #                        #    form_sigma_block_expand(term, cluster_ops,
                    #                        #                               sig_fock, sig_tconfig,
                    #                        #                               ket_fock, ket_tconfig, ket_tuck,
                    #                        #                               max_number=max_number,
                    #                        #                               prescreen=thresh)
                    #                        #end
                    #                        @btime form_sigma_block_expand($term, $cluster_ops,
                    #                                                       $sig_fock, $sig_tconfig,
                    #                                                       $ket_fock, $ket_tconfig, $ket_tuck,
                    #                                                       max_number=$max_number,
                    #                                                       prescreen=$thresh)
                    #                        error("stop")
                    #                    end


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

    # 
    # Add results together to get final FOIS for this job
    for (tconfig, tucks) in data
        if compress_twice
            sig[sig_fock][tconfig] = compress(nonorth_add(tucks), thresh=thresh)
        else
            sig[sig_fock][tconfig] = nonorth_add(tucks)
        end
    end

    # Compute PT2 energy for this job
    v_pt, e_pt, ecorr = compute_pt1_wavefunction(sig, ket, cluster_ops, clustered_ham, clustered_ham_0, E0, F0, verbose=0)

    return ecorr 
end



