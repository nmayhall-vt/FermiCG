function sparse_lanczos(ψ0::TPSCIstate{T,N,R}, cluster_ops, clustered_ham, ϵ=1e-4) where {T,N,R}
    
    @printf(" |== Sparse Lanczos ================================================\n")
    
    clustered_S2 = extract_S2(ψ0.clusters, T=T)
    
    subspace = Vector{TPSCIstate{T,N,R}}([])
  
    push!(subspace, ψ0)

    # H = matrix_element(ψ0, ψ0, cluster_ops, clustered_ham)
    # S2 = matrix_element(ψ0, ψ0, cluster_ops, clustered_S2)

    H = _build_subspace(subspace, cluster_ops, clustered_ham)
    S2 = _build_subspace(subspace, cluster_ops, clustered_S2)
    F = eigen(H)

    H = F.vectors' * H * F.vectors
    S2 = F.vectors' * S2 * F.vectors

    s2_err = norm(H*S2 - S2*H)
    @printf(" Error in [H,S2]: %12.8f\n", s2_err)

    _print_sparse_lanczos_iter(diag(H), diag(S2), R)
    
    # Form residual
    #
    #   |r_s> = |σ_s> - |ψ_t><ψ_t|H|ψ_s>
    σ = FermiCG.open_matvec_thread(ψ0, cluster_ops, clustered_ham, thresh=ϵ)
    # add!(vec_var, vec_pt)

    @printf(" ==================================================================|\n")

end

function _print_sparse_lanczos_iter(e, s2, R)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e[r], abs(s2[r]))
    end

    # if verbose > 1
    #     for r in 1:R
    #         display(vec_out, root=r)
    #     end
    # end
    flush(stdout)
end


function _build_subspace(ss::Vector{TPSCIstate{T,N,R}}, cluster_ops, clustered_ham) where {T,N,R}
    #
    #   |1,1:1,R,  1,1:R:2R, ...   
    #
    ss_dim = length(ss)*R
    H = zeros(T, ss_dim, ss_dim)
    for i in 1:length(ss)
        for j in i:length(ss)
            Hij = matrix_element(ss[i], ss[j], cluster_ops, clustered_ham)
            ir = (i-1)*R+1
            jr = (j-1)*R+1
            # println(ir, " ", ir+R, " ", jr, " ", jr+R)
            H[ir:ir+R-1, jr:jr+R-1] .= Hij
        end
    end

    return H
end
