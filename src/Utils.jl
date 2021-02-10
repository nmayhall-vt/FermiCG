using TensorOperations
using Printf

"""
Tucker Decomposition of dense tensor: 
A ~ X *(1) U1 *(2) U2 ....
"""
function tucker_decompose(A::Array{T,N}; thresh=1e-7, max_number=nothing, verbose=1) where {T,N}
    factors = Vector{Matrix{T}}()
    if verbose > 0
        println(" Tucker Decompose:", size(A))
    end
    for i in 1:ndims(A)
        idx_l = collect(1:ndims(A))
        idx_r = collect(1:ndims(A))
        idx_l[i] = -1
        idx_r[i] = -2
        G = tensorcontract(A,idx_l,A,idx_r)
        #G = @ncon([A, A], [idx_l, idx_r])
        F = eigen(G)
        perm = sortperm(F.values, rev=true)
        l = F.values[perm]
        v = F.vectors[:,perm]

        nkeep = 0
        if verbose > 0
            @printf(" index dimension: %6i\n", size(A)[i])
        end
        for li in l
            if verbose > 0
                @printf(" Singular value = %12.8f\n", li)
            end
            if abs(li) > thresh
                nkeep += 1
            end
        end
        if max_number != nothing
            nkeep = min(nkeep, max_number)
        end

        push!(factors, v[:,1:nkeep])
    end
    core = copy(A)
    for i in 1:ndims(A)
        dims1 = size(core)
        core = reshape(core, dims1[1], prod(dims1[2:end]))
        core = factors[i]' * core
        
        dimi = size(factors[i])[2]
        core = reshape(core, dimi, dims1[2:end]...)
        core = permutedims(core, [collect(2:ndims(core))..., 1])
    end
    return core, factors
end

"""
Recompose Tucker Decomposition 
"""
function tucker_recompose(core, factors)
    A = copy(core)
    for i in 1:ndims(A)
        dims1 = size(A)
        A = reshape(A, dims1[1], prod(dims1[2:end]))
        A = factors[i] * A
        
        dimi = size(factors[i])[1]
        A = reshape(A, dimi, dims1[2:end]...)
        A = permutedims(A, [collect(2:ndims(A))..., 1])
    end
    return A
end
