"""
replace this with table lookup
"""
function get_nchk(n,k)
    return calc_nchk(n,k)
end

function get_fiedler(A)

    dim = size(A)[1]
    dim > 1 || return [1.0]
    size(A)[1] == size(A)[2] || error(" a not square")
    D = zeros(size(A)[1])
    for i in 1:dim
        D[i] = sum(A[i,:])
    end
    L = Diagonal(D) - A
    F = eigen(L)
   
    @printf(" eigenvalues of laplacian\n")
    for (ii,i) in enumerate(F.values)
        @printf(" %4i = %12.8f\n", ii, i)
    end
    return F.vectors[:,2]
end

function fiedler_sort(C,K)
    fvec = get_fiedler(abs.(C'*K*C))
    perm, = bubble_sort(fvec)
    return C[:,perm]
end

function Base.:+(a::Tuple{T,T}, b::Tuple{T,T}) where T<:Integer
    return (a[1]+b[1], a[2]+b[2])
end
function Base.:-(a::Tuple{T,T}, b::Tuple{T,T}) where T<:Integer
    return (a[1]-b[1], a[2]-b[2])
end


