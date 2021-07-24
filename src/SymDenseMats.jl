using LinearAlgebra 

"""
Symmetric Matrix which only stores upper triangle

data::Vector{T}
"""
struct SymDenseMat{T} <: AbstractMatrix{T}
    data::Vector{T}
    N::Int
end

"""
    function vec_tril(M::AbstractMatrix{T}) where T

Export Symmetri
"""
function SymDenseMat{T}(M::AbstractMatrix{T}) where T
    m, n = size(M)
    m == n || throw(error("not square"))
    nv = n*(n+1) ÷ 2
    v = SymDenseMat{T}(Vector{T}(undef,nv), n)
    k = 0
    for i in 1:n
        for j in i:n
            k += 1
            v.data[k] = M[j, i]
        end
    end
    return v
end

function Base.Matrix(A::SymDenseMat{T}) where {T}
    M = Matrix{T}(undef,A.N,A.N)
    k = 1
    for i in 1:A.N
        for j in i:A.N
            M[i,j] = A.data[k]
            M[j,i] = A.data[k]
            k += 1
        end
    end
    return M
end
Base.size(A::SymDenseMat{T}) where {T} = return (A.N, A.N)
function Base.getindex(A::SymDenseMat{T}, i::Integer, j::Integer) where {T} 
    #i>=j || error("i<j")
    if i<j
        return A.data[A.N*(A.N+1)÷2 - (A.N-i+1)*(A.N-i+2)÷2 + j - i + 1] 
    else
        return A.data[A.N*(A.N+1)÷2 - (A.N-j+1)*(A.N-j+2)÷2 + i - j + 1] 
    end
end
function Base.display(A::SymDenseMat{T}) where {T} 
    display(typeof(A))
    display(("N=",A.N))
    display(A.data)
end
function Base.:(*)(A::SymDenseMat{T}, v::AbstractVector{T}) where {T}
    A.N == length(v)  || throw(DimensionMismatch)
    out = zeros(T,A.N)
    for i in 1:A.N
        for j in 1:A.N
            out[i] += A[i,j]*v[j]
        end
    end
    return out
end

