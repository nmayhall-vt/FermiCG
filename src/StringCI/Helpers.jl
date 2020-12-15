"""
    calc_nchk(n::Integer,k::Integer)

Calculates binomial coefficient: n choose k
"""
function calc_nchk(n::Int,k::Int)
    accum::Int = 1
    for i in 1:k
        accum = accum * (n-k+i) ÷ i
    end
    return accum
end


"""
    get_nchk(n::Integer,k::Integer)

Looks up binomial coefficient from a precomputed table: n choose k
"""
@inline function get_nchk(n::Int,k::Int)
    return binom_coeff[n+1,k+1]
end
