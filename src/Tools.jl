function calc_nchk(n::Int,k::Int)
    #= 
    Calculate n choose k
    =#
    
    #@myassert(n>=k)
    accum::BigInt = 1
    for i in 1:k
        accum = accum * (n-k+i) ÷ i
    end
    return accum
end
