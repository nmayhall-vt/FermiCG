module Helpers

export get_nchk, calc_nchk 

function calc_nchk(n::Integer,k::Integer)
    #= 
    Calculate n choose k
    =#
    #={{{=#
    #@myassert(n>=k)
    accum::Integer = 1
    for i in 1:k
        accum = accum * (n-k+i) ÷ i
    end
    return accum
end
#=}}}=#


N = 30
binom_coeff = Array{Integer,2}(undef,N+1,N+1)
for i in 0:N
    for j in i:N
        binom_coeff[j+1,i+1] = calc_nchk(j,i)
    end
end

function get_nchk(n::Integer,k::Integer)
    return binom_coeff[n+1,k+1]
end
end
