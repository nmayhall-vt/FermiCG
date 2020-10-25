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


function build_nchk_table(N)
	binom_coeff = Array{BigInt,2}(undef,N+1,N+1)
	for i in 0:N
    		for j in i:N
        		binom_coeff[j+1,i+1] = calc_nchk(j,i)
    		end
    	end
	return binom_coeff
end
