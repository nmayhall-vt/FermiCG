"""
Simple string-based CI code
"""
module StringCI
using Arpack
using TimerOutputs
using TensorOperations

include("Helpers.jl")
include("DeterminantStrings.jl")
include("FCI.jl")

N = 30
binom_coeff = Array{Int,2}(undef,N+1,N+1)
for i in 0:N
    for j in i:N
        binom_coeff[j+1,i+1] = calc_nchk(j,i)
    end
end

end

