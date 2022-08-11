using JLD2
using LinearAlgebra
if false 
#if true 
    N = 20
    A = rand(N,N)
    B = rand(N,N,N,N)
    U,_,V = svd(A)
    U = U * V
    @save "a.jld2" A B U
else

    @load "a.jld2"
    ints = InCoreInts(0.0,A,B)
    ints = orbital_rotation(ints, U)
    ints2 = subset(ints, [3,5,7,8,9])
    ints3 = subset(ints, [3,5,7,8,9], A,A)

    @save "out.jld2" ints ints2 ints3
end

