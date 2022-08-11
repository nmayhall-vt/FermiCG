using JLD2
using LinearAlgebra

@load "a.jld2"
intsb = InCoreInts(0.0,A,B)
intsb = orbital_rotation(ints, U)
ints2b = subset(ints, [3,5,7,8,9])
ints3b = subset(ints, [3,5,7,8,9], A,A)

@load "out.jld2"

println(norm(intsb.h1 - ints.h1))
println(norm(intsb.h2 - ints.h2))

