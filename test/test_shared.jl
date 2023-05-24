using BenchmarkTools
tuckA = FermiCG.Tucker(rand(12,11,10,9,12,3));
tuckB = FermiCG.Tucker(rand(12,11,10,9,12,3));
scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(tuckA)]);

@btime FermiCG.nonorth_add([tuckA, tuckB])
@btime FermiCG.nonorth_add([tuckA, tuckB],scr)