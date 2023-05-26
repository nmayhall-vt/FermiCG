using BenchmarkTools
tuckA = FermiCG.Tucker(rand(12,11,1,1,1,10,1,1,3,10,2,1,1,10,1,1,1,1,1,1));
tuckB = FermiCG.Tucker(rand(12,11,1,1,1,10,1,1,3,10,2,1,1,10,1,1,1,1,1,1));

# tuckA = FermiCG.compress(tuckA)
# tuckB = FermiCG.compress(tuckB)
scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(tuckA)]);
@timev FermiCG.nonorth_add([tuckA, tuckB]);
@timev FermiCG.nonorth_add([tuckA, tuckB],scr);
@btime FermiCG.nonorth_add([tuckA, tuckB]);
@btime FermiCG.nonorth_add([tuckA, tuckB],scr);