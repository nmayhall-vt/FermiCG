using JuliaTPSCI
using Printf
using NPZ

#include("../src/string_ci/ConfigStrings.jl")
function test_a(x::Int)
	print(x)
end

test_a(1)

ket_a = ConfigString(no=4, ne=4)

ints_0b = npzread("../src/python/data/ints_0b.npy")
ints_1b = npzread("../src/python/data/ints_1b.npy")
ints_2b = npzread("../src/python/data/ints_2b.npy")

# ham = ElecInts_InCore(ints_0b, ints_1b, ints_2b)
#problem = FCI.Problem(no=size(ints_1b,1), na=n_elec_a, nb=n_elec_b)
