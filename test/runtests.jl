using FermiCG
using Test
using Random

Random.seed!(1234567)

@testset "FermiCG.jl" begin
    include("test_hamiltonians.jl")
end
