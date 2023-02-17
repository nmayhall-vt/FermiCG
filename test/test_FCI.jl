using LinearAlgebra
using FermiCG
using Printf
using Test
using Arpack
using ActiveSpaceSolvers

# This is really just checking to make sure we've got ActiveSpaceSolvers connected correctly

@testset "fci" begin
    @load "_testdata_hf_h6.jld2" ints clusters init_fspace e_fci
    n_elec_a = 1
    n_elec_b = 1

    norb = size(ints.h1)[1]
    problem = FCIAnsatz(norb, n_elec_a, n_elec_b)


    display(problem)

    @time Hmat = build_H_matrix(ints, problem)
    @time e,v = Arpack.eigs(Hmat, nev = 10, which=:SR)
    e = real(e)
    for ei in e
        @printf(" Energy: %12.8f\n",ei)
    end
    @test isapprox(e[1], e_fci+ints.h0 , atol=1e-10)
end

