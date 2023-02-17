using FermiCG
using ActiveSpaceSolvers

using LinearAlgebra
using Printf
using Test
using JLD2

@testset "S2" begin
    @load "_testdata_cmf_h12_64bit.jld2"
    
    norb = size(ints.h1)[1]
    ansatz = FCIAnsatz(norb, 1,2)


    display(ansatz)

    @time Hmat = build_H_matrix(ints, ansatz)
    @time Ssq = build_S2_matrix(ansatz)

    print(size(Hmat))
    print(size(Ssq))

    println(" Norm of [H,S2]: ", norm(Ssq*Hmat - Hmat*Ssq))
    @test isapprox(norm(Ssq*Hmat - Hmat*Ssq), 0.0, atol=1e-12)

    F = eigen(Hmat+0.01*Ssq)
    e = F.values
    v = F.vectors
    e = diag(v'*Hmat*v)
    ss = diag(v'*Ssq*v)

    e = real(e[1:10])
    for (i,ei) in enumerate(e)
        @printf(" Energy: %12.8f   <S2>: %12.8f\n",ei+ints.h0,ss[i])
    end
    ref = [
           0.75000000
           0.75000000
           3.75000000
           0.75000000
           0.75000000
           3.75000000
           0.75000000
           0.75000000
           0.75000000
           0.75000000
          ]

    @test isapprox(ref, ss[1:10], atol=1e-12)
end

