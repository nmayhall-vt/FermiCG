using LinearAlgebra
using FermiCG
using Printf
using Test
using LinearMaps
using Arpack
using Random
using Profile 


@testset "symm" begin

    N = 1000 
    Nv = N*(N+1)÷2
    println(N)
    println(Nv)
    A = Diagonal(rand(N)) .- .5 + 1.0*rand(N,N)
    A = A'+A
    Hv = FermiCG.SymDenseMat{Float64}(A)

    Hm = Matrix(Hv)
    v = rand(Float64,N)

    r1 = A*v
    r2 = Hv*v
    r3 = Hm*v

    @test all(isapprox(r1,r2, atol=1e-10))
    @test all(isapprox(r1,r3, atol=1e-10))

    
    nr = 2
    
    e1,v1 = Arpack.eigs(A, nev = nr, which=:SR)
    e2,v2 = Arpack.eigs(Hv, nev = nr, which=:SR)

    @test all(isapprox(e1,e2, atol=1e-10))
end
