using NPZ
using JSON
using Random
using LinearAlgebra
using FermiCG

@testset "hamiltonian stuff" begin
    testdir = joinpath(dirname(pathof(FermiCG)), "..", "test")

    filepath = joinpath(testdir, "data_h4/ints_0b.npy")
    ints_0b = npzread(filepath)

    filepath = joinpath(testdir, "data_h4/ints_1b.npy")
    ints_1b = npzread(filepath)

    filepath = joinpath(testdir, "data_h4/ints_2b.npy")

    ints_2b = npzread(filepath)

    filepath = joinpath(testdir, "data_h4/problem.json")

    data = JSON.parsefile(filepath)
    spin = data["spin"]
    n_elec = data["n_elec"]
    n_elec_a = round(Int,(spin - n_elec)/2)
    n_elec_b = n_elec - n_elec_a

    ham 	= ElectronicInts(ints_0b, ints_1b, ints_2b)

    # test square
    U = rand(Float64,size(ham.h1))
    display(U)
    F = svd(U)

    @test all(F.U*Diagonal(F.S)*F.Vt .≈ U)
    U = F.U * F.Vt
    new_ham = FermiCG.orbital_rotation(ham,U)
    new_ham = FermiCG.orbital_rotation(new_ham,U')
    @test all(isapprox(ham.h0,new_ham.h0, atol=1e-12))
    @test all(isapprox(ham.h1,new_ham.h1, atol=1e-12))
    @test all(isapprox(ham.h2,new_ham.h2, atol=1e-12))
    display(ham.h1)
    display(new_ham.h1)

    FermiCG.orbital_rotation!(ham,U)
    new_ham = FermiCG.orbital_rotation(ham,U)
    @test all(isapprox(ham.h0,new_ham.h0, atol=1e-12))
    @test all(isapprox(ham.h1,new_ham.h1, atol=1e-12))
    @test all(isapprox(ham.h2,new_ham.h2, atol=1e-12))

end
