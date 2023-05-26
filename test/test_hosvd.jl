using FermiCG
using Printf
using Test
using Random
using LinearAlgebra 

@testset "hosvd" begin
    atoms = []
    clusters = []
    na = 0
    nb = 0
    init_fspace = []
    
    function generate_H_ring(n,radius)
        theta = 2*pi/n

        atoms = []
        for i in 0:n-1
            push!(atoms,Atom(i+1,"H",[radius*cos(theta*i), radius*sin(theta*i), 0]))
        end
        return atoms
    end

    #
    # Test basic Tucker stuff
    Random.seed!(2);
    A = rand(4,6,3,3,5)
    #tuck = FermiCG.Tucker(A, thresh=20, verbose=1)
    tuck = FermiCG.Tucker((A,), thresh=10, verbose=1)
    
    display(size.(tuck.core))
    display(size.(tuck.factors))
    B = FermiCG.recompose(tuck)
    println()
    println(FermiCG.dims_small(tuck))
    println(FermiCG.dims_large(tuck))
    @test all(FermiCG.dims_small(tuck) .== [1, 1, 1, 1, 1])
    @test all(FermiCG.dims_large(tuck) .== [4, 6, 3, 3, 5])
     
    A = rand(4,6,3,3,5)
    tuck = FermiCG.Tucker(A, thresh=-1, verbose=1)
    B = FermiCG.recompose(tuck)
    @test isapprox(abs.(A), abs.(B[1]), atol=1e-12)

    A = rand(4,6,3,3,5)*.1
    B = rand(4,6,3,3,5)*.1
    C = A+B


    #tuckA = FermiCG.Tucker(A, thresh=-1, verbose=1, max_number=2)
    #tuckB = FermiCG.Tucker(B, thresh=-1, verbose=1, max_number=2)
    #tuckC = FermiCG.Tucker(C, thresh=-1, verbose=1, max_number=2)
    tuckA = FermiCG.Tucker(A, thresh=-1, verbose=0)
    tuckB = FermiCG.Tucker(B, thresh=-1, verbose=0)
    tuckC = FermiCG.Tucker(C, thresh=-1, verbose=0)

    tuckD = FermiCG.Tucker(tuckC, R=3)
    FermiCG.randomize!(tuckD)

    tuckDc = FermiCG.compress(tuckD)

    # test Tucker addition
    test = FermiCG.nonorth_add(tuckA, tuckB)
    @test isapprox(FermiCG.orth_dot(tuckC,tuckC), FermiCG.orth_dot(test,test), atol=1e-12)
    @test isapprox(FermiCG.nonorth_dot(tuckC,tuckC), FermiCG.nonorth_dot(test,test), atol=1e-12)


    #
    # Now test basis transformation
    A = rand(4,6,3,3,5)*.1
    
    trans1 = Dict{Int,Matrix{Float64}}() 
    trans1[2] = rand(6,5)
    trans1[4] = rand(3,2)

    trans2 = Vector{Matrix{Float64}}([])
    for i = 1:5
        if haskey(trans1, i)
            push!(trans2, trans1[i])
        else
            push!(trans2, Matrix(1.0I,size(A,i),size(A,i)))
        end
    end

    display((length(A), size(A)))
    
    A1 = FermiCG.transform_basis(A, trans1)
    display((length(A1), size(A1)))
    
    A2 = FermiCG.transform_basis(A, trans2)
    display((length(A1), size(A2)))
    
    #A2 = FermiCG.tucker_recompose(A, trans2)
    #display((length(A2), size(A2)))
    
  

    scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(A)])
    # A3 = FermiCG.transform_basis(A, trans2, scr)
    # display((length(A3), size(A3)))

    # @test norm(A3-A2) < 1e-16
    
   
    # # Now test transpose
    # trans2 = [Matrix(t') for t in trans2]
    # println("old")
    # @timev A2 = FermiCG.transform_basis(A, trans2, trans=true)
    # println("new")
    # @timev A3 = FermiCG.transform_basis(A, trans2, scr, trans=true)
    # # println("newnew")
    # # @timev A4 = FermiCG.transform_basis(A, trans2, scr[1], scr[2], trans=true)
    
    # println(norm(A2-A3))
    # # println(norm(A2-A4))
    # @test norm(A3-A2) < 1e-16
    # # @test norm(A4-A2) < 1e-16

    tuckA = FermiCG.Tucker(rand(2,3,4,5,6,8))
    tuckB = FermiCG.Tucker(rand(2,3,4,5,6,8))
    tuckA = FermiCG.compress(tuckA)
    tuckB = FermiCG.compress(tuckB)
    @timev tuckC = FermiCG.nonorth_add([tuckA, tuckB, tuckB, tuckA])
    
    scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(tuckA)])
    @timev tuckD = FermiCG.nonorth_add([tuckA, tuckB, tuckB, tuckA], scr)
    println(norm(tuckC))
    println(norm(tuckD))
    @test abs(norm(tuckC) - norm(tuckD)) < 1e-12
end

