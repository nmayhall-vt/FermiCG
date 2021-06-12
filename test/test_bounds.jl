using FermiCG
using Printf
using Test
using LinearAlgebra
using TensorOperations 


function test1()
    g1 = rand(4,10) .- .5
    g2 = rand(4,30) .- .5

    c = .1
    @tensor V[I,J] := g1[i,I] * g2[i,J] *c 

    b = FermiCG.upper_bound(g1,g2,c=c)
    maximum(abs.(V)) <= b || @printf(" %12.8f %12.8f\n", maximum(abs.(V)), b)
    return maximum(abs.(V)) <= b
end

function test2()
    v = rand(2,4)
    g1 = rand(2,10)
    g2 = rand(4,30)

    c = .1
    @tensor V[I,J] := v[i,j] * g1[i,I] * g2[j,J] * c 

    b = FermiCG.upper_bound(v,g1,g2,c=c)
    maximum(abs.(V)) <= b || @printf(" %12.8f %12.8f\n", maximum(abs.(V)), b)
    return maximum(abs.(V)) <= b
end

function test3()
    v = rand(2,4,2)
    g1 = rand(2,10)
    g2 = rand(4,30)
    g3 = rand(2,20)

    c = .1
    @tensor V[I,J,K] := v[i,j,k] * g1[i,I] * g2[j,J] * g3[k,K] * c

    b = FermiCG.upper_bound(v,g1,g2,g3,c=c)
    maximum(abs.(V)) <= b || @printf(" %12.8f %12.8f\n", maximum(abs.(V)), b)
    return maximum(abs.(V)) <= b
end

function test4()
    v = rand(2,4,2,3)
    g1 = rand(2,10).- .5
    g2 = rand(4,30).- .5
    g3 = rand(2,20).- .5
    g4 = rand(3,9) .- .5

    c = .1
    @tensor V[I,J,K,L] := v[i,j,k,l] * g1[i,I] * g2[j,J] * g3[k,K] * g4[l,L]* c

    b = FermiCG.upper_bound(v,g1,g2,g3,g4,c=c)
    maximum(abs.(V)) <= b || @printf(" %12.8f %12.8f\n", maximum(abs.(V)), b)
    return maximum(abs.(V)) <= b
end

function test4a()
    v = rand(2,4,2,3) .- .5
    g1 = rand(2,10) .- .5
    g2 = rand(4,30) .- .5
    g3 = rand(2,20) .- .5
    g4 = rand(3,9)  .- .5

    c = .1
    @tensor V[I,J,K,L] := v[i,j,k,l] * g1[i,I] * g2[j,J] * g3[k,K] * g4[l,L] * c

    thresh = 1.0
    new1, new2, new3, new4 = FermiCG.upper_bound2(v,g1,g2,g3,g4,thresh, c=c)
    g1a = g1[:,new1]
    g2a = g2[:,new2]
    g3a = g3[:,new3]
    g4a = g4[:,new4]
    
    l1 = count(V.>thresh)
    l2 = 0
    if minimum(length.([new1,new2,new3,new4])) > 0
        @tensor V2[I,J,K,L] := v[i,j,k,l] * g1a[i,I] * g2a[j,J] * g3a[k,K] * g4a[l,L] * c
        l2 = count(V2.>thresh)
    end
    return l1 == l2 
end

function test3a()
    v = rand(2,4,2) .- .5
    g1 = rand(2,10) .- .5
    g2 = rand(4,30) .- .5
    g3 = rand(2,20) .- .5

    c = .1
    @tensor V[I,J,K] := v[i,j,k] * g1[i,I] * g2[j,J] * g3[k,K] * c

    thresh = 1.0
    new1, new2, new3 = FermiCG.upper_bound2(v,g1,g2,g3,thresh, c=c)
    g1a = g1[:,new1]
    g2a = g2[:,new2]
    g3a = g3[:,new3]
    
    l1 = count(V.>thresh)
    l2 = 0
    if minimum(length.([new1,new2,new3])) > 0
        @tensor V2[I,J,K] := v[i,j,k] * g1a[i,I] * g2a[j,J] * g3a[k,K] * c
        l2 = count(V2.>thresh)
    end
    return l1 == l2 
end

function tuck_bound()
    v = FermiCG.recompose(FermiCG.Tucker(rand(6,7,8,9).-.5,max_number=4)) .+ .01*(rand(6,7,8,9) .- .5)
    v = v./norm(v)
    tuck = FermiCG.Tucker_tot(v, thresh=.1,verbose=1)
    err1 = norm(v-FermiCG.recompose(tuck)) <= .1
    
    v = (rand(6,7,8,9) .- .5)
    v = v./norm(v)
    tuck = FermiCG.Tucker_tot(v, thresh=.4,verbose=1)
    err2 = norm(v-FermiCG.recompose(tuck)) <= .4
    return err1 && err2
end

@testset "tpsci" begin
    for i in 1:10
        @test tuck_bound()
        @test test1() 
        @test test2() 
        @test test3() 
        @test test4() 
        @test test3a()
        @test test4a()
    end
end
