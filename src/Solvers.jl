using LinearAlgebra 
using Printf

mutable struct Davidson
    op 
    dim::Int
    nroots::Int
    max_iter::Int
    converged::Bool
    vec_curr::Array{Float64,2}
    sig_curr::Array{Float64,2}
    vec_prev::Array{Float64,2}
    sig_prev::Array{Float64,2}
end

function Davidson(op; max_iter=100, nroots=1, v0=nothing)
    size(op)[1] == size(op)[2] || throw(DimensionError())
    dim = size(op)[1]
    if v0==nothing
        v0 = rand(dim,nroots)
        S = v0'*v0
        v0 = v0*inv(sqrt(S))
    end
    #display(v0'*v0)
    return Davidson(op, 
                    dim, 
                    nroots, 
                    max_iter, 
                    false, 
                    v0,
                    v0,
                    zeros(dim,0),
                    zeros(dim,0))
end


function solve(solver::Davidson, print=0)

    for iter = 1:solver.max_iter
        #@printf(" Davidson Iter: %4i SS_Curr %4i\n", iter, size(solver.vec_curr,2))
        #
        # perform sig_curr = A*vec_curr 
        solver.sig_curr = Matrix(solver.op * solver.vec_curr)
        #
        # add these new vectors to previous quantities
        solver.sig_prev = hcat(solver.sig_prev, solver.sig_curr)
        solver.vec_prev = hcat(solver.vec_prev, solver.vec_curr)
        #
        # form H in current subspace
        Hss = solver.vec_prev' * solver.sig_prev
        F = eigen(Hss)
        ritz_e = F.values[1:solver.nroots]
        ritz_v = F.vectors[:,1:solver.nroots]
        solver.sig_prev = solver.sig_prev * ritz_v 
        solver.vec_prev = solver.vec_prev * ritz_v
        Hss = ritz_v' * Hss * ritz_v
        if print>0
            println(" Determinant of subspace overlap: ",det(solver.vec_prev'*solver.vec_prev))
        end
        #Hss = solver.vec_curr' * sigma
        if print>0
            [@printf(" Ritz Value: %12.8f \n",i) for i in ritz_e]
        end
        res = solver.sig_prev - solver.vec_prev * Hss
        ss = copy(solver.vec_prev)
        new = zeros(solver.dim,0) 
        for i in 1:size(res,2)
            # |v'> = (I-|SS><SS|)|v>
            #      = |v> - |SS><SS|V>
            res[:,i] = res[:,i] - ss * ss' * res[:,i]
            #println(" Res Norm:", norm(res[:,i]))
            nres = norm(res[:,i])
            if nres>1e-7
                ss = hcat(ss,res[:,i]/nres)
                new = hcat(new,res[:,i]/nres)
            end
        end
        @printf(" Iter: %3i", iter)
        @printf(" E: ")
        for i in 1:solver.nroots
            @printf("%12.8f ", Hss[i,i])
        end
        @printf(" R: ")
        for i in 1:solver.nroots
            @printf("%5.1e ", norm(res[:,i]))
        end
        println("")
        #display(res'*res)
        #display(ss'*ss)
        #res = res*inv(sqrt(res'*res))
        solver.vec_curr = new 
        #display(solver.vec_curr' * solver.vec_prev)
    end
end

