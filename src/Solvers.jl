using LinearAlgebra 
using Printf

mutable struct Davidson
    op 
    dim::Int
    nroots::Int
    max_iter::Int
    tol::Float64
    converged::Bool
    status::Vector{Bool}        # converged status of each root 
    iter_curr::Int
    vec_curr::Array{Float64,2}
    sig_curr::Array{Float64,2}
    vec_prev::Array{Float64,2}
    sig_prev::Array{Float64,2}
    ritz_v::Array{Float64,2}
    ritz_e::Vector{Float64}
    resid::Vector{Float64}
end

function Davidson(op; max_iter=100, tol=1e-8, nroots=1, v0=nothing)
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
                    tol,
                    false, 
                    [false for i in 1:nroots],
                    0,
                    v0,
                    v0,
                    zeros(dim,0),
                    zeros(dim,0),
                    zeros(nroots,nroots),
                    zeros(nroots),
                    zeros(nroots))
end

function print_iter(solver::Davidson)
    @printf(" Iter: %3i", solver.iter_curr)
    @printf(" E: ")
    for i in 1:solver.nroots
        if solver.status[i]
            @printf("%13.8f* ", solver.ritz_e[i])
        else
            @printf("%13.8f  ", solver.ritz_e[i])
        end
    end
    @printf(" R: ")
    for i in 1:solver.nroots
        if solver.status[i]
            @printf("%5.1e* ", solver.resid[i])
        else
            @printf("%5.1e  ", solver.resid[i])
        end
    end
    println("")
end

function solve(solver::Davidson; print=0, Adiag=nothing)
    for iter = 1:solver.max_iter
        solver.iter_curr = iter
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
        #ritz_e = F.values[1:solver.nroots]
        #ritz_v = F.vectors[:,1:solver.nroots]
        ritz_e = F.values
        ritz_v = F.vectors
        solver.ritz_e = ritz_e
        solver.ritz_v = ritz_v

        solver.sig_prev = solver.sig_prev * ritz_v 
        solver.vec_prev = solver.vec_prev * ritz_v
        Hss1 = solver.vec_prev' * solver.sig_prev
        Hss = ritz_v' * Hss * ritz_v
        # make sure these match
        all([abs(Hss1[i]-Hss[i])<1e-12 for i in 1:Base.length(Hss)]) || throw(Exception)
        
        if print>0
            println(" Determinant of subspace overlap: ",det(solver.vec_prev'*solver.vec_prev))
        end
        #Hss = solver.vec_curr' * sigma
        if print>0
            [@printf(" Ritz Value: %12.8f \n",i) for i in ritz_e]
        end
        res = solver.sig_prev - solver.vec_prev * Hss
        ss = deepcopy(solver.vec_prev)
        new = zeros(solver.dim,0) 
        #solver.statusconv = [false for i in 1:solver.nroots]
        for s in 1:solver.nroots
            # |v'> = (I-|SS><SS|)|v>
            #      = |v> - |SS><SS|V>
            solver.resid[s] = norm(res[:,s])
            if norm(res[:,s]) <= solver.tol
                solver.status[s] = true
                continue
            end
            if Adiag != nothing
                level_shift = 1e-3
                for i in 1:solver.dim
                    res[i,s] = res[i,s] / (ritz_e[s] - Adiag[i] + level_shift)
                end
            end
            res[:,s] = res[:,s] - ss * ss' * res[:,s]
            nres = norm(res[:,s])
            if nres>1e-12
                ss = hcat(ss,res[:,s]/nres)
                new = hcat(new,res[:,s]/nres)
            end
        end
        #display(solver.vec_prev'*solver.vec_prev)
        print_iter(solver)
        if all(solver.status)
            solver.converged == true
            break
        end
        solver.vec_curr = new 
    end
end

