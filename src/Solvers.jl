using LinearAlgebra 
using Printf
using BenchmarkTools

mutable struct LinOp
    matvec
    dim::Int
    sym::Bool
end

Base.size(lop::LinOp) = return (lop.dim,lop.dim)
Base.:(*)(lop::LinOp, v::AbstractVector{T}) where {T} = return lop.matvec(v)
Base.:(*)(lop::LinOp, v::AbstractMatrix{T}) where {T} = return lop.matvec(v)
issymmetric(lop::LinOp) = return lop.sym

mutable struct LinOpMat{T} <: AbstractMatrix{T} 
    matvec
    dim::Int
    sym::Bool
end

Base.size(lop::LinOpMat{T}) where {T} = return (lop.dim,lop.dim)
Base.:(*)(lop::LinOpMat{T}, v::AbstractVector{T}) where {T} = return lop.matvec(v)
Base.:(*)(lop::LinOpMat{T}, v::AbstractMatrix{T}) where {T} = return lop.matvec(v)
issymmetric(lop::LinOpMat{T}) where {T} = return lop.sym
    

mutable struct Davidson
    op 
    dim::Int
    nroots::Int
    max_iter::Int
    max_ss_vecs::Int
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
    lindep::Float64
    lindep_thresh::Float64
end

function Davidson(op; max_iter=100, max_ss_vecs=8, tol=1e-8, nroots=1, v0=nothing, lindep_thresh=1e-10)
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
                    max_ss_vecs*nroots,
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
                    zeros(nroots),
                    1.0,
                    lindep_thresh)
end

function print_iter(solver::Davidson)
    @printf(" Iter: %3i SS: %-4i", solver.iter_curr, size(solver.vec_prev)[2])
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
    @printf(" LinDep: ")
    @printf("%5.1e* ", solver.lindep)
    println("")
    flush(stdout)
end


function _apply_diagonal_precond!(res_s::Vector{Float64}, Adiag::Vector{Float64}, denom::Float64)
    dim = size(Adiag)[1]
    length(size(Adiag)) == 1 || throw(Exception)
    length(size(res_s)) == 1 || throw(Exception)
    @inbounds @simd for i in 1:dim
        res_s[i] = res_s[i] / (denom - Adiag[i])
    end
end

function iteration(solver::Davidson; Adiag=nothing, iprint=0)
    
    #
    # perform sig_curr = A*vec_curr 
    solver.sig_curr = Matrix(solver.op * solver.vec_curr)
   
    #
    # add these new vectors to previous quantities
    solver.sig_prev = hcat(solver.sig_prev, solver.sig_curr)
    solver.vec_prev = hcat(solver.vec_prev, solver.vec_curr)
   
    #
    # Check orthogonality
    ss_metric = solver.vec_prev'*solver.vec_prev
    solver.lindep = abs(1.0 - det(ss_metric))

    #
    # form H in current subspace
    Hss = solver.vec_prev' * solver.sig_prev
    F = eigen(Hss)
    #ritz_e = F.values[1:solver.nroots]
    #ritz_v = F.vectors[:,1:solver.nroots]
    ritz_e = F.values
    ritz_v = F.vectors
    ss_size = length(F.values)
    if ss_size >= solver.max_ss_vecs
        #ritz_v = ritz_v[:,sortperm(ritz_e)][:,1:solver.nroots]
        #ritz_e = ritz_e[sortperm(ritz_e)][1:1:solver.nroots]
        ritz_v = ritz_v[:,sortperm(ritz_e)][:,1:solver.max_ss_vecs]
        ritz_e = ritz_e[sortperm(ritz_e)][1:1:solver.max_ss_vecs]
    else
        ritz_v = ritz_v[:,sortperm(ritz_e)]
        ritz_e = ritz_e[sortperm(ritz_e)]
    end

    solver.ritz_e = ritz_e
    solver.ritz_v = ritz_v

    solver.sig_prev = solver.sig_prev * ritz_v 
    solver.vec_prev = solver.vec_prev * ritz_v
    Hss = ritz_v' * Hss * ritz_v

    res = deepcopy(solver.sig_prev[:,1:solver.nroots])
    for s in 1:solver.nroots
        res[:,s] .-= solver.vec_prev[:,s] * Hss[s,s]
    end

    
    ss = deepcopy(solver.vec_prev)

    new = zeros(solver.dim, 0) 
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
            denom = (ritz_e[s]  + level_shift)
            _apply_diagonal_precond!(res[:,s], Adiag, denom)
        end
        #scr = solver.vec_prev' * res[:,s]
        #res[:,s] = res[:,s] - (solver.vec_prev * scr)
        scr = ss' * res[:,s]
        res[:,s] .= res[:,s] .- (ss * scr)
        nres = norm(res[:,s])
        if nres > solver.tol
            ss = hcat(ss,res[:,s]/nres)
            new = hcat(new,res[:,s]/nres)
        end
    end
    return new
end

function solve(solver::Davidson; Adiag=nothing, iprint=0)

    for iter = 1:solver.max_iter
        #@btime $solver.vec_curr = $iteration($solver)
        solver.vec_curr = iteration(solver, Adiag=Adiag, iprint=iprint)
        solver.iter_curr = iter
        print_iter(solver)
        if all(solver.status)
            solver.converged == true
            break
        end
        if solver.lindep > solver.lindep_thresh
            @warn "Linear dependency detected. Restarting."
            @time F = qr(solver.vec_prev[:,1:solver.nroots])
            solver.vec_curr = Array(F.Q)
            solver.sig_curr = Array(F.Q)
            solver.vec_prev = zeros(solver.dim, 0) 
            solver.sig_prev =  zeros(solver.dim, 0)
            solver.ritz_v = zeros(solver.nroots,solver.nroots)
            solver.ritz_e = zeros(solver.nroots)
            solver.resid = zeros(solver.nroots)
        end
    end
    return solver.ritz_e[1:solver.nroots], solver.vec_prev[:,1:solver.nroots]
    #return solver.fritz_e[1:solver.nroots], solver.vec_prev*solver.ritz_v[:,1:solver.nroots]
end

#function orthogonalize!(solver)
#
#    # |vv_i>  = |v_i> - |w
#    new_v = zeros(solver.dim, nroots)
#    for r in 1:nroots
#end
