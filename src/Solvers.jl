using LinearAlgebra 
using Printf
using BenchmarkTools

mutable struct LinOp
    matvec
    dim::Int
end

Base.size(lop::LinOp) = return (lop.dim,lop.dim)
Base.:(*)(lop::LinOp, v::AbstractVector{T}) where {T} = return lop.matvec(v)
Base.:(*)(lop::LinOp, v::AbstractMatrix{T}) where {T} = return lop.matvec(v)

    

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
end

function Davidson(op; max_iter=100, max_ss_vecs=8, tol=1e-8, nroots=1, v0=nothing)
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
                    1.0)
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
    ss_size = length(F.values)
    if ss_size >= solver.max_ss_vecs
        ritz_v = ritz_v[:,sortperm(ritz_e)][:,1:solver.nroots]
        ritz_e = ritz_e[sortperm(ritz_e)][1:1:solver.nroots]
    else
        ritz_v = ritz_v[:,sortperm(ritz_e)]
        ritz_e = ritz_e[sortperm(ritz_e)]
    end
    solver.ritz_e = ritz_e
    solver.ritz_v = ritz_v

    solver.sig_prev = solver.sig_prev * ritz_v 
    solver.vec_prev = solver.vec_prev * ritz_v
    Hss1 = solver.vec_prev' * solver.sig_prev
    Hss = ritz_v' * Hss * ritz_v
    # make sure these match
    all([abs(Hss1[i]-Hss[i])<1e-12 for i in 1:Base.length(Hss)]) || throw(Exception)

    solver.lindep = det(solver.vec_prev'*solver.vec_prev)
    #Hss = solver.vec_curr' * sigma
    if iprint>0
        [@printf(" Ritz Value: %12.8f \n",i) for i in ritz_e]
    end
    res = similar(solver.vec_prev[:,1:solver.nroots])
    for s in 1:solver.nroots
        res[:,s] .= -solver.vec_prev[:,s] * Hss[s,s]
    end
    res[:,:] .+= solver.sig_prev[:,1:solver.nroots]
    
    #@btime $res = - $solver.vec_prev * $Hss[1:$solver.nroots,1:$solver.nroots]
    #@btime $res .+= $solver.sig_prev
    #res = solver.sig_prev - (solver.vec_prev * Hss)
    #@btime $res = $solver.sig_prev - ($solver.vec_prev * $Hss)
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

            denom = (ritz_e[s]  + level_shift)
            _apply_diagonal_precond!(res[:,s], Adiag, denom)
            #@btime $_apply_diagonal_precond!($res[:,$s], $Adiag, $denom)
            #@inbounds @simd for i in 1:solver.dim
            #    res_s[i] = res_s[i] / (denom - Adiag[i])
            #end
        end
        #scr = solver.vec_prev' * res[:,s]
        #res[:,s] = res[:,s] - (solver.vec_prev * scr)
        scr = ss' * res[:,s]
        res[:,s] = res[:,s] - (ss * scr)
        nres = norm(res[:,s])
        if nres>1e-12
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
    end
    return solver.ritz_e[1:solver.nroots], solver.vec_prev[:,1:solver.nroots]
    #return solver.ritz_e[1:solver.nroots], solver.vec_prev*solver.ritz_v[:,1:solver.nroots]
end

