"""
Simple Tucker (HOSVD) type
# Data
- `core::Array{T, N}`
- `factors::NTuple{N, Matrix{T}}`

Tucker factors are stored as tall matrices
the core tensor is actually a list of core tensors, to enable multistate calculations
"""
struct Tucker{T,N,R} 
    core::NTuple{R, Array{T,N}}
    factors::NTuple{N, Matrix{T}}
    #props::Dict{Symbol, Any}
end


function Tucker(A::Array{T,N}; thresh=-1, max_number=nothing, verbose=0, type="magnitude") where {T<:Number,N}
#={{{=#
    #core,factors = tucker_decompose((A,), thresh=thresh, max_number=max_number, verbose=verbose, type=type)
    #return Tucker{T,N,1}((core,), NTuple{N}(factors))
    return Tucker((A,), thresh=thresh, max_number=max_number, verbose=verbose, type=type)
end
#=}}}=#

function Tucker(A::NTuple{R,Array{T,N}}; thresh=-1, max_number=nothing, verbose=0, type="magnitude") where {T<:Number,N,R}
    #={{{=#
    core,factors = tucker_decompose(A, thresh=thresh, max_number=max_number, verbose=verbose, type=type)
    return Tucker{T,N,R}(core, NTuple{N}(factors))
end
#=}}}=#

Tucker(A::Array{T,N},factors::NTuple{N, Matrix{T}}) where {T<:Number,N} = Tucker{T,N,1}((A,),factors)
Tucker(A::NTuple{R,Array{T,N}},factors::Array{Matrix{T}}) where {T<:Number,N,R} = Tucker{T,N,R}(A, ntuple(i->factors[i],N) )
Tucker(A::Array{Array{T,N}},factors::NTuple{N, Matrix{T}}) where {T<:Number,N} = Tucker{T,N,length(A)}(ntuple(i->A[i],length(A)),factors)

"""
    function Tucker(t::Tucker{T,N,R}, n_roots::Integer) where {T,N,R}

Create a new Tucker with a different number of core tensors; `R`=`n_roots`.
New core tensors added are initialized to zero.
"""
function Tucker(t::Tucker{TT,NN,RR}; R=RR, T=TT) where {TT,NN,RR}
    RR > 0 || error(DimensionMismatch)
    cores = ntuple(ii->zeros(T,size(t.core[1])), R)
    for rr in 1:min(R,RR)
        cores[rr] .= t.core[rr]
    end
    return Tucker{T,NN,R}(cores, t.factors)
end

recompose(t::Tucker) = tucker_recompose(t.core, t.factors)
dims_large(t::Tucker) = return [size(f,1) for f in t.factors]
dims_small(t::Tucker) = return [size(f,2) for f in t.factors]
Base.length(t::Tucker) = prod(dims_small(t))
Base.size(t::Tucker) = dims_small(t) 
Base.permutedims(t::Tucker{T,N,R}, perm) where {T,N,R} = Tucker{T,N,R}(permutedims(t.core, perm), t.factors[perm])
Base.permutedims(t::NTuple{R,Array{T,N}}, perm) where {T,N,R} = ntuple(i->permutedims(t[i], perm), R) 


function randomize!(t::Tucker{T,N,R}; seed=nothing) where {T,N,R}
    seed == nothing || Random.seed!(seed)
    for r in 1:R
        t.core[r] .= rand(T,size(t.core[r]))
    end
end

"""
    add(tucks::Vector{Tucker{T,N}}; thresh=1e-10, max_number=nothing) where {T,N}

Add together multiple Tucker instances. Assumed non-orthogonal.

# Arguments
- `tucks::Vector{Tucker{T,N}}`: Vector of Tucker objects
"""
function nonorth_add(tucks::Vector{Tucker{T,N,R}}; thresh=1e-10, max_number=nothing, type="magnitude") where {T<:Number,N,R}
#={{{=#
    # sort the Tucker objects to add. This puts them in a well-defined order for reproducibility.
    norms = norm.(tucks)
    perm = sortperm(norms,rev=true)
    tucks = tucks[perm]
    #display(norm.(tucks))
    length(tucks) > 0 ||  error("not enough Tuckers to add", length(tucks))
    length(tucks) > 1 ||  return tucks[1] 

    # first, ensure the Tuckers all correpsond to appropriately dimensioned uncompressed tensors
    dims = dims_large(first(tucks))
    for tuck in tucks
        all(dims_large(tuck) .== dims) || error("dimension error.", dims_large.(tucks))
    end

    # first, get orthogonal basis spanning all Tucker objects for each index
    new_factors = Vector{Matrix{T}}()
    for i in 1:N
        Ui = Vector{Matrix{T}}()
        for tuck in tucks
            push!(Ui, tuck.factors[i])
        end
        Ui = hcat(Ui...)
      
        F = svd(Ui)

        nkeep = 0
        if type == "magnitude"
            for si in F.S 
                if si > thresh
                    nkeep += 1
                end
            end
        elseif type == "sum"
            target = sum(F.S )
            curr = 0.0
            for si in F.S 
                if abs(curr-target) > thresh
                    nkeep += 1
                    curr += si*si
                end
            end
        else
            error("wrong type")
        end
        if max_number != nothing
            nkeep = min(nkeep, max_number)
        end
        Ui = F.U[:,1:nkeep]
        push!(new_factors, Ui)
    end


    #new_core = zeros([size(new_factors[i],2) for i in 1:N]...)
    new_core = ntuple(i->zeros([size(new_factors[i],2) for i in 1:N]...), R)

    for tuck in tucks
        transforms = Vector{Matrix{Float64}}()
        for i in 1:N 
            push!(transforms, new_factors[i]'*tuck.factors[i])
        end

        #tmp = recompose(Tucker{T,N,R}(tuck.core, Tuple(transforms)))
        #for r in 1:R
        #    new_core[r] .+= tmp[r]
        #end
        #new_core .= new_core .+ recompose(Tucker{T,N,R}(tuck.core, Tuple(transforms)))
        add!(new_core, recompose(Tucker{T,N,R}(tuck.core, Tuple(transforms))))
    end
    return Tucker{T,N,R}(new_core, Tuple(new_factors))

end
#=}}}=#

#function nonorth_add!(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}; thresh=1e-10, max_number=nothing, type="magnitude") where {T<:Number,N,R}
#    #={{{=#
#
#    all(dims_large(t1) .== dims_large(t2)) || error("dimension error.", dims_large(t1), " neq ", dims_large(t2))
#
#    # first, get orthogonal basis spanning both Tucker objects for each index
#    new_factors = Vector{Matrix{T}}()
#    for i in 1:N
#
#        F = svd(hcat(t1.factors[i],t2.factors[i]))
#
#        nkeep = 0
#        if type == "magnitude"
#            for si in F.S 
#                if si > thresh
#                    nkeep += 1
#                end
#            end
#        elseif type == "sum"
#            target = sum(F.S )
#            curr = 0.0
#            for si in F.S 
#                if abs(curr-target) > thresh
#                    nkeep += 1
#                    curr += si*si
#                end
#            end
#        else
#            error("wrong type")
#        end
#        if max_number != nothing
#            nkeep = min(nkeep, max_number)
#        end
#        push!(new_factors, F.U[:,1:nkeep])
#    end
#
#
#    #new_core = zeros([size(new_factors[i],2) for i in 1:N]...)
#    new_core = ntuple(i->zeros([size(new_factors[i],2) for i in 1:N]...), R)
#
#    for r in 1:R
#        new_core[r] .+= recompose(Tucker{T,N,R}(t1.core[r], ntuple(i->new_factors[i]'*t1.factors[i], N)))
#        new_core[r] .+= recompose(Tucker{T,N,R}(t2.core[r], ntuple(i->new_factors[i]'*t2.factors[i], N)))
#    end
#    
#    return Tucker{T,N,R}(new_core, Tuple(new_factors))
#
#end
##=}}}=#

#Base.:(+)(t1::Tucker, t2::Tucker) = nonorth_add([t1, t2])
Base.:(+)(t1::NTuple{R,Array}, t2::NTuple{R,Array}) where R = ntuple(i->t1[i].+t2[i], R)
add!(t1::NTuple{R,Array}, t2::NTuple{R,Array}) where R = ntuple(i->t1[i] .= t1[i] .+ t2[i], R)
nonorth_add(t1::Tucker, t2::Tucker; thresh=1e-7) = nonorth_add([t1,t2], thresh=thresh)

function scale(t1::Tucker, a)
    t2core = t1.core .* a
    return Tucker(t2core, t1.factors)
end
function scale!(t1::Tucker, a)
    t1.core .*= a
    return 
end


"""
    compress(t::Tucker{T,N}; thresh=1e-7, max_number=nothing) where {T,N}

Try to compress further 
"""
function compress(t::Tucker{T,N,R}; thresh=1e-7, max_number=nothing, type="magnitude") where {T,N,R}
#={{{=#
    length(t) > 0 || return t
    tt = Tucker(t.core, thresh=thresh, max_number=max_number, type=type)

    new_factors = [zeros(T,1,1) for i in 1:N]

    for i in 1:N
        new_factors[i]  = t.factors[i] * tt.factors[i]
    end

    return Tucker(tt.core, NTuple{N}(new_factors)) 
#    return Tucker(recompose(t), thresh=thresh, max_number=max_number)
end
#=}}}=#

"""
    function nonorth_overlap(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}

Note: This doesn't assume `t1` and `t2` have the same compression vectors. Returns RxR overlap matrix 
"""
function nonorth_overlap(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    
    out = zeros(T,R,R)
    overlaps = Dict{Int,Matrix{T}}()
    all(dims_large(t1) .== dims_large(t2)) || error(" t1 and t2 don't have same dimensions")
    for f in 1:N
        #push!(overlaps, t1.factors[f]' * t2.factors[f])
        overlaps[f] = t1.factors[f]' * t2.factors[f]
    end
    for ri in 1:R
        for rj in ri:R
            #display(transform_basis(t1.core[ri], overlaps))
            out[ri,rj] = sum(transform_basis(t1.core[ri], overlaps) .* t2.core[rj])
            out[rj,ri] = out[ri,rj]
        end
    end
    return out
    #return sum(tucker_recompose(t1.core, overlaps) .* t2.core)
end
#=}}}=#

"""
    function nonorth_dot(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}

Note: This doesn't assume `t1` and `t2` have the same compression vectors 
"""
function nonorth_dot(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    
    out = zeros(T,R)
    #overlaps = []
    overlaps = Dict{Int,Matrix{T}}()
    all(dims_large(t1) .== dims_large(t2)) || error(" t1 and t2 don't have same dimensions")
    for f in 1:N
        #push!(overlaps, t1.factors[f]' * t2.factors[f])
        overlaps[f] = t1.factors[f]' * t2.factors[f]
    end
    for r in 1:R
        out[r] = sum(transform_basis(t1.core[r], overlaps) .* t2.core[r])
    end
    return out
    #return sum(tucker_recompose(t1.core, overlaps) .* t2.core)
end
#=}}}=#

"""
    function nonorth_dot(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}

Note: This doesn't assume `t1` and `t2` have the same compression vectors 
"""
function nonorth_dot(t1::Tucker{T,N,R1}, t2::Tucker{T,N,R2}, r1, r2) where {T,N,R1,R2}
#={{{=#
    
    out = zero(T)
    overlaps = Dict{Int,Matrix{T}}()
    all(dims_large(t1) .== dims_large(t2)) || error(" t1 and t2 don't have same dimensions")
    for f in 1:N
        overlaps[f] = t1.factors[f]' * t2.factors[f]
    end
    return sum(transform_basis(t1.core[r1], overlaps) .* t2.core[r2])
end
#=}}}=#

"""
    function orth_dot(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}

Note: This assumes `t1` and `t2` have the same compression vectors 
"""
function orth_dot(t1::Tucker{T,N,R}, t2::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    
    out = zeros(T,R)
    all(dims_large(t1) .== dims_large(t2)) || error(" t1 and t2 don't have same dimensions")
    for r in 1:R
        out[r] = sum(t1.core[r] .* t2.core[r])
    end
    return out
end
#=}}}=#


"""
    function tucker_decompose(A::Array{T,N}; thresh=1e-7, max_number=nothing, verbose=1, type="magnitude") where {T,N}

Tucker Decomposition of dense tensor: 
    A ~ X *(1) U1 *(2) U2 ....
where cluster states are discarded based on the corresponding SVD
#Arguments
- `A`: matrix to decompose
- `thresh`: threshold for discarding tucker factors
- `max_number`: limit number of tucker factors to this value
- `type`: type of trunctation. "magnitude" discards values smaller than this number. 
    "sum" discards values such that the sum of discarded values is smaller than `thresh`.
"""
function tucker_decompose(Av::NTuple{R,Array{T,N}}; thresh=1e-7, max_number=nothing, verbose=1, type="magnitude") where {T,N,R}
#={{{=#
    #R = length(Av)
    
    length(Av) > 0 || error(DimensionMismatch)
    dims = size(Av[1])
    
    factors = [zeros(T,size(Av[1],i),0) for i in 1:length(dims)]
    for r in 1:R
        all(dims .== size(Av[r])) || error(DimensionMismatch)
    end
    verbose <= 0 || println(" Tucker Decompose:", size(Av[1]))

    for i in 1:N
        idx = collect(1:N)
        idx[i] = -1
        perm = sortperm(idx)

        tmp = reshape(permutedims(Av[1],perm), size(Av[1],i), length(Av[1])÷size(Av[1],i))
        G = Symmetric(tmp*tmp')
        for r in 2:R
            tmp = reshape(permutedims(Av[r],perm), size(Av[r],i), length(Av[r])÷size(Av[r],i))
            G += tmp*tmp'
        end
        F = eigen(G) 
        F.values .= abs.(F.values)
        perm2 = sortperm(real(F.values), rev=true)
        Σ = sqrt.(F.values[perm2])
        U = F.vectors[:,perm2]
#        U,Σ, = svd(reshape(permutedims(Av[1],perm), size(Av[1],i), length(Av[1])÷size(Av[1],i))) 
        #U,Σ, = svd(reshape(permutedims(A,perm), size(A,i), length(A)÷size(A,i))) 
   
#        idx_l = collect(1:ndims(A))
#        idx_r = collect(1:ndims(A))
#        idx_l[i] = -1
#        idx_r[i] = -2
#        G = tensorcontract(A,idx_l,A,idx_r)
#        #G = @ncon([A, A], [idx_l, idx_r])
#        F = eigen((G .+ G') .* .5) # should be symmetric, but sometimes values get very small and numerical error builds up
#        perm = sortperm(real(F.values), rev=true)
#        l = F.values[perm]
#        v = F.vectors[:,perm]

        nkeep = 0
        if verbose > 0
            @printf(" index dimension: %6i\n", dims[i])
        end
        nkeep = 0
        if type == "magnitude"
            for (idx,Σi) in enumerate(Σ)
                if abs(Σi) > thresh
                    nkeep += 1
                    if verbose > 0
                        @printf("   Singular Value %4i = %12.8f\n", idx, Σi)
                    end
                end
            end
        elseif type == "sum"
            target = sum(l)
            curr = 0.0
            for (idx,Σi) in enumerate(Σ)
                if abs(curr-target) > thresh
                    nkeep += 1
                    curr += Σi
                    if verbose > 0
                        @printf("   Singular Value %4i = %12.8f\n", idx, Σi)
                    end
                end
            end
        else
            error("wrong type")
        end
        if max_number != nothing
            nkeep = min(nkeep, max_number)
        end
        #if nkeep == 0
        #    return ntuple(i -> zeros(dims), R), [zeros(T,size(Av[1],i),0) for i in 1:length(dims)]
        #end

        if nkeep > 0 
            factors[i] = U[:,1:nkeep]
        end
        #push!(factors, U[:,1:nkeep])
    end
    return transform_basis(Av,NTuple{N,Matrix{T}}(factors)), factors
end
#=}}}=#



"""
    tucker_recompose(core, factors)

Recompose Tucker Decomposition 
"""
tucker_recompose(core, factors) = transform_basis(core, factors, trans=true)

"""
"""
function transform_basis(v::Array{T,N}, transform_list::Dict{Int,Matrix{T}}; trans=false) where {T,N}
  #={{{=#
    length(transform_list) > 0 || return v

    vv = deepcopy(v)

    # TODO: figure out why the inplace contractions aren't working
    
    for i in 1:N
        if haskey(transform_list, i)
    

            v_indices = collect(1:N)
            v_indices[i] = -i
            if trans
#                if size(transform_list[i],1) == size(transform_list[i],2) 
#                    TensorOperations.tensorcontract!(1, vv, collect(1:N), 'C', 
#                                                     transform_list[i], [-i,i], 'C', 
#                                                     0, vv, v_indices)
#                else
#                    vv = TensorOperations.tensorcontract(vv, collect(1:N), transform_list[i], [-i,i], v_indices)
#                end
                    
                vv = TensorOperations.tensorcontract(vv, collect(1:N), transform_list[i], [-i,i], v_indices)
            else
#                if size(transform_list[i],1) == size(transform_list[i],2)
#                    TensorOperations.tensorcontract!(1, vv, collect(1:N), 'N', 
#                                                     transform_list[i], [i,-i], 'N', 
#                                                     0, vv, v_indices)
#                else
#                    vv = TensorOperations.tensorcontract(vv, collect(1:N), transform_list[i], [i,-i], v_indices)
#                end
                vv = TensorOperations.tensorcontract(vv, collect(1:N), transform_list[i], [i,-i], v_indices)
            end
        end
    end
    return vv
end
#=}}}=#

function transform_basis2(v::Array{T,N}, transform_list::Dict{Int,Matrix{T}}; trans=false) where {T,N}
  #={{{=#
    #
    #   e.g., 
    #   v(i,j,k,l) U(iI) U(jJ) U(lL) = V(I,J,k,L)
    #
    #   vv(i,jkl) = reshape(vv(i,j,k,l))
    #
    #   vv(jkl,I) = v(i,jkl)' * U(i,I)
    #   vv(j,klI) = reshape(vv(jkl,I))
    #
    #   vv(klI,J) = v(j,klI)' * U(j,J)
    #   vv(k,lIJ) = reshape(vv(klI,J))
    #
    #   vv(lIJ,k) = v(k,lIJ)' 
    #   vv(l,IJK) = reshape(vv(lIJ,k))
    #
    #   vv(IJk,L) = v(l,IJk)' * U(l,L) 
    #   vv(I,JKL) = reshape(vv(IJK,L))
  
    length(transform_list) > 0 || return v
    #display(("v:",size(v)))
    #display(("t:",[(i,size(j)) for (i,j) in transform_list]))
    vv = deepcopy(v)
    dims = [size(vv)...]
            
    vv = reshape(vv,dims[1],prod(dims[2:end]))
   
    for i in 1:N
        if haskey(transform_list, i)
    

            if trans
                vv = vv' * transform_list[i]'
                dims[1] = size(transform_list[i])[1]
            else
                vv = vv' * transform_list[i]
                dims[1] = size(transform_list[i])[2]
            end
           
            dims = circshift(dims, -1) 
            vv = reshape(vv,dims[1],prod(dims[2:end]))
        else
            vv = vv' 
            dims = circshift(dims, -1) 
            vv = reshape(vv,dims[1],prod(dims[2:end]))
        end
    end

    return reshape(vv,dims...)
end
#=}}}=#

"""
"""
function transform_basis(v::Array{T,N}, transforms::NTuple{N,Matrix{T}}; trans=false) where {T<:Number,N}
  #={{{=#
    #error("here")
    vv = deepcopy(v)
    dims = [size(vv)...]
            
    vv = reshape(vv,dims[1],prod(dims[2:end]))
    
    for i in 1:N

        if trans
            vv = vv' * transforms[i]'
            dims[1] = size(transforms[i])[1]
        else
            vv = vv' * transforms[i]
            dims[1] = size(transforms[i])[2]
        end

        dims = circshift(dims, -1) 
        vv = reshape(vv,dims[1],prod(dims[2:end]))
    end

    return reshape(vv,dims...)
end
#=}}}=#

"""
"""
function transform_basis(v::NTuple{R,Array{T,N}}, transforms; trans=false) where {T<:Number,N,R}
  #={{{=#
    R > 0 || error(DimensionMismatch)
    return ntuple(i->transform_basis(v[i],transforms,trans=trans), R)
end
#=}}}=#



transform_basis(v::Array{T,N}, transforms::Vector{Matrix{T}}; trans=false) where {T,N} = transform_basis(v, NTuple{N, Matrix{T}}(transforms), trans=trans)
#transform_basis(v::Vector{Array{T,N}}, transforms::Vector{Matrix{T}}; trans=false) where {T,N} = transform_basis(NTuple{length(v), Array{T}}(v), NTuple{N, Matrix{T}}(transforms), trans=trans)


function unfold(A::AbstractArray{T,N}, i::Integer) where {T,N}
    # https://github.com/Jutho/TensorOperations.jl/issues/13
    B = permutedims(A,vcat(i,setdiff(1:N,i)))
    d = size(B,1)
    return reshape(B,(d,div(length(B),d)))
end

LinearAlgebra.norm(A::Tucker{T,N}) where {T,N} = norm(A.core)

















#function Tucker(A::Vector{Array{T,N}}; thresh=-1, max_number=nothing, verbose=0, type="magnitude") where {T<:Number,N}
#    #={{{=#
#    core,factors = tucker_decompose(A, thresh=thresh, max_number=max_number, verbose=verbose, type=type)
#    return Tucker{T,N,1}((core,), NTuple{N}(factors))
#end
##=}}}=#

#function Tucker_tot(A::Array{T,N}; thresh=-1, verbose=0) where {T,N}
##={{{=#
#    core,factors = tucker_decompose_tot(A, thresh=thresh, verbose=verbose)
#    return Tucker{T,N}(core, NTuple{N}(factors))
#end
##=}}}=#

#"""
#    function tucker_decompose_tot(A::Array{T,N}; thresh=1e-7, verbose=1) where {T,N}
#
#Tucker Decomposition of dense tensor: 
#    A ~ X *(1) U1 *(2) U2 ....
#where the cluster states are discarded to ensure ||V-v||_F < thresh, with V and v being the full and 
#approximated tensors. This isn't quite ready for use, as it doens't fall back to the optimal case for
#SVD of a matrix as it doesn't consider the possibility of being diagonal in svd basis.
#"""
#function tucker_decompose_tot(A::Array{T,N}; thresh=1e-7, verbose=1) where {T,N}
##={{{=#
#    factors = Vector{Matrix{T}}()
#    values = [] 
#    inds = []
#    if verbose > 0
#        println(" Tucker Decompose:", size(A))
#    end
#    for i in 1:ndims(A)
#        idx_l = collect(1:ndims(A))
#        idx_r = collect(1:ndims(A))
#        idx_l[i] = -1
#        idx_r[i] = -2
#        G = tensorcontract(A,idx_l,A,idx_r)
#        #G = @ncon([A, A], [idx_l, idx_r])
#        F = eigen((G .+ G') .* .5) # should be symmetric, but sometimes values get very small and numerical error builds up
#        perm = sortperm(real(F.values), rev=true)
#        l = F.values[perm]
#        v = F.vectors[:,perm]
#
#        println(i)
#        for li in l
#            if verbose > 0
#                @printf(" Eigenvalue = %12.8f\n", li)
#            end
#        end
#        #if max_number != nothing
#        #    nkeep = min(nkeep, max_number)
#        #end
#
#        push!(factors, v)
#        append!(values, sqrt.(abs.(l)))
#        append!(inds, [i for j in 1:length(l)])
#    end
#        
#        
#    perm = sortperm(real(values), rev=true)
#    values = values[perm]
#    inds = inds[perm]
#
#    dims = zeros(Int,N)
#    target = sum(values)
#    curr = 0.0
#    nkeep = 0
#    for (idx,i) in enumerate(values) 
#        if abs(curr-target) > thresh
#            nkeep += 1
#            curr += i
#        end
#    end
#    for i in 1:nkeep
#        dims[inds[i]] += 1
#    end
#    for i in 1:N
#        factors[i] = factors[i][:,1:dims[i]]
#    end
#    display(target)
#    display(size.(factors,2))
#    return transform_basis(A,factors), factors
#end
##=}}}=#

