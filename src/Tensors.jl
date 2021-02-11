"""
    core::Array{T, N}
    factors::NTuple{N, Matrix{T}}

Tucker factors are stored as Tall matrices
"""
struct Tucker{T, N} 
    core::Array{T, N}
    factors::NTuple{N, Matrix{T}}
    #props::Dict{Symbol, Any}
end


function Tucker(A::Array{T,N}; thresh=-1, max_number=nothing, verbose=0) where {T,N}
    core,factors = tucker_decompose(A, thresh=thresh, max_number=max_number, verbose=verbose)
    return Tucker{T,N}(core, NTuple{N}(factors))
end
recompose(t::Tucker{T,N}) where {T<:Number, N} = tucker_recompose(t.core, t.factors)
dims_large(t::Tucker{T,N}) where {T<:Number, N} = return [size(f,1) for f in t.factors]
dims_small(t::Tucker{T,N}) where {T<:Number, N} = return [size(f,2) for f in t.factors]
Base.length(t::Tucker) = prod(dims_small(t))
Base.size(t::Tucker) = size(t.core) 
function Base.permutedims(t::Tucker{T,N}, perm) where {T,N}
    #t.core .= permutedims(t.core, perm)
    return Tucker{T,N}(permutedims(t.core, perm), t.factors[perm])
end


"""
    add(tucks::Vector{Tucker{T,N}}; thresh=1e-10, max_number=nothing) where {T,N}

Add together multiple Tucker instances. Assumed non-orthogonal.

# Arguments
- `tucks::Vector{Tucker{T,N}}`: Vector of Tucker objects
"""
function add(tucks::Vector{Tucker{T,N}}; thresh=1e-10, max_number=nothing) where {T,N}

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
        for si in F.S 
            if si*si > thresh
                nkeep += 1
            end
        end
        if max_number != nothing
            nkeep = min(nkeep, max_number)
        end
        Ui = F.U[:,1:nkeep]
        push!(new_factors, Ui)
    end


    new_core = zeros([size(new_factors[i],2) for i in 1:N]...)

    for tuck in tucks
        transforms = Vector{Matrix{Float64}}()
        for i in 1:N 
            push!(transforms, new_factors[i]'*tuck.factors[i])
        end

        new_core += recompose(Tucker{T,N}(tuck.core, Tuple(transforms)))
    end
    return Tucker{T,N}(new_core, Tuple(new_factors))

end
Base.:(+)(t1::Tucker{T,N}, t2::Tucker{T,N}) where {T,N} = add([t1, t2])

"""
    compress(t::Tucker)

Try to compress further 
"""
function compress(t::Tucker{T,N}; thresh=1e-7, max_number=nothing) where {T,N}

    tt = Tucker(t.core, thresh=thresh, max_number=max_number)

    new_factors = [zeros(1,1) for i in 1:N]

    for i in 1:N
        new_factors[i]  = t.factors[i] * tt.factors[i]
    end

    return Tucker(tt.core, NTuple{N}(new_factors)) 
end

"""
    dot(t1::Tucker{T,N}, t2::Tucker{T,N}) where {T,N}

Note: This doesn't assume `t1` and `t2` have the same compression vectors 
"""
function dot(t1::Tucker{T,N}, t2::Tucker{T,N}) where {T,N}
    #overlaps = []
    overlaps = Dict{Int,Matrix{T}}()
    all(dims_large(t1) .== dims_large(t2)) || error(" t1 and t2 don't have same dimensions")
    for f in 1:N
        #push!(overlaps, t1.factors[f]' * t2.factors[f])
        overlaps[f] = t1.factors[f]' * t2.factors[f]
    end
    return sum(transform_basis(t1.core, overlaps) .* t2.core)
    #return sum(tucker_recompose(t1.core, overlaps) .* t2.core)
end


"""
Tucker Decomposition of dense tensor: 
A ~ X *(1) U1 *(2) U2 ....
"""
function tucker_decompose(A::Array{T,N}; thresh=1e-7, max_number=nothing, verbose=1) where {T,N}
    factors = Vector{Matrix{T}}()
    if verbose > 0
        println(" Tucker Decompose:", size(A))
    end
    for i in 1:ndims(A)
        idx_l = collect(1:ndims(A))
        idx_r = collect(1:ndims(A))
        idx_l[i] = -1
        idx_r[i] = -2
        G = tensorcontract(A,idx_l,A,idx_r)
        #G = @ncon([A, A], [idx_l, idx_r])
        F = eigen((G .+ G') .* .5) # should be symmetric, but sometimes values get very small and numerical error builds up
        perm = sortperm(real(F.values), rev=true)
        l = F.values[perm]
        v = F.vectors[:,perm]

        nkeep = 0
        if verbose > 0
            @printf(" index dimension: %6i\n", size(A)[i])
        end
        for li in l
            if verbose > 0
                @printf(" Singular value = %12.8f\n", li)
            end
            if abs(li) > thresh
                nkeep += 1
            end
        end
        if max_number != nothing
            nkeep = min(nkeep, max_number)
        end

        push!(factors, v[:,1:nkeep])
    end
    return transform_basis(A,factors), factors
end

"""
    tucker_recompose(core, factors)

Recompose Tucker Decomposition 
"""
tucker_recompose(core, factors) = transform_basis(core, factors, trans=true)

"""
"""
function transform_basis(v::Array{T,N}, transform_list::Dict{Int,Matrix{T}}; trans=false) where {T,N}
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
function transform_basis(v::Array{T,N}, transforms::NTuple{N,Matrix{T}}; trans=false) where {T,N}
  #={{{=#
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

transform_basis(v::Array{T,N}, transforms; trans=false) where {T,N} = transform_basis(v, NTuple{N, Matrix{T}}(transforms), trans=trans)
