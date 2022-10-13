struct ClusterOps{T}
    cluster::MOCluster
    data::Dict{String,Dict{Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}},Array{T}}}
end

Base.iterate(i::ClusterOps, state=1) = iterate(i.data, state)
Base.length(i::ClusterOps) = length(co.data)
Base.getindex(co::ClusterOps,i) = co.data[i] 
Base.setindex!(co::ClusterOps,val,key) = co.data[key] = val
Base.haskey(co::ClusterOps,key) = haskey(co.data, key)
Base.keys(co::ClusterOps) = keys(co.data)

"""
    ClusterOps(co::ClusterOps, T::Type)

Convert from one data type to another
"""
function ClusterOps(co::ClusterOps, T::Type)
    out = ClusterOps{T}(co.cluster, Dict{String,Dict{Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}},Array{T}}}())
    for (op,fspaces) in co.data
        out.data[op] = Dict{Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}},Array{T}}()
        for (fspacetrans,opmat) in fspaces
            out.data[op][fspacetrans] = Array{T}(opmat)
        end
    end
    return out
end

function Base.display(co::ClusterOps) 
    @printf(" ClusterOps for Cluster: %4i\n",co.cluster.idx)
    norb = length(co.cluster)
    for (op,sectors) in co
        print("   Operator: \n", op)
        print("     Sectors: \n")
        for sector in sectors
            println(sector)
        end
    end
end
function ClusterOps(ci::MOCluster; T::Type=Float64)
    dic1 = Dict{Tuple{Tuple{Int16,Int16},Tuple{Int16,Int16}},Array{T}}()
    dic2 = Dict{String,typeof(dic1)}() 
    return ClusterOps(ci, dic2)
end

function ClusterOps(ci::MOCluster, T::Type)
    return ClusterOps(ci, T=T)
end

"""
    rotate!(ops::ClusterOps, U::Dict{Tuple,Matrix{T}}) where {T} 

Rotate `ops` by unitary matrices in `U`
"""
function rotate!(ops::ClusterOps,U::Dict{Tuple,Matrix{T}}) where T
#={{{=#
    for (op,fspace_deltas) in ops
        #println(" Rotate ", op)
        for (fspace_delta,tdm) in fspace_deltas
            fspace_l = fspace_delta[1]
            fspace_r = fspace_delta[2]

            if haskey(U, fspace_l)==true && haskey(U, fspace_r)==true
                Ul = U[fspace_l]
                Ur = U[fspace_r]
                if length(size(tdm)) == 2
                    @tensoropt tmp[q,s] := Ul[p,q] * Ur[r,s] * tdm[p,r]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 3
                    @tensoropt tmp[p,s,t] := Ul[q,s] * Ur[r,t] * tdm[p,q,r]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 4
                    @tensoropt tmp[p,q,t,u] := Ul[r,t] * Ur[s,u] * tdm[p,q,r,s]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 5
                    @tensoropt  tmp[p,q,r,u,v] := Ul[s,u] * Ur[t,v] * tdm[p,q,r,s,t]
                    ops[op][fspace_delta] = tmp 
                else
                    error("Wrong dimension")
                end

            elseif haskey(U, fspace_l)==true && haskey(U, fspace_r)==false
                Ul = U[fspace_l]
                if length(size(tdm)) == 2
                    @tensoropt tmp[q,r] := Ul[p,q] * tdm[p,r]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 3
                    @tensoropt tmp[p,s,r] := Ul[q,s] * tdm[p,q,r]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 4
                    @tensoropt tmp[p,q,t,s] := Ul[r,t] * tdm[p,q,r,s]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 5
                    @tensoropt tmp[p,q,r,u,t] := Ul[s,u] * tdm[p,q,r,s,t]
                    ops[op][fspace_delta] = tmp 
                else
                    error("Wrong dimension")
                end

            elseif haskey(U, fspace_l)==false && haskey(U, fspace_r)==true
                Ur = U[fspace_r]
                if length(size(tdm)) == 2
                    @tensoropt tmp[p,s] := Ur[r,s] * tdm[p,r]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 3
                    @tensoropt tmp[p,q,t] := Ur[r,t] * tdm[p,q,r]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 4
                    @tensoropt tmp[p,q,r,u] := Ur[s,u] * tdm[p,q,r,s]
                    ops[op][fspace_delta] = tmp 
                elseif length(size(tdm)) == 5
                    @tensoropt tmp[p,q,r,s,v] := Ur[t,v] * tdm[p,q,r,s,t]
                    ops[op][fspace_delta] = tmp 
                else
                    error("Wrong dimension")
                end
            end
        end
    end
end
#=}}}=#


"""
    adjoint(co::ClusterOps; verbose=0)

Take ClusterOps, `co`, and return a new ClusterOps'
"""
#todo
