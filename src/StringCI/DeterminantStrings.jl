using Printf
using Parameters
using StaticArrays
using BenchmarkTools
using InteractiveUtils
import Base: display


### 
function string_to_index(str::String)
    return parse(Int, reverse(str); base=2)
end

function index_to_string(index::Int)
    return [parse(Int, ss) for ss in reverse(bitstring(index))]
end





"""
Type to organize all the configuration DeterminantString
"""
#struct DeterminantString
mutable struct DeterminantString
    no::Int
    ne::Int
    sign::Int
    lin_index::Int
    config::Array{Int,1}
    #ca_lookup::Array{Int, 3}
    max::Int
end

function DeterminantString(no::Int, ne::Int)
    return DeterminantString(no, ne, 1, 1, Vector(1:ne), get_nchk(no,ne))
    #return DeterminantString(no, ne, 1, 1, Vector(1:ne), zeros(get_nchk(no,ne),no,no), get_nchk(no,ne))
end

function display(d::DeterminantString)
    print(d)
end


"""
    get_unoccupied(c::DeterminantString)

get list of orbitals that are unoccupied
"""
@inline function get_unoccupied(c::DeterminantString)
    unoccupied = zeros(Int, c.no-c.ne,1)
    j = 1
    for i in 1:c.no
        if i in c.config
            continue
        end
        unoccupied[j] = i
        j+=1
    end
    return unoccupied
end


"""
    get_unoccupied(c::DeterminantString)

get list of orbitals that are unoccupied
"""
function get_unoccupied!(unoccupied::Array{Int,1}, c::DeterminantString)
    j = 1
    for i in 1:c.no
        if i in c.config
            continue
        end
        unoccupied[j] = i
        j+=1
    end
end

function Base.length(c::DeterminantString)
    #=
    return number of DeterminantStrings
    =#
    return c.max
end

"""
    Base.copy!(c1::DeterminantString,c2::DeterminantString)

copy c2 into c1
"""
function Base.copy!(c1::DeterminantString,c2::DeterminantString)
    c1.config = copy(c2.config)
    c1.no = c2.no
    c1.ne = c2.ne
    c1.sign = c2.sign
    c1.lin_index = c2.lin_index
    c1.max = c2.max
end


function Base.print(c::DeterminantString)
    #=
    Pretty print of an determinant DeterminantString
    =#
    @printf("Index: %-10i (%3i,%-3i) Dim: %-10i Sign: %2i ",c.lin_index, c.no, c.ne, c.max, c.sign)
    print(" Config:")
    [@printf("%3i",i) for i in c.config]
    print('\n')
end



function incr!(c::DeterminantString)
    #=
    Increment determinant DeterminantString
    =#
    #={{{=#
    if c.max == nothing
        calc_max!(c)
    end
    if c.lin_index == c.max
        return
    end
    c.lin_index += 1
    incr_comb!(c.config, c.no)
end
#=}}}=#


function calc_max(c::DeterminantString)
    #=
    Calculate dimension of space accessible to a DeterminantString
    =#
    #={{{=#
    return get_nchk(c.no,c.ne)
end
#=}}}=#


function calc_max!(c::DeterminantString)
    #=
    Calculate dimension of space accessible to a DeterminantString
    =#
    #={{{=#
    c.max = get_nchk(c.no,c.ne)
end
#=}}}=#


function incr_comb!(comb::Array{Int,1}, Mend::Int)
    #=
    For a given combination, form the next combination
    =#
    #={{{=#
    N = length(comb)
    for i in N:-1:1
        if comb[i] < Mend - N + i
            comb[i] += 1
            for j in i+1:N
                comb[j]=comb[j-1]+1
            end
            return
        end
    end
    return
end
#=}}}=#


"""
    calc_linear_index!(c::DeterminantString)

Calculate the linear index
"""
function calc_linear_index!(c::DeterminantString)
    #={{{=#
    c.lin_index = 1
    v_prev::Int = 0

    for i::Int in 1:c.ne
        v = c.config[i]
        for j::Int in v_prev+1:v-1
            c.lin_index += StringCI.binom_coeff[c.no-j+1,c.ne-i+1]
            #@btime $c.lin_index += $binom_coeff[$c.no-$j+1,$c.ne-$i+1]
            #c.lin_index += get_nchk(c.no-j,c.ne-i)
        end
        v_prev = v
    end
end
#=}}}=#


"""
    calc_linear_index!(c::DeterminantString, binomcoeff::Array{Int,2})

Calculate the linear index, passing in binomial coefficient matrix makes it much faster
"""
function calc_linear_index!(c::DeterminantString, binomcoeff::Array{Int,2})
    #={{{=#
    c.lin_index = 1
    v_prev::Int = 0

    for i::Int in 1:c.ne
        v = c.config[i]
        for j::Int in v_prev+1:v-1
            c.lin_index += binomcoeff[c.no-j+1,c.ne-i+1]
        end
        v_prev = v
    end
end
#=}}}=#


"""
    calc_linear_index(c::DeterminantString)
    
Return linear index for lexically ordered __config DeterminantString
"""
function calc_linear_index(c::DeterminantString)
    #={{{=#
    v_prev::Int = 0
    lin_index = 1
    for i in 1:c.ne
        v = c.config[i]
        #todo: change mchn from function call to data lookup!
        @inbounds @simd for j in v_prev+1:v-1
            lin_index += get_nchk(c.no-j,c.ne-i)
        end
        v_prev = v
    end
    return lin_index
end
#=}}}=#


"""
    fill_ca_lookup(c::DeterminantString)

Create an index table relating each DeterminantString with all ia substitutions
i.e., ca_lookup[Ka][c(p) + a(p)*n_p] = La
"""
function fill_ca_lookup(c::DeterminantString)
    #={{{=#

    ket = DeterminantString(c.no, c.ne)
    bra = DeterminantString(c.no, c.ne)

    max = calc_max(ket)

    tbl = []
    for K in 1:max
        Kv::Array{Tuple{Int,Int},1} = []
        for p in 1:ket.no
            for q in 1:ket.no
                bra = deepcopy(ket)
                apply_annihilation!(bra,p)
                apply_creation!(bra,q)
                @assert(issorted(bra.config))
                #print("--\n")
                #print(p,'\n')
                #print(q,'\n')
                #print(ket)
                #print(bra)
                if bra.sign == 0
                    push!(Kv,(1,0))
                    continue
                else
                    #calc_max!(bra)
                    #print(bra)
                    calc_linear_index!(bra)
                    push!(Kv,(bra.sign , bra.lin_index))
                end
            end
        end
        push!(tbl,Kv)
        incr!(ket)
    end
    return tbl
end
#=}}}=#



"""
    fill_ca_lookup2(c::DeterminantString)

Create an index table relating each DeterminantString with all ia substitutions
i.e., ca_lookup[Ka,p,q] = sign*La

<L|p'q|K> = sign
"""
function fill_ca_lookup2(c::DeterminantString)
    #={{{=#

    ket = DeterminantString(c.no, c.ne)
    bra = DeterminantString(c.no, c.ne)

    max = calc_max(ket)

    tbl = zeros(Int,ket.no, ket.no, max)
    for K in 1:max
        for q in 1:ket.no
            for p in ket.config
                #bra = deepcopy(ket)
                copy!(bra,ket)
                
                apply_annihilation!(bra,p)
                if bra.sign == 0
                    continue
                end
                apply_creation!(bra,q)
                issorted(bra.config) || throw(Exception)
                if bra.sign == 0
                    continue
                else
                    calc_linear_index!(bra,binom_coeff)
                    #@code_warntype calc_linear_index!(bra,binom_coeff)
                    tbl[q, p, K] = bra.sign*bra.lin_index
                end
            end
        end
        incr!(ket)
    end
    return tbl
end
#=}}}=#

"""
    fill_ca_lookup3(c::DeterminantString)

Create an index table relating each DeterminantString with all ia substitutions
i.e., ca_lookup[Ka,p,q] = (sign,La)

<L|p'q|K> = sign
"""
function fill_ca_lookup3(c::DeterminantString)
    #={{{=#

    ket = DeterminantString(c.no, c.ne)
    bra = DeterminantString(c.no, c.ne)

    max = calc_max(ket)

    tbl = Array{Tuple,3}(undef,ket.no,ket.no,max)
    #tbl = zeros(Int,ket.no, ket.no, max)
    for K in 1:max
        for p in 1:ket.no
            for q in 1:ket.no
                bra = deepcopy(ket)
                apply_annihilation!(bra,p)
                apply_creation!(bra,q)
                #@assert(issorted(bra.config))
                if bra.sign == 0
                    tbl[q, p, K] = (0.0,0)
                else
                    calc_linear_index!(bra)
                    tbl[q, p, K] = (1.0*bra.sign,bra.lin_index)
                end
            end
        end
        incr!(ket)
    end
    return tbl
end
#=}}}=#


#"""
#    fill_vo_lookup(c::DeterminantString)
#
#Create an index table relating each DeterminantString with all ia substitutions
#i.e., ca_lookup[v,o,Ka] = sign*La
#where o and v are occupied and virtual indices
#
#<L|v'o|K> = sign
#"""
#function fill_vo_lookup(c::DeterminantString)
#    #={{{=#
#
#    ket = DeterminantString(c.no, c.ne)
#    bra = DeterminantString(c.no, c.ne)
#
#    max = calc_max(ket)
#
#    no = ket.no
#    ne = ket.ne
#    nv = no-ne
#
#    tbl = zeros(Int,nv, ne, max)
#    #tbl = Array{Tuple,3}(undef,nv,ne,max)
#    #tbl = Array{SVector{4,Int},nv, ne, max}
#    #println("max:",max)
#    for K in 1:max
#        virt = get_unoccupied(ket)
#        for pp in 1:ne
#            p = ket.config[pp]
#            bra1 = deepcopy(ket)
#            apply_annihilation!(bra1,p)
#            for qq in 1:nv 
#                q = virt[qq]
#                bra = deepcopy(bra1)
#                apply_creation!(bra,q)
#                @assert(issorted(bra.config))
#                if bra.sign == 0
#                    continue
#                else
#                    calc_linear_index!(bra)
#                    tbl[qq, pp, K] = bra.sign*bra.lin_index
#                    #tbl[qq, pp, K] = (bra.sign, bra.lin_index, q, p)
#                    #println()
#                    #display(bra)
#                    #println("qq=", qq,", pp=", pp,", K=", K)
#                    #println("sgn=", bra.sign, ", lin=", bra.lin_index,", q=", q, ", p=", p)
#                end
#            end
#        end
#        incr!(ket)
#    end
#    #println(maximum(tbl))
#    #println(minimum(tbl))
#    #throw(Exception)
#    return tbl
#end
##=}}}=#


"""
    reset!(c::DeterminantString)

Reset the DeterminantString to the first config
"""
function reset!(c::DeterminantString)
    #={{{=#
    c.config = Vector(1:c.ne)
    c.sign = 1
    c.lin_index = 1
end
#=}}}=#


function destroy_config!(c::DeterminantString)
    #={{{=#
    c.config = []
    c.sign = 0
    c.lin_index = 0
    c.max = 0
    c.ne = 0
end
#=}}}=#


"""
    apply_annihilation!(c::DeterminantString, orb_index::Int)

Apply an annihilation operator to `c` corresponding to orbital `orb_index` 
"""
function apply_annihilation!(c::DeterminantString, orb_index::Int)
    #=
    apply annihilation operator a_i to current DeterminantString
    where orb_index is i
    =#
    #={{{=#
    @assert(orb_index <= c.no)
    if c.sign == 0
        return
    end
    found = -1
    for i in 1:c.ne
        if c.config[i] == orb_index
            found = i
            break
        end
    end


    if found == -1
        destroy_config!(c)
        return
    end

    if found % 2 != 1
        c.sign *= -1
    end

    deleteat!(c.config,found)

    c.ne -= 1

    #unset data that need to be recomputed
    c.max = 0
    c.lin_index = 0
end
#=}}}=#




"""
    apply_creation!(c::DeterminantString, orb_index::Int)

Apply a creation operator to `c` corresponding to orbital `orb_index` 
"""
function apply_creation!(c::DeterminantString, orb_index::Int)
    #=
    apply creation operator a_i to current DeterminantString
    where orb_index is i
    =#
    #={{{=#
    @assert(orb_index <= c.no)
    if c.sign == 0
        return
    end

    insert_here = 1
    for i in 1:c.ne
        if c.config[i] > orb_index
            insert_here = i
            break
        elseif c.config[i] == orb_index
            destroy_config!(c)
            return
        else
            insert_here += 1
        end
    end

    if insert_here % 2 != 1
        c.sign *= -1
    end

    #print("insert_here ", insert_here, ' ', c.ne, "\n")
    insert!(c.config,insert_here,orb_index)

    c.ne += 1
    #unset data that need to be recomputed
    c.max = 0
    c.lin_index = 0
end
#=}}}=#


function slater_det_energy(h0, h1, h2, deta::DeterminantString, detb::DeterminantString)
    #Compute the energy of a Slater Det specified by
    #   deta is alpha determinant
    #   detb is beta  determinant
    #={{{=#
    E0 = h0
    E1 = 0
    E2 = 0

    na = deta.config
    nb = detb.config
    for ii=1:deta.ne
        i = na[ii]
        E1 += h1[i,i]
        for jj=ii:deta.ne
            j = na[jj]
            E2 += h2[i,i,j,j]
            E2 -= h2[i,j,j,i]
        end
        for jj=1:detb.ne
            j = nb[jj]
            E2 += h2[i,i,j,j]
        end
    end
    for ii=1:detb.ne
        i = nb[ii]
        E1 += h1[i,i]
        for jj=i:detb.ne
            j = nb[jj]
            E2 += h2[i,i,j,j]
            E2 -= h2[i,j,j,i]
        end
    end
    return E0+E1+E2
end
#=}}}=#
