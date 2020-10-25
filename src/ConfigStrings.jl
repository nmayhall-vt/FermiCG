using Printf
using Parameters


import Base: length, print


@with_kw mutable struct ConfigString
	#=
	Type to organize all the configuration ConfigString
	=#
	no::Int = 0
	ne::Int = 0
	sign::Int = 1
	lin_index::Int = 1
	config::Array{Int,1} = Vector(1:ne)
	#ca_lookup::Array{Array{Int,1},1}
	#max::Int = get_nchk(no,ne)
end



function get_unoccupied(c::ConfigString)
    #=
    return number of ConfigStrings
    =#
end

function length(c::ConfigString)
    #=
    return number of ConfigStrings
    =#
    return c.max
end


function print(c::ConfigString)
	#=
	Pretty print of an determinant ConfigString
	=#
	@printf("Index: %-10i NOrb: %-4i Dim: %-10i Sign: %2i ",c.lin_index, c.no, c.max, c.sign)
	print(c.config)
	print('\n')
end



function incr!(c::ConfigString)
	#=
	Increment determinant ConfigString
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


function calc_max(c::ConfigString)
    #=
    Calculate dimension of space accessible to a ConfigString
    =#
#={{{=#
    return Tools.get_nchk(c.no,c.ne)
end
#=}}}=#


function calc_max!(c::ConfigString)
    #=
    Calculate dimension of space accessible to a ConfigString
    =#
    #={{{=#
    c.max = Tools.get_nchk(c.no,c.ne)
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


function calc_linear_index!(c::ConfigString)
    #=
    Return linear index for lexically ordered __config ConfigString
    =#
#={{{=#
    c.lin_index = 1
    v_prev::Int = 0

    for i in 1:c.ne
        v = c.config[i]
        #todo: change mchn from function call to data lookup!
        for j in v_prev+1:v-1
            #print(c)
            #print(c.no-j, " ", c.ne-i,'\n')
            c.lin_index += Tools.get_nchk(c.no-j,c.ne-i)
        end
        v_prev = v
    end
    return
end
#=}}}=#


function calc_linear_index(c::ConfigString)
    #=
    Return linear index for lexically ordered __config ConfigString
    =#
#={{{=#
    v_prev::Int = 0
    lin_index = 1
    for i in 1:c.ne
        v = c.config[i]
        #todo: change mchn from function call to data lookup!
        for j in v_prev+1:v-1
            #print(c)
            #print(c.no-j, " ", c.ne-i,'\n')
            lin_index += Tools.get_nchk(c.no-j,c.ne-i)
        end
        v_prev = v
    end
    return lin_index
end
#=}}}=#


function fill_ca_lookup(c::ConfigString)
    #=
    Create an index table relating each ConfigString with all ia substitutions
        i.e., ca_lookup[Ka][c(p) + a(p)*n_p] = La
    =#
#={{{=#

    ket = ConfigString(no=c.no, ne=c.ne)
    bra = ConfigString(no=c.no, ne=c.ne)

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


function reset!(c::ConfigString)
    #={{{=#
    c.config = Vector(1:c.ne)
    c.sign = 1
    c.lin_index = 1
end
#=}}}=#


function destroy_config!(c::ConfigString)
#={{{=#
    c.config = []
    c.sign = 0
    c.lin_index = 0
    c.max = 0
    c.ne = 0
end
#=}}}=#


function apply_annihilation!(c::ConfigString, orb_index::Int)
    #=
    apply annihilation operator a_i to current ConfigString
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


function apply_creation!(c::ConfigString, orb_index::Int)
    #=
    apply creation operator a_i to current ConfigString
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


function slater_det_energy(h0, h1, h2, deta::ConfigString, detb::ConfigString)
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
