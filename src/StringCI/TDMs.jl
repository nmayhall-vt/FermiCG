using Printf
using BenchmarkTools
using InteractiveUtils


"""
    compute_annihilation(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)

Compute an creation operator between states `bra_v` and `ket_v`
# Arguments
- `no`: number of orbitals
- `bra_na`: number of alpha electrons in bra
- `bra_nb`: number of beta electrons in bra
- `ket_na`: number of alpha electrons in ket
- `ket_nb`: number of beta electrons in ket
- `bra_v`: basis vectors in bra
- `ket_v`: basis vectors ket
- `spin_case`: either (alpha|beta)
"""
function compute_annihilation(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)
    
    if spin_case == "alpha"
        bra_na == ket_na+1 || throw(DimensionMismatch)
        bra_nb == ket_nb || throw(DimensionMismatch)
    elseif spin_case == "beta"
        bra_nb == ket_nb+1 || throw(DimensionMismatch)
        bra_na == ket_na || throw(DimensionMismatch)
    end

    ket = ket_a
    bra = bra_a
    if spin_case == "beta"
        ket = ket_b
        bra = bra_b
    end
    virt = zeros(typeof(ket.no), ket.no-ket.ne)

    #
    # precompute lookup table for binomial coefficients
    _binomial = Array{Int,2}(undef,no+1,no+1)
    for i in 0:no
        for j in i:no
            _binomial[j+1,i+1] = binomial(j,i)
        end
    end

    bra_M = size(bra_v,2)
    ket_M = size(ket_v,2)

    #println((no, bra_na, bra_nb, bra_a.max, bra_b.max))
    #println(size(bra_v), bra_a.max, " ", bra_b.max, " ", bra_M)
    
    v1 = reshape(bra_v, bra_a.max, bra_b.max, bra_M)
    v2 = reshape(ket_v, ket_a.max, ket_b.max, ket_M)
    if spin_case == "alpha"
        v1 = permutedims(v1, [2,3,1])
        v2 = permutedims(v2, [2,3,1])
    elseif spin_case == "beta"
        v1 = permutedims(v1, [1,3,2])
        v2 = permutedims(v2, [1,3,2])
    else
        throw(Exception)
    end

    #
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M, no)
    reset!(ket)
    reset!(bra)
    for J in 1:ket.max
        get_unoccupied!(virt, ket)
        for a in virt
            copy!(bra,ket)
            apply_creation!(bra,a)
            issorted(bra.config) || throw(Exception)
            calc_linear_index!(bra,_binomial)
            I = bra.lin_index
            @views bra_vI = v1[:,:,I]
            @views ket_vJ = v2[:,:,J]
            @views tdma = tdm[:,:,a]
            sgn = bra.sign
            if spin_case == "beta"
                if ket_a.ne%2 != 0
                    sgn = -sgn
                end
            end

            if sgn == 1 
                @tensor begin 
                    tdma[s,t] += bra_vI[I,s] * ket_vJ[I,t]
                end
                #tdma += bra_vI' * ket_vJ
            elseif sgn == -1
                #tdma -= bra_vI' * ket_vJ
                @tensor begin 
                    tdma[s,t] -= bra_vI[I,s] * ket_vJ[I,t]
                end
            else
                throw(Exception)
            end
        end
        incr!(ket)
    end
    tdm = permutedims(tdm, [3,1,2])
    #println(size(tdm))
    return tdm
#=}}}=#
end


"""
    compute_AA(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)

Compute represntation of 2creation operators between states `bra_v` and `ket_v`
# Arguments
- `no`: number of orbitals
- `bra_na`: number of alpha electrons in bra
- `bra_nb`: number of beta electrons in bra
- `ket_na`: number of alpha electrons in ket
- `ket_nb`: number of beta electrons in ket
- `bra_v`: basis vectors in bra
- `ket_v`: basis vectors ket
- `spin_case`: either (alpha|beta)
"""
function compute_AA(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)

    if spin_case == "alpha"
        bra_na == ket_na+2 || throw(DimensionMismatch)
        bra_nb == ket_nb || throw(DimensionMismatch)
    elseif spin_case == "beta"
        bra_nb == ket_nb+2 || throw(DimensionMismatch)
        bra_na == ket_na || throw(DimensionMismatch)
    end
    
    ket = ket_a
    bra = bra_a
    if spin_case == "beta"
        ket = ket_b
        bra = bra_b
    end
    virt = zeros(typeof(ket.no), ket.no-ket.ne)

    #
    # precompute lookup table for binomial coefficients
    _binomial = Array{Int,2}(undef,no+1,no+1)
    for i in 0:no
        for j in i:no
            _binomial[j+1,i+1] = binomial(j,i)
        end
    end

    bra_M = size(bra_v,2)
    ket_M = size(ket_v,2)

    #println((no, bra_na, bra_nb, bra_a.max, bra_b.max))
    #println(size(bra_v), bra_a.max, " ", bra_b.max, " ", bra_M)
    
    v1 = reshape(bra_v, bra_a.max, bra_b.max, bra_M)
    v2 = reshape(ket_v, ket_a.max, ket_b.max, ket_M)
    if spin_case == "alpha"
        v1 = permutedims(v1, [2,3,1])
        v2 = permutedims(v2, [2,3,1])
    elseif spin_case == "beta"
        v1 = permutedims(v1, [1,3,2])
        v2 = permutedims(v2, [1,3,2])
    else
        throw(Exception)
    end

    #
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M, no)
    reset!(ket)
    reset!(bra)
    for J in 1:ket.max
        get_unoccupied!(virt, ket)
        for a in virt
            copy!(bra,ket)
            apply_creation!(bra,a)
            issorted(bra.config) || throw(Exception)
            calc_linear_index!(bra,_binomial)
            I = bra.lin_index
            @views bra_vI = v1[:,:,I]
            @views ket_vJ = v2[:,:,J]
            @views tdma = tdm[:,:,a]
            sgn = bra.sign
            if spin_case == "beta"
                if ket_a.ne%2 != 0
                    sgn = -sgn
                end
            end

            if sgn == 1 
                @tensor begin 
                    tdma[s,t] += bra_vI[I,s] * ket_vJ[I,t]
                end
                #tdma += bra_vI' * ket_vJ
            elseif sgn == -1
                #tdma -= bra_vI' * ket_vJ
                @tensor begin 
                    tdma[s,t] -= bra_vI[I,s] * ket_vJ[I,t]
                end
            else
                throw(Exception)
            end
        end
        incr!(ket)
    end
    tdm = permutedims(tdm, [3,1,2])
    #println(size(tdm))
    return tdm
#=}}}=#
end
