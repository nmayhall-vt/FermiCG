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
    tdm = zeros(Float64, bra_M, ket_M, no, no)
    reset!(ket)
    reset!(bra)
    for J in 1:ket.max
        get_unoccupied!(virt, ket)
        for a in virt
            for b in virt
                a != b || continue
                copy!(bra,ket)
                apply_creation!(bra,b)
                apply_creation!(bra,a)
                calc_linear_index!(bra,_binomial)
                I = bra.lin_index
                @views bra_vI = v1[:,:,I]
                @views ket_vJ = v2[:,:,J]
                @views tdma = tdm[:,:,a,b]
                sgn = bra.sign

                if sgn == 1 
                    @tensor begin 
                        tdma[s,t] += bra_vI[I,s] * ket_vJ[I,t]
                    end
                elseif sgn == -1
                    @tensor begin 
                        tdma[s,t] -= bra_vI[I,s] * ket_vJ[I,t]
                    end
                else
                    throw(Exception)
                end
            end
        end
        incr!(ket)
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
#=}}}=#
end


"""
    compute_Aa(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)

Compute representation of a'a operators between states `bra_v` and `ket_v`
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
function compute_Aa(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)

    if spin_case == "alpha"
        bra_na == ket_na || throw(DimensionMismatch)
        bra_nb == ket_nb || throw(DimensionMismatch)
    elseif spin_case == "beta"
        bra_nb == ket_nb || throw(DimensionMismatch)
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
    tdm = zeros(Float64, bra_M, ket_M, no, no)
    reset!(ket)
    reset!(bra)
    for J in 1:ket.max
        get_unoccupied!(virt, ket)
        for q in 1:no 
            for p in 1:no 
                copy!(bra,ket)
                apply_annihilation!(bra,q)
                apply_creation!(bra,p)
                bra.sign != 0 || continue
                calc_linear_index!(bra,_binomial)
                I = bra.lin_index
                @views bra_vI = v1[:,:,I]
                @views ket_vJ = v2[:,:,J]
                @views tdma = tdm[:,:,p,q]
                sgn = bra.sign

                if sgn == 1 
                    @tensor begin 
                        tdma[s,t] += bra_vI[I,s] * ket_vJ[I,t]
                    end
                elseif sgn == -1
                    @tensor begin 
                        tdma[s,t] -= bra_vI[I,s] * ket_vJ[I,t]
                    end
                else
                    throw(DomainError(sgn))
                end
            end
        end
        incr!(ket)
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
#=}}}=#
end


"""
    compute_Ab(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix)

Compute representation of p'q operators between states `bra_v` and `ket_v`, where
`p` is alpha and `q` is beta.
# Arguments
- `no`: number of orbitals
- `bra_na`: number of alpha electrons in bra
- `bra_nb`: number of beta electrons in bra
- `ket_na`: number of alpha electrons in ket
- `ket_nb`: number of beta electrons in ket
- `bra_v`: basis vectors in bra
- `ket_v`: basis vectors ket


    G(pqst) = v(IJs) <IJ|p'q|KL> v(KLt) 
            = v(IJs) <J|<I|p'q|K>|L> v(KLt)
            = v(IJs) <J|<I|p'|K>q|L> v(KLt) (-1)^N(K)
            = v(IJs) <I|p'|K> <J|q|L> v(KLt) (-1)^N(K)
        
            this can be vectorized if needed. for now:
            loop over p,q
                loop over K,L
                    sign = -1^N(K)
                    find indices of I,J
                    G(pq,st) += v(IJ,s) v(KL,t) * sign 
                end
            end
"""
function compute_Ab(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)

    bra_na == ket_na+1 || throw(DimensionMismatch)
    bra_nb == ket_nb-1 || throw(DimensionMismatch)
   
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

    v1 = reshape(bra_v, bra_a.max, bra_b.max, bra_M)
    v2 = reshape(ket_v, ket_a.max, ket_b.max, ket_M)
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])

    #
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M, no, no)
    bra_a = deepcopy(ket_a)
    bra_b = deepcopy(ket_b)
    
    sgnK = 1
    if ket_a.ne % 2 != 0 
        sgnK = -sgnK
    end

    reset!(ket_a)
    for K in 1:ket_a.max
        for p in 1:no
            copy!(bra_a,ket_a)
            apply_creation!(bra_a,p)
            bra_a.sign != 0 || continue
            calc_linear_index!(bra_a,_binomial)
            I = bra_a.lin_index

            reset!(ket_b)
            for L in 1:ket_b.max
                for q in 1:no
                    copy!(bra_b,ket_b)
                    apply_annihilation!(bra_b,q)
                    bra_b.sign != 0 || continue
                    calc_linear_index!(bra_b,_binomial)
                    J = bra_b.lin_index
                    @views tdm_pq = tdm[:,:,p,q] 
                    @views v1_IJ = v1[:,I,J]
                    @views v2_KL = v2[:,K,L]
                    sgn = sgnK * bra_a.sign * bra_b.sign
                

                    if sgn == 1 
                        @tensor begin 
                            tdm_pq[s,t] += v1_IJ[s] * v2_KL[t]
                        end
                    elseif sgn == -1
                        @tensor begin 
                            tdm_pq[s,t] -= v1_IJ[s] * v2_KL[t]
                        end
                    else
                        throw(Exception)
                    end
                end #q
                incr!(ket_b)
            end #L
        end #p
        incr!(ket_a)
    end #K
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
#=}}}=#
end


"""
    compute_AB(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix)

Compute representation of p'q' operators between states `bra_v` and `ket_v`, where
`p` is alpha and `q` is beta.
# Arguments
- `no`: number of orbitals
- `bra_na`: number of alpha electrons in bra
- `bra_nb`: number of beta electrons in bra
- `ket_na`: number of alpha electrons in ket
- `ket_nb`: number of beta electrons in ket
- `bra_v`: basis vectors in bra
- `ket_v`: basis vectors ket


    G(pqst) = v(IJs) <IJ|p'q'|KL> v(KLt) 
            = v(IJs) <J|<I|p'q'|K>|L> v(KLt)
            = v(IJs) <J|<I|p'|K>q'|L> v(KLt) (-1)^N(K)
            = v(IJs) <I|p'|K> <J|q'|L> v(KLt) (-1)^N(K)
        
            this can be vectorized if needed. for now:
            loop over p,q
                loop over K,L
                    sign = -1^N(K)
                    find indices of I,J
                    G(pq,st) += v(IJ,s) v(KL,t) * sign 
                end
            end
"""
function compute_AB(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)

    bra_na == ket_na+1 || throw(DimensionMismatch)
    bra_nb == ket_nb+1 || throw(DimensionMismatch)
   
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

    v1 = reshape(bra_v, bra_a.max, bra_b.max, bra_M)
    v2 = reshape(ket_v, ket_a.max, ket_b.max, ket_M)
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])

    #
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M, no, no)
    bra_a = deepcopy(ket_a)
    bra_b = deepcopy(ket_b)
    
    sgnK = 1
    if ket_a.ne % 2 != 0 
        sgnK = -sgnK
    end

    reset!(ket_a)
    for K in 1:ket_a.max
        for p in 1:no
            copy!(bra_a,ket_a)
            apply_creation!(bra_a,p)
            bra_a.sign != 0 || continue
            calc_linear_index!(bra_a,_binomial)
            I = bra_a.lin_index

            reset!(ket_b)
            for L in 1:ket_b.max
                for q in 1:no
                    copy!(bra_b,ket_b)
                    apply_creation!(bra_b,q)
                    bra_b.sign != 0 || continue
                    calc_linear_index!(bra_b,_binomial)
                    J = bra_b.lin_index
                    @views tdm_pq = tdm[:,:,p,q] 
                    @views v1_IJ = v1[:,I,J]
                    @views v2_KL = v2[:,K,L]
                    sgn = sgnK * bra_a.sign * bra_b.sign
                

                    if sgn == 1 
                        @tensor begin 
                            tdm_pq[s,t] += v1_IJ[s] * v2_KL[t]
                        end
                    elseif sgn == -1
                        @tensor begin 
                            tdm_pq[s,t] -= v1_IJ[s] * v2_KL[t]
                        end
                    else
                        throw(Exception)
                    end
                end #q
                incr!(ket_b)
            end #L
        end #p
        incr!(ket_a)
    end #K
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
#=}}}=#
end


"""
    compute_AAa(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)

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
function compute_AAa(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)
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
    #   TDM[s,t,p,q,r] = 
    tdm = zeros(Float64, bra_M, ket_M, no, no, no)
    reset!(ket)
    reset!(bra)
    for J in 1:ket.max
        get_unoccupied!(virt, ket)
        for p in 1:no 
            for q in 1:no 
                for r in 1:no 
                copy!(bra,ket)
                apply_annihilation!(bra,r)
                apply_creation!(bra,q)
                apply_creation!(bra,p)
                bra.sign != 0 || continue
                calc_linear_index!(bra,_binomial)
                I = bra.lin_index
                @views bra_vI = v1[:,:,I]
                @views ket_vJ = v2[:,:,J]
                @views tdm_pqr = tdm[:,:,p,q,r]
                sgn = bra.sign
                if spin_case == "beta"
                    if ket_a.ne%2 != 0
                        sgn = -sgn
                    end
                end

                if sgn == 1 
                    @tensor begin 
                        tdm_pqr[s,t] += bra_vI[I,s] * ket_vJ[I,t]
                    end
                elseif sgn == -1
                    @tensor begin 
                        tdm_pqr[s,t] -= bra_vI[I,s] * ket_vJ[I,t]
                    end
                else
                    throw(Exception)
                end
            end
            end
        end
        incr!(ket)
    end
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
#=}}}=#
end


"""
    compute_ABa(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)

Compute represntation of 2creation operators between states `bra_v` and `ket_v`
# Arguments
- `no`: number of orbitals
- `bra_na`: number of alpha electrons in bra
- `bra_nb`: number of beta electrons in bra
- `ket_na`: number of alpha electrons in ket
- `ket_nb`: number of beta electrons in ket
- `bra_v`: basis vectors in bra
- `ket_v`: basis vectors ket
- `spin_case`: either (alpha|beta) giving ABa or BAb
"""
function compute_ABa(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)

    bra_na == ket_na+0 || throw(DimensionMismatch)
    bra_nb == ket_nb+1 || throw(DimensionMismatch)
   
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

    v1 = reshape(bra_v, bra_a.max, bra_b.max, bra_M)
    v2 = reshape(ket_v, ket_a.max, ket_b.max, ket_M)
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])

    #
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M, no, no, no)
    bra_a = deepcopy(ket_a)
    bra_b = deepcopy(ket_b)
    
    sgnK = -1
    if (ket_a.ne) % 2 != 0 
        sgnK = -sgnK
    end

    # here, p'r are Alpha
    #  v(K,L,s) <KL|p'q'r|IJ> V(I,J,t)
    # -v(K,L,s) <L|<K|p'rq'|I>|J> V(I,J,t)
    # -v(K,L,s) <L|<K|p'r|I>q'|J> V(I,J,t) (-1)^N(I)
    # -v(K,L,s) <K|p'r|I> <L|q'|J> V(I,J,t) (-1)^N(I)
    #
    reset!(ket_a)
    for K in 1:ket_a.max

        reset!(ket_b)
        for L in 1:ket_b.max
            for p in 1:no
                for r in 1:no
                    copy!(bra_a,ket_a)
                    apply_annihilation!(bra_a,r)
                    bra_a.sign != 0 || continue
                    apply_creation!(bra_a,p)
                    bra_a.sign != 0 || continue
                    calc_linear_index!(bra_a,_binomial)
                    I = bra_a.lin_index
                    
                    for q in 1:no

                        copy!(bra_b,ket_b)
                        apply_creation!(bra_b,q)
                        bra_b.sign != 0 || continue
                        calc_linear_index!(bra_b,_binomial)
                        J = bra_b.lin_index

                        @views tdm_pqr = tdm[:,:,p,q,r] 
                        @views v1_IJ = v1[:,I,J]
                        @views v2_KL = v2[:,K,L]
                        #sgn = bra_a.sign * bra_b.sign
                        sgn = sgnK * bra_a.sign * bra_b.sign

                        if sgn == 1 
                            @tensor begin 
                                tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                            end
                        elseif sgn == -1
                            @tensor begin 
                                tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                            end
                        else
                            throw(Exception)
                        end
                    end #q
                end #r
            end #p
            incr!(ket_b)
        end #L
        incr!(ket_a)
    end #K
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
#=}}}=#
end


"""
    compute_ABb(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix, spin_case)

Compute represntation of 2creation operators between states `bra_v` and `ket_v`
# Arguments
- `no`: number of orbitals
- `bra_na`: number of alpha electrons in bra
- `bra_nb`: number of beta electrons in bra
- `ket_na`: number of alpha electrons in ket
- `ket_nb`: number of beta electrons in ket
- `bra_v`: basis vectors in bra
- `ket_v`: basis vectors ket
- `spin_case`: either (alpha|beta) giving ABa or BAb
"""
function compute_ABb(no::Integer, bra_na, bra_nb, ket_na, ket_nb, bra_v::Matrix, ket_v::Matrix)
#={{{=#
    bra_a = DeterminantString(no,bra_na)
    bra_b = DeterminantString(no,bra_nb)

    ket_a = DeterminantString(no,ket_na)
    ket_b = DeterminantString(no,ket_nb)

    bra_na == ket_na+1 || throw(DimensionMismatch)
    bra_nb == ket_nb+0 || throw(DimensionMismatch)
   
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

    v1 = reshape(bra_v, bra_a.max, bra_b.max, bra_M)
    v2 = reshape(ket_v, ket_a.max, ket_b.max, ket_M)
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])

    #
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M, no, no, no)
    bra_a = deepcopy(ket_a)
    bra_b = deepcopy(ket_b)

    # here, q'r are Beta
    #  v(K,L,s) <KL|p'q'r|IJ> V(I,J,t)
    # -v(K,L,s) <L|<K|p'|I>q'r|J> V(I,J,t) 
    # -v(K,L,s) <K|p'|I> <L|q'r|J> V(I,J,t) 
    #
    reset!(ket_a)
    for K in 1:ket_a.max

        reset!(ket_b)
        for L in 1:ket_b.max
            for p in 1:no
                copy!(bra_a,ket_a)
                apply_creation!(bra_a,p)
                bra_a.sign != 0 || continue
                calc_linear_index!(bra_a,_binomial)
                I = bra_a.lin_index

                for q in 1:no
                    for r in 1:no

                        copy!(bra_b,ket_b)
                        apply_annihilation!(bra_b,r)
                        apply_creation!(bra_b,q)
                        
                        bra_b.sign != 0 || continue
                        calc_linear_index!(bra_b,_binomial)
                        J = bra_b.lin_index

                        @views tdm_pqr = tdm[:,:,p,q,r] 
                        @views v1_IJ = v1[:,I,J]
                        @views v2_KL = v2[:,K,L]
                        sgn = bra_a.sign * bra_b.sign 

                        if sgn == 1 
                            @tensor begin 
                                tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                            end
                        elseif sgn == -1
                            @tensor begin 
                                tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                            end
                        else
                            error(" sign?: ", sgn) 
                        end
                    end #r
                end #q
            end #p
            incr!(ket_b)
        end #L
        incr!(ket_a)
    end #K
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
#=}}}=#
end
