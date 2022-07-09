using LinearAlgebra 
using Printf
using Parameters
using Profile
using LinearMaps
using BenchmarkTools
using OrderedCollections
using PyCall




struct FCIProblem
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
end

function FCIProblem(no, na, nb)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    dima = calc_nchk(no,na)
    dimb = calc_nchk(no,nb)
    return FCIProblem(no, na, nb, dima, dimb, dima*dimb, false, false, 1, "direct", 1)
end

function display(p::FCIProblem)
    @printf(" FCIProblem:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
    #@printf(" FCIProblem::  NOrbs: %2i NAlpha: %2i NBeta: %2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end

function compute_spin_diag_terms_full!(H, P::FCIProblem, Hmat)
    #={{{=#

    print(" Compute same spin terms.\n")
    size(Hmat,1) == P.dim || throw(DimensionMismatch())

    Hdiag_a = FCI.precompute_spin_diag_terms(H,P,P.na)
    Hdiag_b = FCI.precompute_spin_diag_terms(H,P,P.nb)
    Hmat .+= kron(Matrix(1.0I, P.dimb, P.dimb), Hdiag_a)
    Hmat .+= kron(Hdiag_b, Matrix(1.0I, P.dima, P.dima))

end
#=}}}=#


"""
    build_H_matrix(ints, P::FCIProblem)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis 
in the sector of Fock space specified by `P`
"""
function build_H_matrix(ints, P::FCIProblem)

    T = eltype(ints.h0)

    Hmat = zeros(P.dim, P.dim)

    Hdiag_a = precompute_spin_diag_terms(ints,P,P.na)
    Hdiag_b = precompute_spin_diag_terms(ints,P,P.nb)
    # 
    #   Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)
    #   
    #   Add spin diagonal components
    Hmat += kron(Matrix(1.0I, P.dimb, P.dimb), Hdiag_a)
    Hmat += kron(Hdiag_b, Matrix(1.0I, P.dima, P.dima))
    #
    #   Add opposite spin term (todo: make this reasonably efficient)
    Hmat += compute_ab_terms_full(ints, P, T=T)
    
    Hmat = .5*(Hmat+Hmat')

    return Hmat
end



"""
    compute_fock_diagonal!(H, P::FCIProblem, e_mf::Float64)
"""
function compute_fock_diagonal(P::FCIProblem, orb_energies::Vector, e_mf::Real)
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    
    meanfield_e = e_mf 
    meanfield_e -=sum(orb_energies[ket_a.config])
    meanfield_e -=sum(orb_energies[ket_b.config])

    fdiag = zeros(ket_a.max, ket_b.max)
    
    reset!(ket_a) 
    reset!(ket_b) 
    for Ib in 1:ket_b.max
        eb = sum(orb_energies[ket_b.config]) + meanfield_e
        for Ia in 1:ket_a.max
            fdiag[Ia,Ib] = eb + sum(orb_energies[ket_a.config]) 
            incr!(ket_a)
        end
        incr!(ket_b)
    end
    fdiag = reshape(fdiag,ket_a.max*ket_b.max)
    #display(fdiag)
    return fdiag
end


"""
    compute_ab_terms_full!(H, P::FCIProblem, Hmat)
"""
function compute_ab_terms_full!(H, P::FCIProblem, Hmat)
    #={{{=#

    print(" Compute opposite spin terms.\n")
    @assert(size(Hmat,1) == P.dim)

    #v = transpose(vin)

    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)


    ket_a_lookup = fill_ca_lookup(ket_a)
    ket_b_lookup = fill_ca_lookup(ket_b)

    reset!(ket_b)

    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb-1) * ket_a.max

            #  <pq|rs> p'q'sr  --> (pr|qs) (a,b)
            for r in 1:ket_a.no
                for p in 1:ket_a.no
                    sign_a, La = ket_a_lookup[Ka][p+(r-1)*ket_a.no]
                    if La == 0
                        continue
                    end

                    Lb = 1
                    sign_b = 1
                    L = 1 
                    for s in 1:ket_b.no
                        for q in 1:ket_b.no
                            sign_b, Lb = ket_b_lookup[Kb][q+(s-1)*ket_b.no]

                            if Lb == 0
                                continue
                            end

                            L = La + (Lb-1) * bra_a.max

                            Hmat[K,L] += H.h2[p,r,q,s] * sign_a * sign_b

                        end
                    end
                end
            end
            incr!(ket_a)

        end
        incr!(ket_b)
    end
    return  
end
#=}}}=#


"""
    compute_ab_terms_full(H, P::FCIProblem)
"""
function compute_ab_terms_full(H, P::FCIProblem; T::Type=Float64)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")

    #v = transpose(vin)

    Hmat = zeros(T, P.dim, P.dim)

    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)

    ket_a_lookup = fill_ca_lookup(ket_a)
    ket_b_lookup = fill_ca_lookup(ket_b)
    ket_a_lookup2 = fill_ca_lookup2(ket_a)
    ket_b_lookup2 = fill_ca_lookup2(ket_b)

    a_max = bra_a.max
    reset!(ket_b)

    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb-1) * ket_a.max

            #  <pq|rs> p'q'sr  --> (pr|qs) (a,b)
            for r in 1:ket_a.no
                for p in 1:ket_a.no
                    #sign_a, La = ket_a_lookup[Ka][p+(r-1)*ket_a.no]
                    La = ket_a_lookup2[p,r,Ka]
                    if La == 0
                        continue
                    end
                    sign_a = sign(La)
                    La = abs(La)

                    Lb = 1
                    sign_b = 1
                    L = 1 
                    for s in 1:ket_b.no
                        for q in 1:ket_b.no
                            Lb = ket_b_lookup2[q,s,Kb]
                            if Lb == 0
                                continue
                            end
                            sign_b = sign(Lb)
                            Lb = abs(Lb)
                            #sign_b, Lb = ket_b_lookup[Kb][q+(s-1)*ket_b.no]

                            if Lb == 0
                                continue
                            end

                            L = La + (Lb-1) * a_max

                            Hmat[K,L] += H.h2[p,r,q,s] * sign_a * sign_b
                            continue
                        end
                    end
                end
            end
            incr!(ket_a)

        end
        incr!(ket_b)
    end
    #sig = transpose(sig)
    return Hmat 
end
#=}}}=#




# Helper functions for Olsen's agorithm
function _gather!(FJb::Vector{T}, occ::Vector{Int}, vir::Vector{Int}, vkl::Array{T,2}, Ib::Int,ket_b_lookup) where {T}
#={{{=#
    i::Int = 1
    j::Int = 1
    Jb::Int = 1
    sgn::T = 1.0
    @inbounds @simd for j in occ 
        for i in vir
            Jb = ket_b_lookup[i,j,Ib]
            sgn = sign(Jb)
            Jb = abs(Jb)
            FJb[Jb] = FJb[Jb] + vkl[j,i]*sgn
        end
    end
end
#=}}}=#

function _mult!(Ckl::Array{T,3}, FJb::Array{T,1}, VI::Array{T,2}) where {T}
    #={{{=#
    VI .= 0
    nI = size(Ckl)[1]
    n_roots::Int = size(Ckl)[3]
    ket_max = size(FJb)[1]
    tmp = 0.0
    for si in 1:n_roots
        @views V = VI[:,si]
        for Jb in 1:ket_max
            tmp = FJb[Jb]
            if abs(tmp) > 1e-14
                @inbounds @simd for I in 1:nI
                    VI[I,si] += tmp*Ckl[I,Jb,si]
                end
                #@views LinearAlgebra.axpy!(tmp, Ckl[:,Jb,si], VI[:,si])
                #@inbounds VI[:,si] .+= tmp .* Ckl[:,Jb,si]
                #@inbounds @views @. VI[:,si] += tmp * Ckl[:,Jb,si]
            end
        end
    end
end
#=}}}=#

function _scatter!(sig::Array{T,3}, VI::Array{T,2}, L::Vector{Int}, R::Vector{Int}, Ib::Int) where {T}
    #={{{=#
    n_roots = size(sig)[3]
    @inbounds @simd for si in 1:n_roots
        for Li in 1:length(L)
            sig[R[Li],Ib,si] += VI[Li,si] 
            #@inbounds sig[R[Li],Ib,si] += VI[Li,si] 
        end
    end
end
#=}}}=#

function _getCkl!(Ckl::Array{T,3}, v,L::Vector{Int}) where {T}
    #={{{=#
    nI = length(L)
    n_roots = size(v)[3]
    ket_max = size(v)[2]
    @inbounds @simd for si in 1:n_roots
        for Jb in 1:ket_max
            for Li in 1:nI
                Ckl[Li,Jb,si] = v[abs(L[Li]), Jb, si] * sign(L[Li])
            end
        end
    end
end
#=}}}=#

"""
    compute_ab_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup, ket_b_lookup)
"""
function compute_ab_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup, ket_b_lookup)
    #={{{=#

    T = eltype(v[1])

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")
    @assert(size(v,1)*size(v,2) == P.dim)

    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)


    no = ket_a.no

    a_max::Int = bra_a.max
    reset!(ket_b)
    
    #
    #   sig3(Ia,Ib,s) = <Ia|k'l|Ja> <Ib|i'j|Jb> V(ij,kl) C(Ja,Jb,s)
    n_roots::Int = size(v,3)
    #v = reshape(v,ket_a.max, ket_b.max, n_roots) 
    sig = zeros(T, ket_a.max, ket_b.max, n_roots) 
    FJb_scr1 = zeros(T, ket_b.max) 
    Ckl_scr1 = zeros(T, binomial(ket_a.no-1,ket_a.ne-1), size(v)[2], size(v)[3])
    Ckl_scr2 = zeros(T, binomial(ket_a.no-2,ket_a.ne-1), size(v)[2], size(v)[3])
    Ckl = Array{T,3}
    virt = zeros(Int,ket_b.no-ket_b.ne)
    diff_ref = Set(collect(1:ket_b.no))
    FJb = copy(FJb_scr1)
    for k in 1:ket_a.no,  l in 1:ket_a.no
        #@printf(" %4i, %4i\n",k,l)
        L = Vector{Int}()
        R = Vector{Int}()
        #sgn = Vector{Int}()
        #for (Iidx,I) in enumerate(ket_a_lookup[k,l,:])
        #    if I[2] != 0
        #        push!(R,Iidx)
        #        push!(sgn,I[1])
        #        push!(L,I[2])
        #    end
        #end
        for (Iidx,I) in enumerate(ket_a_lookup[k,l,:])
            if I != 0
                push!(R,Iidx)
                push!(L,I)
            end
        end
        VI = zeros(T, length(L),n_roots)
        #Ckl = zeros(T, size(v)[2], length(L), size(v)[3])
        if k==l
            Ckl = deepcopy(Ckl_scr1)
        else
            Ckl = deepcopy(Ckl_scr2)
        end
        #nI = length(L)
        #for si in 1:n_roots
        #    @simd for Jb in 1:ket_b.max
        #        for Li in 1:nI
        #            @inbounds Ckl[Li,Jb,si] = v[abs(L[Li]), Jb, si] * sign(L[Li])
        #        end
        #    end
        #end
        _getCkl!(Ckl, v, L)
        
        vkl = H.h2[:,:,l,k]
        reset!(ket_b)
        for Ib in 1:ket_b.max
            fill!(FJb,T(0.0))
            Jb = 1
            sgn = 1
            zero_num = 0
        
            no = ket_b.no
            ne = ket_b.ne
            nv = no-ne
            scr1 = 0.0
            get_unoccupied!(virt, ket_b)
            #virt = setdiff(diff_ref, ket_b.config)
            #@btime $_gather!($FJb, $ket_b.config, $virt, $vkl, $Ib, $ket_b_lookup)
            _gather!(FJb, ket_b.config, virt, vkl, Ib, ket_b_lookup)
            #
            # diagonal part
            for j in ket_b.config
                FJb[Ib] += H.h2[j,j,l,k]
            end
          
            if 1==0
                _mult_old!(Ckl, FJb, VI)
            end
            if 1==0
                @tensor begin
                    VI[I,s] = FJb[J] * Ckl[J,I,s]
                end
            end
            if 1==1
                _mult!(Ckl, FJb, VI)
            end
           
            _scatter!(sig,VI,L,R,Ib)
            #@btime $_scatter!($sig,$VI,$L,$R,$Ib)

            incr!(ket_b)
        end
    end


    return sig
    

end
#=}}}=#




function _ss_sum!(sig::Array{T,3}, v::Array{T,3}, F::Vector{T},Ia::Int) where {T}
    nKb     = size(v)[1]
    n_roots = size(v)[2]
    nJa     = size(v)[3]

    for Ja in 1:nJa
        if abs(F[Ja]) > 1e-14 
            @inbounds @simd for si in 1:n_roots
                for Kb in 1:nKb
                    sig[Kb,si,Ia] += F[Ja]*v[Kb,si,Ja]
                end
            end
        end
    end
end

function _ss_sum_Ia!(sig::Array{T,3}, v::Array{T,3}, F::Vector{T},Ia::Int) where {T}
    nJa = size(v)[3]
    nKb = size(v)[1]
    n_roots = size(v)[2]

    for Ja in 1:nJa
        if abs(F[Ja]) > 1e-14 
            @inbounds @simd for si in 1:n_roots
                for Kb in 1:nKb
                    sig[Kb,si,Ia] += F[Ja]*v[Kb,si,Ja]
                end
            end
        end
    end
end


"""
    compute_aa_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup)
"""
function compute_ss_terms2(v, H, P::FCIProblem, ket_a_lookup, ket_b_lookup)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")
    @assert(size(v,1)*size(v,2) == P.dim)

    T = eltype(v[1])
    #v = transpose(vin)

    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)

    no = ket_a.no
    n_roots = P.n_roots

    a_max::Int = bra_a.max
    reset!(ket_a)

    #
    #   sig1(Ia,Ib,s) = <Ib|i'j|Jb> V(ij,kl) C(Ja,Jb,s)
    n_roots::Int = size(v,3)
    #v = reshape(v,ket_a.max, ket_b.max, n_roots) 
    sig = zeros(T, ket_a.max, ket_b.max, n_roots) 
    size(sig) == size(v) || throw(DimensionError())

    h1eff = deepcopy(H.h1)
    @tensor begin
        h1eff[p,q] -= .5 * H.h2[p,j,j,q]  
    end

    #bb
    sig = permutedims(sig,[1,3,2])
    v = permutedims(v,[1,3,2])
    
    ket = ket_b
    reset!(ket) 
    F = zeros(T, ket_b.max)
    for I in 1:ket.max
        F .= 0
        for k in 1:ket.no, l in 1:ket.no
            K = ket_b_lookup[k,l,I]
            if K == 0
                continue
            end
            sign_kl = sign(K)
            K = abs(K)

            @inbounds F[K] += sign_kl * h1eff[k,l]
            for i in 1:ket.no, j in 1:ket.no
                J = ket_b_lookup[i,j,K]
                if J == 0
                    continue
                end
                sign_ij = sign(J)
                J = abs(J)
                if sign_kl == sign_ij
                    @inbounds F[J] += .5 * H.h2[i,j,k,l]
                else
                    @inbounds F[J] -= .5 * H.h2[i,j,k,l]
                end
            end
        end
        _ss_sum!(sig,v,F,I)
    end
    #sig = permutedims(sig,[1,3,2])
    #v = permutedims(v,[1,3,2])



    #aa
    sig = permutedims(sig,[3,2,1])
    v = permutedims(v,[3,2,1])

    ket = ket_a
    reset!(ket) 
    F = zeros(T, ket_a.max)
    bra = deepcopy(ket)
    for I in 1:ket.max
        F .= 0
        for k in 1:ket.no, l in 1:ket.no
            K = ket_a_lookup[k,l,I]
            if K == 0
                continue
            end
            sign_kl = sign(K)
            K = abs(K)

            @inbounds F[K] += sign_kl * h1eff[k,l]
            for i in 1:ket.no, j in 1:ket.no
                J = ket_a_lookup[i,j,K]
                if J == 0
                    continue
                end
                sign_ij = sign(J)
                J = abs(J)
                if sign_kl == sign_ij
                    @inbounds F[J] += .5 * H.h2[i,j,k,l]
                else
                    @inbounds F[J] -= .5 * H.h2[i,j,k,l]
                end

            end
        end

        _ss_sum_Ia!(sig,v,F,I)

    end
    
    sig = permutedims(sig,[3,1,2])
    v = permutedims(v,[3,1,2])
    
    return sig

end
#=}}}=#


"""
    compute_ab_terms(v, H, P::FCIProblem)
"""
function compute_ab_terms(v, H, P::FCIProblem)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")
    @assert(size(v,1) == P.dim)

    #v = transpose(vin)

    sig = deepcopy(v)
    sig .= 0


    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)

    ket_a_lookup = fill_ca_lookup2(ket_a)
    ket_b_lookup = fill_ca_lookup2(ket_b)

    a_max = bra_a.max
    reset!(ket_b)

    n_roots = size(sig,2)
    scr = zeros(1,ket_a.max*ket_b.max)
    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb-1) * ket_a.max

            #  <pq|rs> p'q'sr  --> (pr|qs) (a,b)
            for r in 1:ket_a.no
                for p in 1:ket_a.no
                    #sign_a, La = ket_a_lookup[Ka][p+(r-1)*ket_a.no]
                    La = ket_a_lookup[p,r,Ka]
                    sign_a = sign(La)
                    La = abs(La)
                    if La == 0
                        continue
                    end

                    Lb = 1
                    sign_b = 1
                    L = 1 
                    for s in 1:ket_b.no
                        for q in 1:ket_b.no
                            Lb = ket_b_lookup[q,s,Kb]
                            sign_b = sign(Lb)
                            Lb = abs(Lb)

                            if Lb == 0
                                continue
                            end

                            L = La + (Lb-1) * a_max

                            #sig[K,:] += H.h2[p,r,q,s] * v[L,:]
                            #sig[K,:] .+= H.h2[p,r,q,s] * sign_a * sign_b * v[L,:]
                            for si in 1:n_roots
                                sig[K,si] += H.h2[p,r,q,s] * sign_a * sign_b * v[L,si]
                                #@views sig[K,si] .+= H.h2[p,r,q,s] * sign_a * sign_b * v[L,si]
                            end
                            continue
                        end
                    end
                end
            end
            incr!(ket_a)

        end
        incr!(ket_b)
    end
    #sig = transpose(sig)
    return sig 
end
#=}}}=#

"""
    compute_ab_terms(v, H, P::FCIProblem, ket_a_lookup, ket_b_lookup)
"""
function compute_ab_terms(v, H, P::FCIProblem, ket_a_lookup, ket_b_lookup)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")
    size(v,1) == P.dim || throw(DimensionError())

    #v = transpose(vin)

    sig = 0*v


    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)

    a_max = bra_a.max
    reset!(ket_b)

    n_roots = size(sig,2)
    scr = zeros(1,ket_a.max*ket_b.max)
    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb-1) * ket_a.max

            #  <pq|rs> p'q'sr  --> (pr|qs) (a,b)
            for r in 1:ket_a.no
                for p in 1:ket_a.no
                    sign_a, La = ket_a_lookup[Ka][p+(r-1)*ket_a.no]
                    if La == 0
                        continue
                    end

                    Lb = 1
                    sign_b = 1
                    L = 1 
                    for s in 1:ket_b.no
                        for q in 1:ket_b.no
                            sign_b, Lb = ket_b_lookup[Kb][q+(s-1)*ket_b.no]

                            if Lb == 0
                                continue
                            end

                            L = La + (Lb-1) * a_max

                            #sig[K,:] += H.h2[p,r,q,s] * v[L,:]
                            #sig[K,:] += H.h2[p,r,q,s] * sign_a * sign_b * v[L,:]
                            for si in 1:n_roots
                                sig[K,si] += H.h2[p,r,q,s] * sign_a * sign_b * v[L,si]
                                #@views sig[K,si] .+= H.h2[p,r,q,s] * sign_a * sign_b * v[L,si]
                            end
                            continue
                        end
                    end
                end
            end
            incr!(ket_a)

        end
        incr!(ket_b)
    end
    #sig = transpose(sig)
    return sig 
end
#=}}}=#

"""
    precompute_spin_diag_terms(H, P::FCIProblem, e)
"""
function precompute_spin_diag_terms(H, P::FCIProblem, e)
    #={{{=#

    #   Create local references to ci_strings
    ket = DeterminantString(P.no, e)
    bra = DeterminantString(P.no, e)

    ket_ca_lookup = fill_ca_lookup(ket)

    Hout = zeros(ket.max,ket.max)

    reset!(ket)

    for K in 1:ket.max

        #  hpq p'q 
        for p in 1:ket.no
            for q in 1:ket.no
                bra = deepcopy(ket)
                apply_annihilation!(bra,q)
                if bra.sign == 0
                    continue
                end
                apply_creation!(bra,p)
                if bra.sign == 0
                    continue
                end

                L = calc_linear_index(bra)

                term = H.h1[q,p]
                Hout[K,L] += term * bra.sign
            end
        end


        #  <pq|rs> p'q'sr -> (pr|qs) 
        for r in 1:ket.no
            for s in r+1:ket.no
                for p in 1:ket.no
                    for q in p+1:ket.no

                        bra = deepcopy(ket)

                        apply_annihilation!(bra,r) 
                        if bra.sign == 0
                            continue
                        end
                        apply_annihilation!(bra,s) 
                        if bra.sign == 0
                            continue
                        end
                        apply_creation!(bra,q) 
                        if bra.sign == 0
                            continue
                        end
                        apply_creation!(bra,p) 
                        if bra.sign == 0
                            continue
                        end
                        L = calc_linear_index(bra)
                        Ipqrs = H.h2[p,r,q,s]-H.h2[p,s,q,r]
                        Hout[K,L] += bra.sign*Ipqrs
#                        if bra.sign == -1
#                            Hout[K,L] -= Ipqrs
#                        elseif bra.sign == +1
#                            Hout[K,L] += Ipqrs
#                        else
#                            throw(Exception())
#                        end
                    end
                end
            end
        end
        incr!(ket)
    end
    return Hout
end
#=}}}=#


"""
    get_map(ham, prb::FCIProblem, HdiagA, HdiagB)

Assumes you've already computed the spin diagonal components
"""
function get_map(ham, prb::FCIProblem, HdiagA, HdiagB)
    #=
    Get LinearMap with takes a vector and returns action of H on that vector
    =#
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,ket_a.max*ket_b.max, nr)
        else 
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ham, prb, lookup_a, lookup_b)
        @tensor begin
            sig[I,J,s] += HdiagA[I,K] * v[K,J,s]
            sig[I,J,s] += HdiagB[J,K] * v[I,K,s]
        end

        v = reshape(v, ket_a.max*ket_b.max, nr)
        sig = reshape(sig, ket_a.max*ket_b.max, nr)
        return sig 
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#


"""
    get_map(ham, prb::FCIProblem)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ham, prb::FCIProblem)
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    #@btime lookup_a = $fill_ca_lookup2($ket_a)
    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,ket_a.max*ket_b.max, nr)
        else 
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ham, prb, lookup_a, lookup_b)
        sig += compute_ss_terms2(v, ham, prb, lookup_a, lookup_b)

        v = reshape(v, ket_a.max*ket_b.max, nr)
        sig = reshape(sig, ket_a.max*ket_b.max, nr)
        return sig 
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#


"""
    run_fci(ints, prb::FCIProblem)

input: ints is a struct containing 0, 2, and 4 dimensional tensors
- `h0`: energy shift
- `h1`: 1 electron integrals
- `h2`: 2 electron integrals (chemists notation)
- `prb`: FCIProblem just defines the current CI problem (i.e., fock sector)

ints is simply an InCoreInts object from FermiCG

"""
function run_fci(ints, problem::FCIProblem; v0=nothing, nroots=1, tol=1e-6,
                precompute_ss = false)

    T = eltype(ints.h0)

    if precompute_ss
        print(" Compute spin_diagonal terms\n")
        @time Hdiag_a = StringCI.precompute_spin_diag_terms(ints,problem,problem.na)
        @time Hdiag_b = StringCI.precompute_spin_diag_terms(ints,problem,problem.nb)
        print(" done\n")

        Hmap = StringCI.get_map(ints, problem, Hdiag_a, Hdiag_b)
    else
        Hmap = StringCI.get_map(ints, problem)
    end
    
    e = 0
    v = Array{T,2}
    if v0 == nothing
        @time e,v = eigs(Hmap, nev = nroots, which=:SR, tol=tol)
        e = real(e)
        for ei in e
            @printf(" Energy: %12.8f\n",ei+ints.h0)
        end
    else
        @time e,v = eigs(Hmap, v0=v0[:,1], nev = nroots, which=:SR, tol=tol)
        e = real(e)
        for ei in e
            @printf(" Energy: %12.8f\n",ei+ints.h0)
        end
    end
    return e,v 
end


"""
    build_s2(prb::FCIProblem)
- `prb`: FCIProblem just defines the current CI problem (i.e., fock sector)
"""
function build_S2_matrix(P::FCIProblem)

    #={{{=#
    S2 = zeros(P.dim, P.dim)


    #   Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)


    #   lookup the ket space
    ket_a_lookup = fill_ca_lookup(ket_a)
    ket_b_lookup = fill_ca_lookup(ket_b)

    reset!(ket_b)
    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb-1) * ket_a.max

	    # Sz.Sz

            for ai in ket_a.config
                for aj in ket_a.config
                    if ai != aj
                        S2[K,K] += 0.25
                    end
                end
            end

            for bi in ket_b.config
                for bj in ket_b.config
                    if bi != bj
                        S2[K,K] += 0.25
                    end
                end
            end

            for ai in ket_a.config
                for bj in ket_b.config
                    if ai != bj
                        S2[K,K] -= 0.50
                    end
                end
            end

	    # Sp.Sm
            for ai in ket_a.config
                if ai in ket_b.config
		    temp = 10
		else
                    S2[K,K] += 0.75
                end
            end


	    # Sm.Sp
            for bi in ket_b.config
                if bi in ket_a.config
		    temp = 10
		else
                    S2[K,K] += 0.75
                end
            end


            ket_a2 = DeterminantString(P.no, P.na)
            ket_b2 = DeterminantString(P.no, P.nb)

	    for ai in ket_a.config
	        for bj in ket_b.config
	            if ai ∉ ket_b.config
	                if bj ∉ ket_a.config

	        	    ket_a2 = deepcopy(ket_a)
	        	    ket_b2 = deepcopy(ket_b)

                            apply_annihilation!(ket_a2,ai)
	        	    ket_a2.sign != 0 || continue

                            apply_creation!(ket_b2,ai)
	        	    ket_b2.sign != 0 || continue

                            apply_creation!(ket_a2,bj)
	        	    ket_a2.sign != 0 || continue

                            apply_annihilation!(ket_b2,bj)
	        	    ket_b2.sign != 0 || continue

	        	    sign_a = ket_a2.sign
	        	    sign_b = ket_b2.sign

                            La = calc_linear_index(ket_a2)
                            Lb = calc_linear_index(ket_b2)

                            L = La + (Lb-1) * ket_a.max
			    #print("Init ",ket_a.config,"    ",ket_b.config,"\n")
			    #print("Final",ket_a2.config,"    ",ket_b2.config,"\n")
    	        	    #print(K,"  ",L,"\n")
                            S2[K,L] += 1 * sign_a * sign_b
	        	end
	            end
	        end
	    end


            incr!(ket_a)

        end
        incr!(ket_b)
    end
    return S2
end
#=}}}=#


"""
    svd_state(prb::FCIProblem)
Do an SVD of the FCI vector partitioned into clusters with (norbs1 | norbs2)
where the orbitals are assumed to be ordered for cluster 1| cluster 2 haveing norbs1 and 
norbs2, respectively.

- `prb`: FCIProblem just defines the current CI problem (i.e., fock sector)
- `norbs1`:number of orbitals in left cluster
- `norbs2`:number of orbitals in right cluster
- `svd_thresh`: the threshold below which the states will be discarded
"""
function svd_state(v,P::FCIProblem,norbs1,norbs2,svd_thresh)

    #={{{=#

    @assert(norbs1+norbs2 ==P.no)

    schmidt_basis = OrderedDict()
    #vector = OrderedDict{Tuple{UInt8,UInt8},Float64}()
    vector = OrderedDict{Tuple{Int,Int},Any}()

    #schmidt_basis = Dict{Tuple,Matrix{Float64}}

    println("----------------------------------------")
    println("          SVD of state")
    println("----------------------------------------")

    # Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)

    v = reshape(v,(ket_a.max, ket_b.max))
    @assert(size(v,1) == ket_a.max)
    @assert(size(v,2) == ket_b.max)

    fock_labels_a = Array{Int,1}(undef,ket_a.max)
    fock_labels_b = Array{Int,1}(undef,ket_b.max)


    # Get the fock space using the bisect method in python
    bisect = pyimport("bisect")
    for I in 1:ket_a.max
        fock_labels_a[I] = bisect.bisect(ket_a.config,norbs1)
        #print("nick: ", ket_a.config, " " , norbs1, " ", fock_labels_a[I], "\n")
        incr!(ket_a)
    end
    for I in 1:ket_b.max
        fock_labels_b[I] = bisect.bisect(ket_b.config,norbs1)
        incr!(ket_b)
    end
    for J in 1:ket_b.max
        for I in 1:ket_a.max
            fock = (fock_labels_a[I], fock_labels_b[J])

            #if fock in vector
            #    append!(vector[fock],v[I,J])
            #else
            #    vector[fock] = [v[I,J]]
            #end
            try
                append!(vector[tuple(fock_labels_a[I],fock_labels_b[J])],v[I,J])
            catch
                vector[tuple(fock_labels_a[I],fock_labels_b[J])] = [v[I,J]]
            end
        end
    end

    for (fock,fvec)  in vector

        println()
        @printf("Prepare Fock Space:  %iα, %iβ\n",fock[1] ,fock[2] )

        ket_a1 = DeterminantString(norbs1, fock[1])
        ket_b1 = DeterminantString(norbs1, fock[2])

        ket_a2 = DeterminantString(norbs2, P.na - fock[1])
        ket_b2 = DeterminantString(norbs2, P.nb - fock[2])


        temp_fvec = reshape(fvec,ket_b1.max*ket_b2.max,ket_a1.max*ket_a2.max)'
        #temp_fvec = reshape(fvec,ket_b1.max*ket_b2.max,ket_a1.max*ket_a2.max)'
        #st = "temp_fvec"*string(fock)
        #npzwrite(st, temp_fvec)


        #when swapping alpha2 and beta1 do we flip sign?
        sign = 1
        if (P.na-fock[1])%2==1 && fock[2]%2==1
            sign = -1
        end
        #println("sign",sign)
        @printf("   Dimensions: %5i x %-5i \n",ket_a1.max*ket_b1.max, ket_a2.max*ket_b2.max)

        norm_curr = fvec' * fvec
        @printf("   Norm: %12.8f\n",sqrt(norm_curr))
        #println(size(fvec))
        #display(fvec)

        fvec = sign *fvec

        #opposite to python with transpose on fvec
        #fvec2 = reshape(fvec',ket_b2.max,ket_b1.max,ket_a2.max,ket_a1.max)
        fvec2 = reshape(fvec,ket_a1.max,ket_a2.max,ket_b1.max,ket_b2.max)
        fvec3 = permutedims(fvec2, [ 1, 3, 2, 4])
        fvec4 = reshape(fvec3,ket_a1.max*ket_b1.max,ket_a2.max*ket_b2.max)

        # fvec4 is transpose of what we have in python code
        fvec5 = fvec4'

        F = svd(fvec5,full=true)


        nkeep = 0
        @printf("   %5s %12s\n","State","Weight")
        for (ni_idx,ni) in enumerate(F.S)
            if ni > svd_thresh
                nkeep += 1
                @printf("   %5i %12.8f\n",ni_idx,ni)
            else
                @printf("   %5i %12.8f (discarded)\n",ni_idx,ni)
            end
        end
        

        if nkeep > 0
            schmidt_basis[fock] = Matrix(F.U[:,1:nkeep])
            #st = "fin_vec"*string(fock)
            #npzwrite(st, F.U[:,1:nkeep])
        end

        #norm += norm_curr
    end

    return schmidt_basis
end
#=}}}=#



"""
    function do_fci(problem::FCIProblem, ints, nr; v0=Nothing, tol=1e-12)

Use Arpack.eigs to diagonalize the problem
- `problem`: FCIProblem to solve
- `ints`: InCoreIntegrals
- `nr`: number of roots 
- `v0`: Initial vector
- `tol`: convergence tolerance
"""
function do_fci(problem::FCIProblem, ints, nr; v0=Nothing, tol=1e-12)
    #={{{=#
    Hmap = get_map(ints, problem)
    if v0 == Nothing
        e, v = Arpack.eigs(Hmap, nev = nr, which=:SR, tol=tol)
        e = real(e)[1:nr]
        return e, v[:,1:nr]
    else
        e, v = Arpack.eigs(Hmap, nev = nr, which=:SR, v0=v0, tol=tol)
        e = real(e)[1:nr]
        return e, v[:,1:nr]
    end
end
#=}}}=#


"""
"""
function compute_1rdm(problem::FCIProblem, vl::Vector{T}, vr::Vector{T}) where T
    #={{{=#


    rdma = compute_Aa(problem.no,                    
                     problem.na, problem.nb,
                     problem.na, problem.nb,
                     reshape(vl, length(vl), 1), 
                     reshape(vr, length(vr), 1), 
                    "alpha") 
   
    rdmb = compute_Aa(problem.no,                    
                     problem.na, problem.nb,
                     problem.na, problem.nb,
                     reshape(vl, length(vl), 1), 
                     reshape(vr, length(vr), 1), 
                    "beta") 
   
     
    rdma = reshape(rdma, problem.no, problem.no)
    rdmb = reshape(rdmb, problem.no, problem.no)
    return rdma, rdmb
end
#=}}}=#



"""
"""
function compute_rdm1_rdm2(P::FCIProblem, vec_l::Vector{T}, vec_r::Vector{T}) where T
    #={{{=#

    no = P.no
    na = P.na
    nb = P.nb

    rdm1a = zeros(T, no, no)
    rdm1b = zeros(T, no, no)
    rdm2aa = zeros(T, no, no, no, no)
    rdm2bb = zeros(T, no, no, no, no)
    rdm2ab = zeros(T, no, no, no, no)

    #   Create local references to ci_strings
    ket_a = DeterminantString(no, na)
    bra_a = DeterminantString(no, na)
    ket_b = DeterminantString(no, nb)
    bra_b = DeterminantString(no, nb)

    ket_a_lookup = fill_ca_lookup(ket_a)
    ket_b_lookup = fill_ca_lookup(ket_b)

    vl = reshape(vec_l, bra_a.max, bra_b.max)
    vr = reshape(vec_r, ket_a.max, ket_b.max)

    #################
    #   a
    #################
    reset!(ket_a)
    for Ka in 1:ket_a.max

        #  p'q 
        for p in 1:ket_a.no
            for q in 1:ket_a.no
                bra = deepcopy(ket_a)

                apply_annihilation!(bra,q)
                bra.sign != 0 || continue
                apply_creation!(bra,p)
                bra.sign != 0 || continue

                L = calc_linear_index(bra)

                if bra.sign == 1
                    @views rdm1a[p,q] += dot(vl[L,:], vr[Ka,:])
                elseif bra.sign == -1
                    @views rdm1a[p,q] -= dot(vl[L,:], vr[Ka,:])
                else
                    error(" Shouldn't be here")
                end
            end
        end
        incr!(ket_a)
    end
    #################
    #   aa
    #################
    reset!(ket_a)
    for Ka in 1:ket_a.max

        #  p'q'rs 
        for p in 1:ket_a.no
            for q in 1:ket_a.no
                for r in 1:ket_a.no
                    for s in 1:ket_a.no
                        bra = deepcopy(ket_a)

                        apply_annihilation!(bra,s)
                        bra.sign != 0 || continue
                        apply_annihilation!(bra,r)
                        bra.sign != 0 || continue
                        apply_creation!(bra,q)
                        bra.sign != 0 || continue
                        apply_creation!(bra,p)
                        bra.sign != 0 || continue

                        L = calc_linear_index(bra)

                        if bra.sign == 1
                            @views rdm2aa[p,q,r,s] += dot(vl[L,:], vr[Ka,:])
                        elseif bra.sign == -1
                            @views rdm2aa[p,q,r,s] -= dot(vl[L,:], vr[Ka,:])
                        else
                            error(" Shouldn't be here")
                        end
                    end
                end
            end
        end
        incr!(ket_a)
    end
    #################
    #   b
    #################
    reset!(ket_b)
    for Kb in 1:ket_b.max

        #  p'q 
        for p in 1:ket_b.no
            for q in 1:ket_b.no
                bra = deepcopy(ket_b)

                apply_annihilation!(bra,q)
                bra.sign != 0 || continue
                apply_creation!(bra,p)
                bra.sign != 0 || continue

                L = calc_linear_index(bra)

                if bra.sign == 1
                    @views rdm1b[p,q] += dot(vl[:,L], vr[:,Kb])
                elseif bra.sign == -1
                    @views rdm1b[p,q] -= dot(vl[:,L], vr[:,Kb])
                else
                    error(" Shouldn't be here")
                end
            end
        end
        incr!(ket_b)
    end
    #################
    #   bb
    #################
    reset!(ket_b)
    for Kb in 1:ket_b.max

        #  p'q'rs 
        for p in 1:ket_b.no
            for q in 1:ket_b.no
                for r in 1:ket_b.no
                    for s in 1:ket_b.no
                        bra = deepcopy(ket_b)

                        apply_annihilation!(bra,s)
                        bra.sign != 0 || continue
                        apply_annihilation!(bra,r)
                        bra.sign != 0 || continue
                        apply_creation!(bra,q)
                        bra.sign != 0 || continue
                        apply_creation!(bra,p)
                        bra.sign != 0 || continue

                        L = calc_linear_index(bra)

                        if bra.sign == 1
                            @views rdm2bb[p,q,r,s] += dot(vl[:,L], vr[:,Kb])
                        elseif bra.sign == -1
                            @views rdm2bb[p,q,r,s] -= dot(vl[:,L], vr[:,Kb])
                        else
                            error(" Shouldn't be here")
                        end
                    end
                end
            end
        end
        incr!(ket_b)
    end
    #################
    #   ab
    #################
    reset!(ket_b)
    for Kb in 1:ket_b.max
        reset!(ket_a)
        for Ka in 1:ket_a.max

            #  p'q'rs (abba) 
            for p in 1:ket_a.no
                for s in 1:ket_a.no
                    sign_a, La = ket_a_lookup[Ka][p+(s-1)*ket_a.no]
                   
                    La != 0 || continue

                    for q in get_unoccupied(ket_b) 
                        for r in ket_b.config
                    
                            sign_b, Lb = ket_b_lookup[Kb][q+(r-1)*ket_b.no]
                            Lb != 0 || continue
                  
                            sign = sign_a*sign_b
                            rdm2ab[p,q,r,s] += vl[La,Lb] * vr[Ka,Kb] * sign
                        end
                    end
                end
            end
            incr!(ket_a)
        end
        incr!(ket_b)
    end

    vl = reshape(vl, bra_a.max * bra_b.max)
    vr = reshape(vr, ket_a.max * ket_b.max)

    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab
end
#=}}}=#


