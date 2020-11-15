using LinearAlgebra 
using Printf
using Parameters
using Profile
using LinearMaps



struct FCIProblem
    no::Integer  # number of orbitals
    na::Integer  # number of alpha
    nb::Integer  # number of beta
    dima::Integer 
    dimb::Integer 
    dim::Integer
    converged::Bool
    restarted::Bool
    iteration::Integer
    algorithm::String   #  options: direct/davidson
    n_roots::Integer
end

function FCIProblem(no, na, nb)
    dima = calc_nchk(no,na)
    dimb = calc_nchk(no,nb)
    return FCIProblem(no, na, nb, dima, dimb, dima*dimb, false, false, 1, "direct", 1)
end

function display(p::FCIProblem)
    @printf(" FCIProblem::  NOrbs: %2i NAlpha: %2i NBeta: %2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
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
    Hmat += compute_ab_terms_full(ints, P)
    
    Hmat = .5*(Hmat+Hmat')

    return Hmat
end



"""
    compute_fock_diagonal!(H, P::FCIProblem, e_mf::Float64)
"""
function compute_fock_diagonal(P::FCIProblem, orb_energies::Vector{Float64}, e_mf::Float64)
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
function compute_ab_terms_full(H, P::FCIProblem)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")

    #v = transpose(vin)

    Hmat = zeros(Float64, P.dim, P.dim)

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
                    La = ket_a_lookup2[Ka,p,r]
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
                            Lb = ket_b_lookup2[Kb,q,s]
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

"""
    compute_ab_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup, ket_b_lookup)
"""
function compute_ab_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup, ket_b_lookup)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")
    @assert(size(v,1)*size(v,2) == P.dim)

    #v = transpose(vin)

    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)

    no = ket_a.no

    #ket_a_lookup = fill_ca_lookup2(ket_a)
    #ket_b_lookup = fill_ca_lookup2(ket_b)

    a_max::Int = bra_a.max
    reset!(ket_b)

    #
    #   sig3(Ia,Ib,s) = <Ia|k'l|Ja> <Ib|i'j|Jb> V(ij,kl) C(Ja,Jb,s)
    n_roots::Int = size(v,3)
    #v = reshape(v,ket_a.max, ket_b.max, n_roots) 
    sig = zeros(Float64, ket_a.max, ket_b.max, n_roots) 
    FJb = zeros(Float64, ket_b.max) 
    for k in 1:ket_a.no,  l in 1:ket_a.no
        #@printf(" %4i, %4i\n",k,l)
        L = Vector{Int}()
        R = Vector{Int}()
        for (Iidx,I) in enumerate(ket_a_lookup[:,k,l])
            if I != 0
                push!(L,I)
                push!(R,Iidx)
            end
        end
        VI = zeros(Float64, length(L),n_roots)
        #println(L)
        #println(R)
        Ckl = zeros(Float64, size(v)[2], length(L), size(v)[3])
        for Li in 1:length(L)
            @views Ckl[:,Li,:] = v[abs(L[Li]), :, :] * sign(L[Li])
        end
        #Vkl_ij = H.h2[:,:,k,l]
        for Ib in 1:ket_b.max
            FJb .= 0 
            Jb = 1
            sgn::Int = 0
            for i=1:no, j=1:no
                Jb = ket_b_lookup[Ib,i,j]
                if Jb != 0
                    sgn = sign(Jb)
                    Jb = abs(Jb)
                    @inbounds FJb[Jb] += H.h2[j,i,l,k]*sgn
                end
            end
            @tensor begin
                VI[I,s] = FJb[J] * Ckl[J,I,s]
            end
            #for I = 1:length(L), s = 1:n_roots
            #    @views VI[I,s] = FJb[:]' * Ckl[:,I,s]
            #end
            #println(size(sig), size(VI))
            for si in 1:n_roots
                for Li in 1:length(L)
                    @inbounds sig[R[Li],Ib,si] += VI[Li,si] 
                end
            end
            #                @views sig[R,Ib,:] .+= VI[:,:]
        end
    end

    #v = reshape(v,ket_a.max*ket_b.max, n_roots) 
    #sig = reshape(sig,ket_a.max*ket_b.max, n_roots) 

    return sig
    

end
#=}}}=#

"""
    compute_aa_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup)
"""
function compute_ss_terms2(v, H, P::FCIProblem, 
                          ket_a_lookup, ket_b_lookup)
    #={{{=#

    #print(" Compute opposite spin terms. Shape of v: ", size(v), "\n")
    @assert(size(v,1)*size(v,2) == P.dim)

    #v = transpose(vin)

    #   Create local references to ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)

    no = ket_a.no

    a_max::Int = bra_a.max
    reset!(ket_a)

    #
    #   sig1(Ia,Ib,s) = <Ib|i'j|Jb> V(ij,kl) C(Ja,Jb,s)
    n_roots::Int = size(v,3)
    #v = reshape(v,ket_a.max, ket_b.max, n_roots) 
    sig = zeros(Float64, ket_a.max, ket_b.max, n_roots) 
    size(sig) == size(v) || throw(DimensionError())
#    for K in 1:ket_a.max
#        #  hpq p'q 
#        for p in 1:ket_a.no, q in 1:ket_a.no
#            L = ket_a_lookup[K,p,q]
#            if L==0
#                continue
#            end
#            sgn = sign(L)
#            L = abs(L)
#
#            @views sig[K,:,:] += H.h1[q,p] * sgn * v[L,:,:]
#        end
#    end

    h1eff = deepcopy(H.h1)
    @tensor begin
        h1eff[p,q] -= .5 * H.h2[p,j,j,q]  
    end

    #aa
    ket = ket_a
    reset!(ket) 
    F = zeros(ket_a.max)
    for I in 1:ket.max
        F .= 0
        for k in 1:ket.no, l in 1:ket.no
            K = ket_a_lookup[I,k,l]
            if K == 0
                continue
            end
            sign_kl = sign(K)
            K = abs(K)

            bra = deepcopy(ket)
            apply_annihilation!(bra,l) 
            if bra.sign == 0
                continue
            end
            apply_creation!(bra,k) 
            if bra.sign == 0
                continue
            end
            @inbounds F[K] += sign_kl * h1eff[k,l]
            #@views sig[:,I,:] += H.h1[k,l] * sign_kl * v[:,K,:]
            for i in 1:ket.no, j in 1:ket.no
                J = ket_a_lookup[K,i,j]
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
        @views tmp = sig[I,:,:]
        @tensor begin
            tmp[K,s] += F[J] * v[J,K,s]
        end
        incr!(ket)
    end

    #bb
    ket = ket_b
    reset!(ket) 
    F = zeros(ket_b.max)
    for I in 1:ket.max
        F .= 0
        for k in 1:ket.no, l in 1:ket.no
            K = ket_b_lookup[I,k,l]
            if K == 0
                continue
            end
            sign_kl = sign(K)
            K = abs(K)

            bra = deepcopy(ket)
            apply_annihilation!(bra,l) 
            if bra.sign == 0
                continue
            end
            apply_creation!(bra,k) 
            if bra.sign == 0
                continue
            end
            @inbounds F[K] += sign_kl * h1eff[k,l]
            #@views sig[:,I,:] += H.h1[k,l] * sign_kl * v[:,K,:]
            for i in 1:ket.no, j in 1:ket.no
                J = ket_b_lookup[K,i,j]
                if J == 0
                    continue
                end
                sign_ij = sign(J)
                J = abs(J)
                #@inbounds F[J] += .5 *sign_kl * sign_ij * H.h2[i,j,k,l]
                if sign_kl == sign_ij
                    @inbounds F[J] += .5 * H.h2[i,j,k,l]
                else
                    @inbounds F[J] -= .5 * H.h2[i,j,k,l]
                end
            end
        end
        @views tmp = sig[:,I,:]
        @tensor begin
            tmp[K,s] += F[J] * v[K,J,s]
        end
        incr!(ket)
    end

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
                    La = ket_a_lookup[Ka,p,r]
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
                            Lb = ket_b_lookup[Kb,q,s]
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
        - h0: energy shift
        - h1: 1 electron integrals
        - h2: 2 electron integrals (chemists notation)
"""
function run_fci(ints, problem::FCIProblem; v0=nothing, nroots=1, tol=1e-6,
                precompute_ss = false)

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
    v = Array{Float64,2}
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

