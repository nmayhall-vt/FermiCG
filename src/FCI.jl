using LinearAlgebra 
using Printf
using Parameters
using Profile
using LinearMaps

using .Helpers


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
    dima = Helpers.calc_nchk(no,na)
    dimb = Helpers.calc_nchk(no,nb)
    return FCIProblem(no, na, nb, dima, dimb, dima*dimb, false, false, 1, "direct", 1)
end

function display(p::FCIProblem)
    @printf(" FCIProblem::  NOrbs: %2i NAlpha: %2i NBeta: %2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end

function compute_spin_diag_terms_full!(H::ElectronicInts, P::FCIProblem, Hmat)
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
    build_H_matrix(ints::ElectronicInts, P::FCIProblem)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis 
in the sector of Fock space specified by `P`
"""
function build_H_matrix(ints::ElectronicInts, P::FCIProblem)

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



function compute_ab_terms_full!(H::ElectronicInts, P::FCIProblem, Hmat)
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


function compute_ab_terms_full(H::ElectronicInts, P::FCIProblem)
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


function compute_ab_terms(v, H::ElectronicInts, P::FCIProblem)
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

    ket_a_lookup = fill_ca_lookup(ket_a)
    ket_b_lookup = fill_ca_lookup(ket_b)

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


function compute_ab_terms(v, H::ElectronicInts, P::FCIProblem, ket_a_lookup, ket_b_lookup)
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


function precompute_spin_diag_terms(H::ElectronicInts, P::FCIProblem, e)
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


function get_map(ham::ElectronicInts, prb::FCIProblem, HdiagA, HdiagB)
    #=
    Get LinearMap with takes a vector and returns action of H on that vector
    =#
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    lookup_a = fill_ca_lookup(ket_a)
    lookup_b = fill_ca_lookup(ket_b)

    function mymatvec(v)
        @time sig = compute_ab_terms(v, ham, prb)
        #sig = compute_ab_terms(v, ham, prb, lookup_a, lookup_b)
        sig += HdiagA * v
        sig += HdiagB * v
        return sig 
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#

