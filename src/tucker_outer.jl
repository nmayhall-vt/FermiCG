using Profile
using LinearMaps
using BenchmarkTools
using IterativeSolvers
#using TensorDecompositions





"""
    get_map(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; shift = nothing, cache=false)
    #={{{=#
    iters = 0
    
    dim = length(ci_vector)
    function mymatvec(v)
        iters += 1

        set_vector!(ci_vector, v)

        #fold!(ci_vector)
        sig = deepcopy(ci_vector)
        zero!(sig)
        build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)

        #unfold!(ci_vector)

        sig = get_vector(sig)

        if shift != nothing
            # this is how we do CEPA
            sig += shift * get_vector(ci_vector)
        end
        flush(stdout)

        return sig
    end
    return LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#

"""
    tucker_ci_solve(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5)

Solve for ground state in the space spanned by `ci_vector`'s compression vectors
"""
function tucker_ci_solve(ci_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5)
#={{{=#

    @printf(" Solve CI with # variables = %i\n", length(ci_vector))
    vec = deepcopy(ci_vector)
    normalize!(vec)
    #flush term cache
    flush_cache(clustered_ham)
    
    Hmap = get_map(vec, cluster_ops, clustered_ham, cache=true)

    v0 = get_vector(vec)
    nr = size(v0)[2]
   

    cache=true
    if cache
        @time cache_hamiltonian(vec, vec, cluster_ops, clustered_ham)
    end


    #cache_hamiltonian(ci_vector, ci_vector, cluster_ops, clustered_ham)
    
    davidson = Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=tol)
    #Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    e,v = FermiCG.solve(davidson)
    set_vector!(vec,v)
    
    println(" Memory used by cache: ", mem_used_by_cache(clustered_ham))

    #flush term cache
    flush_cache(clustered_ham)

    return e,vec
end
#=}}}=#







"""
    tucker_cepa_solve!(ref_vector::CompressedTuckerState, cepa_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5, cache=true)

# Arguments
- `ref_vector`: Input reference state. 
- `cepa_vector`: CompressedTuckerState which defines the configurational space defining {X}. This 
should be the first-order interacting space (or some compressed version of it).
- `cluster_ops`
- `clustered_ham`
- `tol`: haven't yet set this up (NYI)
- `cache`: Should we cache the compressed H operators? Speeds up drastically, but uses lots of memory

Compute compressed CEPA.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |1 | = |I   + Sax| |1 | E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa + Hax*Cx = (1 + Sax*Cx)E
Hxa + HxxCx = SxaE + CxE

The idea for CEPA is to approximate E in the amplitude equation.
CEPA(0): E = Eref

(Hxx-Eref)*Cx = Sxa*Eref - Hxa

Ax=b

After solving, the Energy can be obtained as:
E = (Eref + Hax*Cx) / (1 + Sax*Cx)
"""
function tucker_cepa_solve(ref_vector::CompressedTuckerState, cepa_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5, cache=true, max_iter=30, verbose=false)
#={{{=#

    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham, cache=false)
    e0 = nonorth_dot(ref_vector, sig)
    length(e0) == 1 || error("Only one state at a time please", e0)
    e0 = e0[1]
    @printf(" Reference Energy: %12.8f\n",e0)


    x_vector = deepcopy(cepa_vector)
    a_vector = deepcopy(ref_vector)


#    project_out!(x_vector, a_vector)
#    #
#    # Project out reference space
#    for (fock,tconfigs) in x_vector 
#        for (tconfig, tuck) in tconfigs
#            if haskey(ref_vector, fock)
#                if haskey(ref_vector[fock], tconfig)
#                    ref_tuck = ref_vector[fock][tconfig]
#
#                    ovlp = nonorth_dot(tuck, ref_tuck) / nonorth_dot(ref_tuck, ref_tuck)
#                    tmp = scale(ref_tuck, -1.0 * ovlp)
#                    x_vector[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
#                end
#            end
#        end
#    end

    b = deepcopy(x_vector)
    zero!(b)
    build_sigma!(b, ref_vector, cluster_ops, clustered_ham, cache=false)
    bv = -get_vector(b)

    @printf(" Overlap between <0|0>:          %18.12e\n", nonorth_dot(ref_vector, ref_vector, verbose=0))
    @printf(" Overlap between <1|0>:          %18.12e\n", nonorth_dot(x_vector, ref_vector, verbose=0))
    @printf(" Overlap between <1|1>:          %18.12e\n", nonorth_dot(x_vector, x_vector, verbose=0))
    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(x_vector)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        for (tconfig, tuck) in tconfigs
            if haskey(ref_vector, fock)
                if haskey(ref_vector[fock], tconfig)
                    ref_tuck = ref_vector[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = []
                    for i in 1:length(Sx.clusters)
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    Sx[fock][tconfig].core .= transform_basis(ref_tuck.core, overlaps)
                end
            end
        end
    end
    @printf(" Norm of Sx overlap: %18.12f\n", orth_dot(Sx,Sx))
    @printf(" Norm of b         : %18.12f\n", sum(bv.*bv))

    bv .= bv .+ get_vector(Sx)*e0

    @printf(" Norm of b         : %18.12f\n", sum(bv.*bv))

    function mymatvec(v)
        set_vector!(x_vector, v)
        #@printf(" Overlap between <1|0>:          %8.1e\n", nonorth_dot(x_vector, ref_vector, verbose=0))
        sig = deepcopy(x_vector)
        zero!(sig)
        #build_sigma!(sig, x_vector, cluster_ops, clustered_ham, cache=false)
        build_sigma!(sig, x_vector, cluster_ops, clustered_ham, cache=cache)

        tmp = deepcopy(x_vector)
        scale!(tmp, -e0)
        orth_add!(sig, tmp)
        return get_vector(sig)
    end
    
    dim = length(x_vector)
    Axx = LinearMap(mymatvec, dim, dim)
    #Axx = LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)

    #flush term cache
    println(" Now flushing:")
    flush_cache(clustered_ham)

    if cache
        @time cache_hamiltonian(x_vector, x_vector, cluster_ops, clustered_ham)
    end
   
    println(" Start CEPA iterations with dimension = ", length(x_vector))
    x, solver = cg!(get_vector(x_vector), Axx,bv,log=true, maxiter=max_iter, verbose=verbose, abstol=tol)
    
    #flush term cache
    println(" Now flushing:")
    flush_cache(clustered_ham)

    set_vector!(x_vector, x)

    SxC = nonorth_dot(Sx,x_vector)
    @printf(" <A|X>C(X) = %18.12e\n", SxC)

    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(sig,ref_vector)
    @printf(" Cepa: %18.12f\n", ecorr)
    
    sig = deepcopy(x_vector)
    zero!(sig)
    build_sigma!(sig,ref_vector, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(sig,x_vector)
    @printf(" Cepa: %18.12f\n", ecorr)
    
    length(ecorr) == 1 || error(" Dimension Error", ecorr)
    ecorr = ecorr[1]
    @printf(" <1|1> = %18.12f\n", orth_dot(x_vector,x_vector))
    @printf(" <1|1> = %18.12f\n", sum(x.*x))

    @printf(" E(CEPA) = %18.12f\n", (e0 + ecorr)/(1+SxC))

    #x, info = linsolve(Hmap,zeros(size(v0)))
    return (ecorr+e0)/(1+SxC), x_vector 
end#=}}}=#

"""
    tucker_cepa_solve!(ref_vector::CompressedTuckerState, cepa_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5, cache=true)

# Arguments
- `ref_vector`: Input reference state. 
- `cepa_vector`: CompressedTuckerState which defines the configurational space defining {X}. This 
should be the first-order interacting space (or some compressed version of it).
- `cluster_ops`
- `clustered_ham`
- `tol`: haven't yet set this up (NYI)
- `cache`: Should we cache the compressed H operators? Speeds up drastically, but uses lots of memory

Compute compressed CEPA.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |1 | = |I   + Sax| |1 | E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa + Hax*Cx = (1 + Sax*Cx)E
Hxa + HxxCx = SxaE + CxE

The idea for CEPA is to approximate E in the amplitude equation.
CEPA(0): E = Eref

(Hxx-Eref)*Cx = Sxa*Eref - Hxa

Ax=b

After solving, the Energy can be obtained as:
E = (Eref + Hax*Cx) / (1 + Sax*Cx)
"""
function tucker_cepa_solve2(ref_vector::CompressedTuckerState, cepa_vector::CompressedTuckerState, cluster_ops, clustered_ham; tol=1e-5, cache=true, max_iter=30, verbose=false, do_pt2=false)
#={{{=#
    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham, cache=false)
    e0 = nonorth_dot(ref_vector, sig)
    length(e0) == 1 || error("Only one state at a time please", e0)
    e0 = e0[1]
    @printf(" Reference Energy: %12.8f\n",e0)

    e0_1b = 0.0
    if do_pt2
        sig = deepcopy(ref_vector)
        zero!(sig)
        build_sigma!(sig, ref_vector, cluster_ops, clustered_ham, nbody=1)
        e0_1b = nonorth_dot(ref_vector, sig)
    end


    x_vector = deepcopy(cepa_vector)
    a_vector = deepcopy(ref_vector)


#    project_out!(x_vector, a_vector)
#    #
#    # Project out reference space
#    for (fock,tconfigs) in x_vector 
#        for (tconfig, tuck) in tconfigs
#            if haskey(ref_vector, fock)
#                if haskey(ref_vector[fock], tconfig)
#                    ref_tuck = ref_vector[fock][tconfig]
#
#                    ovlp = nonorth_dot(tuck, ref_tuck) / nonorth_dot(ref_tuck, ref_tuck)
#                    tmp = scale(ref_tuck, -1.0 * ovlp)
#                    x_vector[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
#                end
#            end
#        end
#    end
    @printf(" Overlap between <1|0>:          %8.1e\n", nonorth_dot(x_vector, ref_vector, verbose=0))

    b = deepcopy(x_vector)
    zero!(b)
    build_sigma!(b, ref_vector, cluster_ops, clustered_ham, cache=false)
    bv = -get_vector(b)

    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(x_vector)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        for (tconfig, tuck) in tconfigs
            if haskey(ref_vector, fock)
                if haskey(ref_vector[fock], tconfig)
                    ref_tuck = ref_vector[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = []
                    for i in 1:length(Sx.clusters)
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    Sx[fock][tconfig].core .= transform_basis(ref_tuck.core, overlaps)
                end
            end
        end
    end

    bv .= bv .+ get_vector(Sx)*e0

    @printf(" Norm of Sx overlap: %12.8f\n", orth_dot(Sx,Sx))
    @printf(" Norm of b         : %12.8f\n", sum(bv.*bv))

    function mymatvec(v)
        set_vector!(x_vector, v)
        #@printf(" Overlap between <1|0>:          %8.1e\n", nonorth_dot(x_vector, ref_vector, verbose=0))
        sig = deepcopy(x_vector)
        zero!(sig)
        #build_sigma!(sig, x_vector, cluster_ops, clustered_ham, nbody=nbody, cache=false)
        build_sigma!(sig, x_vector, cluster_ops, clustered_ham, cache=cache)

        tmp = deepcopy(x_vector)
        if do_pt2
            scale!(tmp, -e0_1b)
        else
            scale!(tmp, -e0)
        end
        orth_add!(sig, tmp)
        return get_vector(sig)
    end
    dim = length(x_vector)
    Axx = LinearMap(mymatvec, dim, dim)
    #Axx = LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)

    #flush term cache
    println(" Now flushing:")
    flush_cache(clustered_ham)
   
    println(" Start CEPA iterations with dimension = ", length(x_vector))
    x, solver = cg!(get_vector(x_vector), Axx,bv,log=true, maxiter=max_iter, verbose=verbose, abstol=tol)
    
    #flush term cache
    println(" Now flushing:")
    flush_cache(clustered_ham)

    set_vector!(x_vector, x)


    SxC = orth_dot(Sx,x_vector)
    @printf(" <A|X>C(X) = %12.3e\n", SxC)

    sig = deepcopy(ref_vector)
    zero!(sig)
    @time build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(sig,ref_vector)
    @printf(" Cepa: %12.8f\n", ecorr)
    length(ecorr) == 1 || error(" Dimension Error", ecorr)
    ecorr = ecorr[1]
    @printf(" <1|1> = %12.8f\n", orth_dot(x_vector,x_vector))

    @printf(" E(CEPA) = %12.8f\n", (e0 + ecorr)/(1+SxC))

    #x, info = linsolve(Hmap,zeros(size(v0)))
    return (ecorr+e0)/(1+SxC), x_vector 
end#=}}}=#



"""
    define_foi_space(v::CompressedTuckerState, clustered_ham; nbody=2)
Compute the first-order interacting space as defined by clustered_ham

#Arguments
- `v::CompressedTuckerState`: input state
- `clustered_ham`: Hamiltonian
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `foi::OrderedDict{FockConfig,Vector{TuckerConfig}}`

"""
function define_foi_space(cts::T, clustered_ham; nbody=2) where T<:Union{TuckerState, CompressedTuckerState}
    println(" Define the FOI space for CompressedTuckerState. nbody = ", nbody)#={{{=#

    foi_space = OrderedDict{FockConfig,Vector{TuckerConfig}}()

    for (fock, tconfigs) in cts

        for (fock_trans, terms) in clustered_ham

            #
            # new fock sector configuration
            new_fock = fock + fock_trans


            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in cts.clusters
                if new_fock[ci.idx][1] > length(ci) || new_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if new_fock[ci.idx][1] < 0 || new_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue



            #
            # find the cluster state index ranges (TuckerConfigs) reached by Hamiltonian
            for (tconfig, coeffs) in tconfigs
                for term in terms
                    new_tconfig = deepcopy(tconfig)

                    length(term.clusters) <= nbody || continue

                    new_tconfigs = []
                    tmp = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        new_tconfig[ci.idx] = cts.q_spaces[ci.idx][new_fock[ci.idx]]
                        tmp2 = []
                        if haskey(cts.p_spaces[ci.idx], new_fock[ci.idx])
                            push!(tmp2, cts.p_spaces[ci.idx][new_fock[ci.idx]])
                        end
                        if haskey(cts.q_spaces[ci.idx], new_fock[ci.idx])
                            push!(tmp2, cts.q_spaces[ci.idx][new_fock[ci.idx]])
                        end
                        push!(tmp, tmp2)
                    end

                    for prod in product(tmp...)
                        new_tconfig = deepcopy(tconfig)
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            new_tconfig[ci.idx] = prod[cidx]
                        end
                        push!(new_tconfigs, new_tconfig)
                    end

                    if haskey(foi_space, new_fock)
                        foi_space[new_fock] = unique((foi_space[new_fock]..., new_tconfigs...))
                    else
                        foi_space[new_fock] = new_tconfigs
                    end


                end
            end
        end
    end
    return foi_space
#=}}}=#
end





"""
    hylleraas_compressed_mp2(sig_in::CompressedTuckerState, ref::CompressedTuckerState,
            cluster_ops, clustered_ham;
            H0 = "Hcmf", tol=1e-6, nbody=4, max_iter=40, verbose=1, do_pt = true, thresh=1e-8)

- `H0`: ["H", "Hcmf"] 
"""
function hylleraas_compressed_mp2(sig_in::CompressedTuckerState, ref::CompressedTuckerState,
            cluster_ops, clustered_ham;
            H0 = "Hcmf", tol=1e-6, nbody=4, max_iter=100, verbose=1, do_pt = true, thresh=1e-8)
#={{{=#
    
#
            

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    
    # 
    # get <X|H|0>
    #sig = compress(sig_in, thresh=thresh)
    sig = deepcopy(sig_in)
    @printf(" Length of input      FOIS: %i\n", length(sig_in)) 
    @printf(" Length of compressed FOIS: %i\n", length(sig)) 
    #project_out!(sig, ref, thresh=thresh)
    @printf(" Build exact <X|V|0>\n")
    zero!(sig)
    @time build_sigma!(sig, ref, cluster_ops, clustered_ham)
    
    
    # (H0 - E0) |1> = X H |0>

    e2 = 0.0
   
    # 
    # get E_ref = <0|H|0>
    tmp = deepcopy(ref)
    zero!(tmp)
    build_sigma!(tmp, ref, cluster_ops, clustered_ham)
    e_ref = orth_dot(ref, tmp)
    @printf(" <0|H|0> 0 : %12.8f\n",e_ref)


    # 
    # get E0 = <0|H0|0>
    tmp = deepcopy(ref)
    zero!(tmp)
    build_sigma!(tmp, ref, cluster_ops, clustered_ham_0)
    e0 = orth_dot(ref,tmp)
    @printf(" <0|sig>  : %12.8f\n",nonorth_dot(ref,sig))
    @printf(" <0|H0|0>  : %12.8f\n",e0)


    @printf(" Length of FOIS      : %i\n", length(sig)) 
    
   
    @printf(" Project out reference\n")
    #sig = compress(sig, thresh=thresh)
    @printf(" <0|sig>  : %12.8f\n",nonorth_dot(ref,sig))
    @printf(" Length of FOIS      : %i\n", length(sig)) 
   
    b = -get_vector(sig)
    
    # 
    # get <X|F|0>
    tmp = deepcopy(sig)
    zero!(tmp)
    @time build_sigma!(tmp, ref, cluster_ops, clustered_ham_0)

    @printf(" Norm of <X|F|0> = %12.8f\n", sqrt(orth_dot(tmp,tmp)))
    b .+= get_vector(tmp)
    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(sig)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        for (tconfig, tuck) in tconfigs
            if haskey(ref, fock)
                if haskey(ref[fock], tconfig)
                    ref_tuck = ref[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = []
                    for i in 1:length(Sx.clusters)
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    Sx[fock][tconfig].core .= transform_basis(ref_tuck.core, overlaps)
                end
            end
        end
    end

    b .= b .+ get_vector(Sx).*(e_ref - e0)

    
    function mymatvec(x)

        xr = deepcopy(sig)
        xl = deepcopy(sig)
        set_vector!(xr,x)
        zero!(xl)
        build_sigma!(xl, xr, cluster_ops, clustered_ham_0, cache=true)

        # subtract off -E0|1>
        #
        scale!(xr,-e0)
        orth_add!(xl,xr)
        flush(stdout)

        return get_vector(xl)
    end

    dim = length(b)
    Axx = LinearMap(mymatvec, dim, dim)

    @printf(" Norm of b         : %18.12f\n", sum(b.*b))
    flush_cache(clustered_ham_0)
    
    @time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0)
    #@time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0, nbody=1)

    x_vector = zeros(dim)
    @time x, solver = cg!(x_vector, Axx, b, log=true, maxiter=max_iter, verbose=true, abstol=tol)
    
    flush_cache(clustered_ham_0)

    psi1 = deepcopy(sig)
    set_vector!(psi1,x_vector)
    
    SxC = orth_dot(Sx,psi1)
    @printf(" <A|X>C(X) = %12.8f\n", SxC)
   
    tmp = deepcopy(ref)
    zero!(tmp)
    build_sigma!(tmp,psi1, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(tmp,ref)
    @printf(" <1|1> = %12.8f\n", orth_dot(psi1,psi1))
    @printf(" <0|H|1> = %12.8f\n", ecorr)
    length(ecorr) == 1 || error(" Dimension Error", ecorr)
    ecorr = ecorr[1]

    @printf(" E(PT2)  = %12.8f\n", (e_ref + ecorr)/(1+SxC))

    return psi1, (ecorr+e_ref)/(1+SxC) 

end#=}}}=#




"""
    build_compressed_1st_order_state(cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham; thresh=1e-7, max_number=nothing, nbody=2) where {T,N}
Apply the Hamiltonian to `v` expanding into the uncompressed space.
This is done only partially, where each term is recompressed after being computed.
Lots of overhead probably from compression, but never completely uncompresses.

#Arguments
- `cts::CompressedTuckerState`: input state
- `cluster_ops`:
- `clustered_ham`: Hamiltonian
- `thresh`: Threshold for Tucker decomosition
- `max_number`: max number of tucker factors kept
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `v1::CompressedTuckerState`

"""
function build_compressed_1st_order_state(ket_cts::CompressedTuckerState{T,N}, cluster_ops, clustered_ham; 
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4, 
        do_pt=false) where {T,N}
    flush(stdout)
    println(" Compute the 1st order wavefunction for CompressedTuckerState. nbody = ", nbody)
#={{{=#
    #
    # Initialize data for our output sigma, which we will convert to a
    sig_cts = CompressedTuckerState(ket_cts.clusters, OrderedDict{FockConfig,OrderedDict{TuckerConfig,Tucker{T,N}} }(),  ket_cts.p_spaces, ket_cts.q_spaces)

    data = OrderedDict{FockConfig, OrderedDict{TuckerConfig, Vector{Tucker{T,N}} } }()

    for (ket_fock, ket_tconfigs) in ket_cts
        for (fock_trans, terms) in clustered_ham

            #
            # new fock sector configuration
            sig_fock = ket_fock + fock_trans

            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in ket_cts.clusters
                if sig_fock[ci.idx][1] > length(ci) || sig_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if sig_fock[ci.idx][1] < 0 || sig_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue

            for term in terms

                #
                # only proceed if current term acts on no more than our requested max number of clusters
                length(term.clusters) <= nbody || continue
                for (ket_tconfig, ket_tuck) in ket_tconfigs

                    #
                    # find the sig TuckerConfigs reached by applying current Hamiltonian term to ket_tconfig.
                    #
                    # For example:
                    #
                    #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                    #
                    #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                    #   have both non-zero P and Q spaces within the current fock sector, "sig_fock".
                    #
                    # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                    # available spaces, this list of which we will keep in "available".
                    #

                    available = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for ci in term.clusters
                        tmp = []
                        if haskey(ket_cts.p_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.p_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        if haskey(ket_cts.q_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.q_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        push!(available, tmp)
                    end


                    #
                    # Now loop over cartesian product of available subspaces (those in X above) and
                    # create the target TuckerConfig and then evaluate the associated terms
                    for prod in product(available...)
                        sig_tconfig = [ket_tconfig.config...]
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            sig_tconfig[ci.idx] = prod[cidx]
                        end
                        sig_tconfig = TuckerConfig(sig_tconfig)

                        #
                        # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                        # let's compute the matrix element block, then compress, then add it to any existing compressed
                        # coefficient tensor for that sig TuckerConfig.
                        #
                        # Both the Compression and addition takes a fair amount of work.


#                        if check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) == false
#       
#                            println()
#                            display(term.delta)
#                            display(sig_fock - ket_fock)
#                        end
                        check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue


                        sig_tuck = form_sigma_block_expand(term, cluster_ops,
                                                                sig_fock, sig_tconfig,
                                                                ket_fock, ket_tconfig, ket_tuck,
                                                                max_number=max_number,
                                                                prescreen=thresh)

                       
                        sig_tuck = compress(sig_tuck, thresh=thresh)

    
                        #sig_tuck = compress(sig_tuck, thresh=1e-16, max_number=max_number)

                        length(sig_tuck) > 0 || continue


                        if haskey(data, sig_fock)
                            if haskey(data[sig_fock], sig_tconfig)
                                #
                                # In this case, our sigma vector already has a compressed coefficient tensor.
                                # Consequently, we need to add these two together
                                
                                push!(data[sig_fock][sig_tconfig], sig_tuck)
                                #sig_tuck = add([sig_tuck, sig_cts[sig_fock][sig_tconfig]])
                                ##sig_tuck = compress(sig_tuck, thresh=thresh, max_number=max_number)
                                #sig_cts[sig_fock][sig_tconfig] = sig_tuck

                            else
                                data[sig_fock][sig_tconfig] = [sig_tuck]
                                #sig_cts[sig_fock][sig_tconfig] = sig_tuck
                            end
                        else
                            #sig_cts[sig_fock] = OrderedDict(sig_tconfig => sig_tuck)
                            data[sig_fock] = OrderedDict(sig_tconfig => [sig_tuck])
                        end

                    end

                end
            end
        end
    end

    for (fock,tconfigs) in data
        for (tconfig, tuck) in tconfigs
            if haskey(sig_cts, fock)
                sig_cts[fock][tconfig] = compress(nonorth_add(tuck), thresh=thresh)
            else
                sig_cts[fock] = OrderedDict(tconfig => nonorth_add(tuck))
            end
        end
    end

#    # 
#    # project out A space
#    for (fock,tconfigs) in sig_cts 
#        for (tconfig, tuck) in tconfigs
#            if haskey(ket_cts, fock)
#                if haskey(ket_cts[fock], tconfig)
#                    ket_tuck_A = ket_cts[fock][tconfig]
#
#                    ovlp = nonorth_dot(tuck, ket_tuck_A) / nonorth_dot(ket_tuck_A, ket_tuck_A)
#                    tmp = scale(ket_tuck_A, -1.0 * ovlp)
#                    #sig_cts[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
#                end
#            end
#        end
#    end
   
  
    # now combine Tuckers, project out reference space and multiply by resolvents
    #prune_empty_TuckerConfigs!(sig_cts)
    #return compress(sig_cts, thresh=thresh)
    return sig_cts
#=}}}=#
end
    
    
    
"""
    solve_for_compressed_space(ref_vector::CompressedTuckerState, cluster_ops, clustered_ham;
        max_iter    = 20,
        H0          = "cmf",
        nbody       = 4,
        thresh_foi  = 1e-6,
        thresh_var  = 1e-4,
        tol_ci      = 1e-5,
        tol_tucker  = 1e-6)

# Arguments
- `ref_vector`: initial state
- `nbody`: max number of nbody terms in the Hamiltonian used for creating FOIS
- `thresh_foi`: Compression threshold for the FOIS, or first order wavefunction
- `thresh_var`: Compression threshold for the variational solution
- `tol_ci`:     Convergence threshold for the CI (norm of residual)
- `tol_tucker`: Convergence threshold for Tucker iterations (energy change)
- `H0`: ["Hcmf", "H"]
"""
function solve_for_compressed_space(input_vec::CompressedTuckerState, cluster_ops, clustered_ham;
        max_iter    = 20,
        max_iter_pt = 200, # max number of iterations for solving for PT1
        nbody       = 4,
        H0          = "Hcmf",
        thresh_var  = 1e-4,
        thresh_foi  = 1e-6,
        thresh_pt   = 1e-5,
        tol_ci      = 1e-5,
        do_pt       = true,
        tol_tucker  = 1e-6)
      #={{{=#
    e_last = 0.0
    e_var  = 0.0
    e_pt2  = 0.0
    ref_vec = deepcopy(input_vec)
    clustered_S2 = extract_S2(input_vec.clusters)

    for iter in 1:max_iter
        println(" --------------------------------------------------------------------")
        println(" Iterate PT-Var:       Iteration #: ",iter)
        println(" --------------------------------------------------------------------")

        #
        # Compress Variational Wavefunction
        dim1 = length(ref_vec)
        norm1 = orth_dot(ref_vec, ref_vec)
        ref_vec = compress(ref_vec, thresh=thresh_var)
        normalize!(ref_vec)
        dim2 = length(ref_vec)
        norm2 = orth_dot(ref_vec, ref_vec)
        @printf(" Compressed Ref state from: %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_var)
        @printf(" Norm of compressed state: %12.8f \n", norm2)
        
        # 
        # Solve variationally in reference space
        println()
        @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
        @time e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, tol=tol_ci)
#       sig = deepcopy(ref_vec)
#       zero!(sig)
#       build_sigma!(sig, ref_vec, cluster_ops, clustered_ham)
#       e0 = orth_dot(ref_vec, sig)
        if iter == 1
            e_last = e0
        end
        
        tmp = deepcopy(ref_vec)
        zero!(tmp)
        build_sigma!(tmp, ref_vec, cluster_ops, clustered_S2)
        @printf(" <S^2> = %12.8f\n", orth_dot(tmp,ref_vec))
   
        #
        # Get First order wavefunction
        println()
        println(" Compute first order wavefunction. Reference space dim = ", length(ref_vec))
        @time pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)

        # 
        # Compress FOIS
        norm1 = orth_dot(pt1_vec, pt1_vec)
        dim1 = length(pt1_vec)
        pt1_vec = compress(pt1_vec, thresh=thresh_foi)
        norm2 = orth_dot(pt1_vec, pt1_vec)
        dim2 = length(pt1_vec)
        @printf(" FOIS Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
        @printf(" Norm of |1>:              %12.8f \n", norm2)
        @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))

        if do_pt
            #
            # 
            println()
            println(" Compute PT vector. Reference space dim = ", length(ref_vec))
            pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol_ci, do_pt=do_pt, max_iter=max_iter_pt, H0=H0)
            # 
            # Compress first order wavefunction 
            norm1 = orth_dot(pt1_vec, pt1_vec)
            dim1 = length(pt1_vec)
            pt1_vec = compress(pt1_vec, thresh=thresh_pt)
            norm2 = orth_dot(pt1_vec, pt1_vec)
            dim2 = length(pt1_vec)
            @printf(" PT   Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_pt)
            @printf(" Norm of |1>:              %12.8f \n", norm2)
            @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))
        end

        # 
        # Solve variationally in compressed FOIS 
        # CI
        println()
        var_vec = deepcopy(ref_vec)
        zero!(pt1_vec)
        nonorth_add!(var_vec, pt1_vec)
        normalize!(var_vec)
        @printf(" Solve in compressed FOIS. Dimension =   %10i\n", length(var_vec))
        @time e_var, var_vec = tucker_ci_solve(var_vec, cluster_ops, clustered_ham, tol=tol_ci)
        
        tmp = deepcopy(var_vec)
        zero!(tmp)
        build_sigma!(tmp, var_vec, cluster_ops, clustered_S2)
        @printf(" <S^2> = %12.8f\n", orth_dot(tmp,var_vec))

#        #
#        # Compress Variational Wavefunction
#        dim1 = length(var_vec)
#        norm1 = orth_dot(var_vec, var_vec)
#        var_vec = compress(var_vec, thresh=thresh_var)
#        normalize!(var_vec)
#        dim2 = length(var_vec)
#        norm2 = orth_dot(var_vec, var_vec)
#        @printf(" Compressed CI state from: %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_var)
#        @printf(" Norm of compressed state: %12.8f \n", norm2)

        ref_vec = var_vec

        @printf(" E(Ref)      = %12.8f\n", e0[1])
        do_pt == false || @printf(" E(PT2) tot  = %12.8f\n", e_pt2)
        @printf(" E(var) tot  = %12.8f\n", e_var[1])

        if abs(e_last[1] - e_var[1]) < tol_tucker 
            println("*Converged")
            return e_var, ref_vec
            break
        end
        e_last = e_var
            
    end
    println(" Not converged")
    return e_var,ref_vec 
end
    #=}}}=#
    


function do_fois_pt2(ref::CompressedTuckerState, cluster_ops, clustered_ham;
            H0          = "Hcmf",
            max_iter    = 50,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            verbose     = true)
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do Hylleraas PT2\n")
    @printf("   H0                      = %-s\n", H0)
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("\n")
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("\n-------------------------------------------------------\n")

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
    @time e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, tol=tol)

    #
    # Get First order wavefunction
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    @time pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)

    @printf(" Nick: %12.8f\n", sqrt(orth_dot(pt1_vec,pt1_vec)))
    project_out!(pt1_vec, ref)

    # 
    # Compress FOIS
    norm1 = sqrt(orth_dot(pt1_vec, pt1_vec))
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    norm2 = sqrt(orth_dot(pt1_vec, pt1_vec))
    dim2 = length(pt1_vec)
    @printf(" FOIS Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
    @printf(" Norm of |1>:              %12.8f → %12.8f (thresh = %8.1e)\n", norm1, norm2, thresh_foi)
    @printf(" Overlap between <1|0>:    %12.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))

    # 
    # Solve for first order wavefunction 
    println(" Compute PT vector. Reference space dim = ", length(ref_vec))
    pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, max_iter=max_iter, H0=H0)
    @printf(" E(Ref)      = %12.8f\n", e0[1])
    @printf(" E(PT2) tot  = %12.8f\n", e_pt2)
    return e_pt2, pt1_vec 
end
    


function do_fois_cepa(ref::CompressedTuckerState, cluster_ops, clustered_ham;
            max_iter    = 20,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            verbose     = true)
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do CEPA\n")
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("\n")
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("\n-------------------------------------------------------\n")

    # 
    # Solve variationally in reference space
    println()
    ref_vec = deepcopy(ref)
    @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
    @time e0, ref_vec = tucker_ci_solve(ref_vec, cluster_ops, clustered_ham, tol=tol)

    #
    # Get First order wavefunction
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    @time pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)

    project_out!(pt1_vec, ref)

#    #
#    # 
#    println()
#    println(" Compute PT vector. Reference space dim = ", length(ref_vec))
#    pt1_vec = hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, do_pt=do_pt)

    # 
    # Compress FOIS
    norm1 = orth_dot(pt1_vec, pt1_vec)
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    norm2 = orth_dot(pt1_vec, pt1_vec)
    dim2 = length(pt1_vec)
    @printf(" FOIS Compressed from:     %8i → %8i (thresh = %8.1e)\n", dim1, dim2, thresh_foi)
    @printf(" Norm of |1>:              %12.8f \n", norm2)
    @printf(" Overlap between <1|0>:    %8.1e\n", nonorth_dot(pt1_vec, ref_vec, verbose=0))

    # 
    # Solve CEPA 
    println()
    cepa_vec = deepcopy(pt1_vec)
    zero!(cepa_vec)
    println(" Do CEPA: Dim = ", length(cepa_vec))
    @time e_cepa, x_cepa = tucker_cepa_solve(ref_vec, cepa_vec, cluster_ops, clustered_ham, tol=tol, max_iter=max_iter, verbose=verbose)

    @printf(" E(cepa) corr =                 %12.8f\n", e_cepa)
    @printf(" X(cepa) norm =                 %12.8f\n", sqrt(orth_dot(x_cepa, x_cepa)))
    nonorth_add!(x_cepa, ref_vec)
    normalize!(x_cepa)
    return e_cepa, x_cepa
end


"""
    project_out!(v::CompressedTuckerState, w::CompressedTuckerState; thresh=1e-16)

Project w out of v 
|v'> = |v> - |w><w|v>
"""
function project_out!(v::CompressedTuckerState, w::CompressedTuckerState; thresh=1e-16)
    
    for (fock,tconfigs) in v 
        for (tconfig, tuck) in tconfigs
            if haskey(w, fock)
                if haskey(w[fock], tconfig)
                    w_tuck = w[fock][tconfig]

                    ovlp = nonorth_dot(tuck, w_tuck) / nonorth_dot(w_tuck, w_tuck)
                    tmp = scale(w_tuck, -1.0 * ovlp)
                    v[fock][tconfig] = nonorth_add(tuck, tmp, thresh=thresh)
                end
            end
        end
    end
end

