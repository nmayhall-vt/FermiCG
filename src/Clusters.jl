using Arpack 
using StaticArrays

"""
    idx::Int
    orb_list::Vector{Int}
"""
struct Cluster
    idx::Int
    orb_list::Vector{Int}
end
"""
	length(c::Cluster)

Return number of orbitals in `Cluster`
"""
function Base.length(c::Cluster)
    return length(c.orb_list)
end
"""
	dim_tot(c::Cluster)

Return dimension of hilbert space spanned by number of orbitals in `Cluster`. 
This is all sectors
"""
function dim_tot(c::Cluster)
    return 2^(2*length(c))
end
"""
	dim_tot(c::Cluster, na, nb)

Return dimension of hilbert space spanned by number of orbitals in `Cluster`
with `na` and `nb` number of alpha/beta electrons.
"""
function dim_tot(c::Cluster, na, nb)
    return binomial(length(c), na)*binomial(length(c), na) 
end
"""
	display(c::Cluster)
"""
function Base.display(c::Cluster)
    @printf("IDX%03i:DIM%04i:" ,c.idx,dim_tot(c))
    for si in c.orb_list
        @printf("%03i|", si)
    end
    @printf("\n")
end
function Base.isless(ci::Cluster, cj::Cluster)
    return Base.isless(ci.idx, cj.idx)
end
function Base.isequal(ci::Cluster, cj::Cluster)
    return Base.isequal(ci.idx, cj.idx)
end
######################################################################################################

"""
    cluster::Cluster                            # Cluster to which basis belongs
    basis::Dict{Tuple,Matrix{Float64}}          # Basis vectors (nα, nβ)=>[I,s]
These basis coefficients map local slater determinants to local vectors
`(nα, nβ): 
V[αstring*βstring, cluster_state]`
"""
struct ClusterBasis
    cluster::Cluster
    basis::Dict{Tuple,Matrix{Float64}}
end
ClusterBasis(ci::Cluster) = ClusterBasis(ci, Dict{Tuple,Matrix{Float64}}())

Base.iterate(cb::ClusterBasis, state=1) = iterate(cb.basis, state)
Base.length(cb::ClusterBasis) = length(cb.basis)
Base.getindex(cb::ClusterBasis,i) = cb.basis[i] 
Base.setindex!(cb::ClusterBasis,val,key) = cb.basis[key] = val
Base.haskey(cb::ClusterBasis,key) = haskey(cb.basis, key)
function Base.display(cb::ClusterBasis) 
    @printf(" ClusterBasis for Cluster: %4i\n",cb.cluster.idx)
    norb = length(cb.cluster)
    for (sector, vecs) in cb.basis
        dim = size(vecs,2)
        total_dim = binomial(norb,sector[1]) * binomial(norb,sector[2]) 
        
        @printf("   FockSector = (%2iα, %2iβ): Total Dim = %5i: Dim = %4i\n", sector[1],sector[2],total_dim, dim)
    end
end


######################################################################################################
struct ClusterOps
    cluster::Cluster
    data::Dict{String,Dict{Tuple,Array}}
end
Base.iterate(i::ClusterOps, state=1) = iterate(i.data, state)
Base.length(i::ClusterOps) = length(co.data)
Base.getindex(co::ClusterOps,i) = co.data[i] 
Base.setindex!(co::ClusterOps,val,key) = co.data[key] = val
Base.haskey(co::ClusterOps,key) = haskey(co.data, key)
Base.keys(co::ClusterOps) = keys(co.data)
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
function ClusterOps(ci::Cluster)
    dic1 = Dict{Tuple,Array{Float64}}()
    dic2 = Dict{String,typeof(dic1)}() 
    return ClusterOps(ci, dic2)
end

"""
    adjoint(co::ClusterOps; verbose=0)

Take ClusterOps, `co`, and return a new ClusterOps'
"""

"""
Defines a single cluster's subspace for Tucker. Each focksector is allowed to have a distinct cluster state range 
for the subspace.

    cluster::Cluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
"""
struct ClusterSubspace
    cluster::Cluster
    data::OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}
end
function ClusterSubspace(cluster::Cluster)
    return ClusterSubspace(cluster,OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}())
end
Base.haskey(css::ClusterSubspace, i) = return haskey(css.data,i)
Base.setindex!(tss::ClusterSubspace, i, j) = tss.data[j] = i
Base.getindex(tss::ClusterSubspace, i) = return tss.data[i] 
function Base.display(tss::ClusterSubspace)
    @printf(" Subspace for Cluster: %4i : ", tss.cluster.idx)
    display(tss.cluster)
    for (fock,range) in tss.data
        @printf("  %10s   Range: %4i → %-4i Dim %4i\n",Int.(fock), first(range), last(range), length(range))
    end
end

"""
    get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)

For a given `ClusterSubspace`, `tss`, return the subspace remaining
"""
function get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)
#={{{=#
    data = OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}()
    for (fock,basis) in cb
    
        if haskey(tss.data,fock)
            first(tss.data[fock]) == 1 || error(" p-space doesn't include ground state?")
            newrange = last(tss[fock])+1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        else
            newrange = 1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        end
    end

    return ClusterSubspace(tss.cluster, data)
#=}}}=#
end




function Base.:+(a::Tuple{T,T}, b::Tuple{T,T}) where T<:Integer
    return (a[1]+b[1], a[2]+b[2])
end
function Base.:-(a::Tuple{T,T}, b::Tuple{T,T}) where T<:Integer
    return (a[1]-b[1], a[2]-b[2])
end


"""
    compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{Cluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing)

Return a Vector of `ClusterBasis` for each `Cluster` 
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{Cluster}`: Clusters 
- `verbose::Int`: Print level
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `delta_elec`: number of electrons different from reference (init_fspace)
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1a`: background density matrix for embedding local hamiltonian (alpha)
- `rdm1b`: background density matrix for embedding local hamiltonian (beta)
"""
function compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{Cluster}; 
                init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
                rdm1a=nothing, rdm1b=nothing)
#={{{=#
    # initialize output
    cluster_bases = Vector{ClusterBasis}()

    for ci in clusters
        verbose == 0 || display(ci)

        #
        # Get subset of integrals living on cluster, ci
        ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 

        if (rdm1a != nothing && init_fspace == nothing)
            error(" Cant embed withing init_fspace")
        end

        if all( (rdm1a,rdm1b,init_fspace) .!= nothing)
            # 
            # Verify that density matrix provided is consistent with reference fock sectors
            occs = diag(rdm1a)
            occs[ci.orb_list] .= 0
            na_embed = sum(occs)
            occs = diag(rdm1b)
            occs[ci.orb_list] .= 0
            nb_embed = sum(occs)
            verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f", na_embed, nb_embed)
        end
            
        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec)
        end
        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = FermiCG.possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci) 
        for sec in sectors
            
            #
            # prepare for FCI calculation for give sector of Fock space
            problem = FermiCG.StringCI.FCIProblem(length(ci), sec[1], sec[2])
            verbose == 0 || display(problem)
            nr = min(max_roots, problem.dim)

            #
            # Build full Hamiltonian matrix in cluster's Slater Det basis
            Hmat = FermiCG.StringCI.build_H_matrix(ints_i, problem)
            if problem.dim < 1000
                F = eigen(Hmat)

                e = F.values[1:nr]
                basis_i[sec] = F.vectors[:,1:nr]
                #display(e)
            else
                e,v = Arpack.eigs(Hmat, nev = nr, which=:SR)
                e = real(e)[1:nr]
                basis_i[sec] = v[:,1:nr]
            end
            if verbose > 0
                state=1
                for ei in e
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#


"""
    compute_cluster_ops(cluster_bases::Vector{ClusterBasis})
"""
function compute_cluster_ops(cluster_bases::Vector{ClusterBasis},ints)

    clusters = Vector{Cluster}()
    for ci in cluster_bases
        push!(clusters, ci.cluster)
    end
    
    cluster_ops = Vector{ClusterOps}()
    for ci in clusters
        push!(cluster_ops, ClusterOps(ci)) 
    end


    for ci in clusters
        cb = cluster_bases[ci.idx]
        
        cluster_ops[ci.idx]["H"] = FermiCG.tdm_H(cb, subset(ints, ci.orb_list)) 

        cluster_ops[ci.idx]["A"], cluster_ops[ci.idx]["a"] = FermiCG.tdm_A(cb,"alpha") 
        cluster_ops[ci.idx]["B"], cluster_ops[ci.idx]["b"] = FermiCG.tdm_A(cb,"beta")
        cluster_ops[ci.idx]["AA"], cluster_ops[ci.idx]["aa"] = FermiCG.tdm_AA(cb,"alpha") 
        cluster_ops[ci.idx]["BB"], cluster_ops[ci.idx]["bb"] = FermiCG.tdm_AA(cb,"beta") 
        cluster_ops[ci.idx]["Aa"] = FermiCG.tdm_Aa(cb,"alpha") 
        cluster_ops[ci.idx]["Bb"] = FermiCG.tdm_Aa(cb,"beta") 
        cluster_ops[ci.idx]["Ab"], cluster_ops[ci.idx]["Ba"] = FermiCG.tdm_Ab(cb) 
        # remove BA and ba account for these terms 
        cluster_ops[ci.idx]["AB"], cluster_ops[ci.idx]["ba"], cluster_ops[ci.idx]["BA"], cluster_ops[ci.idx]["ab"] = FermiCG.tdm_AB(cb)
        cluster_ops[ci.idx]["AAa"], cluster_ops[ci.idx]["Aaa"] = FermiCG.tdm_AAa(cb,"alpha")
        cluster_ops[ci.idx]["BBb"], cluster_ops[ci.idx]["Bbb"] = FermiCG.tdm_AAa(cb,"beta")
        cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"] = FermiCG.tdm_ABa(cb,"alpha")
        cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"] = FermiCG.tdm_ABa(cb,"beta")
        #cluster_ops[ci.idx]["ABa"], cluster_ops[ci.idx]["Aba"], cluster_ops[ci.idx]["BAa"], cluster_ops[ci.idx]["Aab"] = FermiCG.tdm_ABa(cb,"alpha")
        #cluster_ops[ci.idx]["ABb"], cluster_ops[ci.idx]["Bba"], cluster_ops[ci.idx]["BAb"], cluster_ops[ci.idx]["Bab"] = FermiCG.tdm_ABa(cb,"beta")

        to_delete = [
                     #"AAa",
                     #"Aaa",
                     #"BBb",
                     #"Bbb",
                     #
                     #"ABa",
                     #"Aba",
                     ##"BAa",
                     ##"Aab",
                     #
                     #"ABb",
                     #"Bba",
                     ##"BAb",
                     ##"Bab",
                     #"Aa",
                     #"Bb",
                     #"Ab",
                     #"Ba",
                     #"AB",
                     #"ba",
                     #"BA",
                     #"ab",
                     #"AA",
                     #"BB",
                     #"aa",
                     #"bb"
                     ]
        for op in to_delete
            for (ftran,array) in cluster_ops[ci.idx][op]
                cluster_ops[ci.idx][op][ftran] .*= 0
            end
        end

        # Compute single excitation operator
        tmp = Dict{Tuple,Array}()
        for (fock,basis) in cb
            tmp[(fock,fock)] = (cluster_ops[ci.idx]["Aa"][(fock,fock)] + cluster_ops[ci.idx]["Bb"][(fock,fock)])
        end
        cluster_ops[ci.idx]["E1"] = tmp 

        #
        # reshape data into 3index quantities: e.g., (pqr, I, J)
        for opstring in keys(cluster_ops[ci.idx])
            opstring != "H" || continue
            for ftrans in keys(cluster_ops[ci.idx][opstring])
                data = cluster_ops[ci.idx][opstring][ftrans]
                dim1 = prod(size(data)[1:(length(size(data))-2)])
                dim2 = size(data)[length(size(data))-1]
                dim3 = size(data)[length(size(data))-0]
                cluster_ops[ci.idx][opstring][ftrans] = copy(reshape(data, (dim1,dim2,dim3)))
            end
        end
    end
    return cluster_ops
end
    

"""
    possible_focksectors(c::Cluster, delta_elec=nothing)
        
Get list of possible fock spaces accessible to the cluster

- `delta_elec::Vector{Int}`: (nα, nβ, Δ) restricts fock spaces to: (nα,nβ) ± Δ electron transitions
"""
function possible_focksectors(c::Cluster; delta_elec::Tuple=())
    ref_a = nothing
    ref_b = nothing
    delta = nothing
    if length(delta_elec) != 0
        length(delta_elec) == 3 || throw(DimensionMismatch)
        ref_a = delta_elec[1]
        ref_b = delta_elec[2]
        delta = delta_elec[3]
    end

    no = length(c)
   
    fsectors::Vector{Tuple} = []
    for na in 0:no
        for nb in 0:no 
            if length(delta_elec) != 0
                if abs(na-ref_a)+abs(nb-ref_b) > delta
                    continue
                end
            end
            push!(fsectors,(na,nb))
        end
    end
    return fsectors
end





"""
    tdm_H(cb::ClusterBasis; verbose=0)

Compute local Hamiltonian `<s|H|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_H(cb::ClusterBasis, ints; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    verbose == 0 || display(cb.cluster)
    for (fock,basis) in cb
        focktrans = (fock,fock)
        problem = StringCI.FCIProblem(norbs, fock[1], fock[2])
        verbose == 0 || display(problem)
        Hmap = StringCI.get_map(ints, problem)

        dicti[focktrans] = cb[fock]' * Matrix((Hmap * cb[fock]))

        if verbose > 0
            for e in 1:size(cb[fock],2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
            end
        end
    end
    return dicti
#=}}}=#
end


"""
    tdm_A(cb::ClusterBasis; verbose=0)

Compute `<s|p'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_A(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+1,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+1)
            else
                throw(DomainError(spin_case))
            end
            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_annihilation(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [1,3,2])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AA(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AA(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+2,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+2)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                #println()
                #println("::::::::::::::: ", fockbra, fockket)
                #println(":: ", size(basis_bra), size(basis_ket))
                dicti[focktrans] = FermiCG.StringCI.compute_AA(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [2,1,4,3])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = (na,nb)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_Aa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                #dicti[focktrans] = reshape(dicti[focktrans],(norbs*norbs, size(dicti[focktrans],3), size(dicti[focktrans],4)))
            end
        end
    end
    return dicti
#=}}}=#
end


"""
    tdm_Ab(cb::ClusterBasis; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Ab(cb::ClusterBasis; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -1:norbs+1
        for nb in -1:norbs+1
            fockbra = (na+1,nb-1)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_Ab(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                #dicti[focktrans] = FermiCG.StringCI.compute_Ba(norbs, fockket[1], fockket[2], fockket[1], fockket[2], basis_bra, basis_ket)
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [2,1,4,3])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AB(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q'` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AB(cb::ClusterBasis; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    dictj = Dict{Tuple,Array}()
    dictj_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -2:norbs+2
        for nb in -2:norbs+2
            fockbra = (na+1,nb+1)

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_AB(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                dictj[focktrans] = -permutedims(dicti[focktrans], [2,1,3,4])
                
                # adjoint 
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [2,1,4,3])
                dictj_adj[focktrans_adj] =  permutedims(dictj[focktrans], [2,1,4,3])
            end
        end
    end
    return dicti, dicti_adj, dictj, dictj_adj
#=}}}=#
end


"""
    tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in 0:norbs
        for nb in 0:norbs
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na+1,nb)
            elseif spin_case == "beta"
                fockbra = (na,nb+1)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                dicti[focktrans] = FermiCG.StringCI.compute_AAa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket, spin_case)
                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]
                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [3,2,1,5,4])
            end
        end
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    dictj = Dict{Tuple,Array}()
    dictj_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    for na in -2:norbs+2
        for nb in -2:norbs+2
            fockbra = ()
            if spin_case == "alpha"
                fockbra = (na,nb+1)
            elseif spin_case == "beta"
                fockbra = (na+1,nb)
            else
                throw(DomainError(spin_case))
            end

            fockket = (na,nb)
            focktrans = (fockbra,fockket)
            focktrans_adj = (fockket,fockbra)

            if haskey(cb, fockbra) && haskey(cb, fockket)
                basis_bra = cb[fockbra]
                basis_ket = cb[fockket]
                if spin_case == "alpha"
                    dicti[focktrans]     = FermiCG.StringCI.compute_ABa(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                    dictj[focktrans] = -permutedims(dicti[focktrans], [2,1,3,4,5])
                elseif spin_case == "beta"
                    dicti[focktrans]     = FermiCG.StringCI.compute_ABb(norbs, fockbra[1], fockbra[2], fockket[1], fockket[2], basis_bra, basis_ket)
                    dictj[focktrans] = -permutedims(dicti[focktrans], [2,1,3,4,5])
                else
                    error("Wrong spin_case: ",spin_case)
                end

                # adjoint 
                basis_bra = cb[fockket]
                basis_ket = cb[fockbra]

                dicti_adj[focktrans_adj] =  permutedims(dicti[focktrans], [3,2,1,5,4])
                dictj_adj[focktrans_adj] =  permutedims(dictj[focktrans], [3,2,1,5,4])
            end
        end
    end
    return dicti, dicti_adj, dictj, dictj_adj
    #=}}}=#
end



function add_cmf_operators!(ops::Vector{ClusterOps}, bases::Vector{ClusterBasis}, ints, Da, Db; verbose=1)
    
    n_clusters = length(bases)
    for ci_idx in 1:n_clusters
        cb = bases[ci_idx]
        ci = cb.cluster
        verbose == 0 || println()
        verbose == 0 || display(ci)
        norbs = length(cb.cluster)
        
        ints_i = form_casci_ints(ints, ci, Da, Db)


        dicti = Dict{Tuple,Array}()
        
        #
        # loop over fock-space transitions
        verbose == 0 || display(cb.cluster)
        for (fock,basis) in cb
            focktrans = (fock,fock)
            problem = StringCI.FCIProblem(norbs, fock[1], fock[2])
            verbose == 0 || display(problem)
            Hmap = StringCI.get_map(ints_i, problem)

            dicti[focktrans] = cb[fock]' * Matrix((Hmap * cb[fock]))

            if verbose > 0
                for e in 1:size(cb[fock],2)
                    @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
                end
            end
        end
        ops[ci.idx]["Hcmf"] = dicti
    end
    return 
end
