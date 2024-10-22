using ActiveSpaceSolvers
using BlockDavidson




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



"""
    compute_cluster_ops(cluster_bases::Vector{ClusterBasis})
"""
function compute_cluster_ops(cluster_bases, ints::InCoreInts{T}) where {T}
#={{{=#
    clusters = Vector{MOCluster}()
    for ci in cluster_bases
        push!(clusters, ci.cluster)
    end
    
    cluster_ops = Vector{ClusterOps{T}}()
    for ci in clusters
        push!(cluster_ops, ClusterOps(ci, T=T)) 
    end


    for ci in clusters

        display(ci)
        flush(stdout)

        cb = cluster_bases[ci.idx]
       
        cluster_ops[ci.idx]["H"] = FermiCG.tdm_H(cb, subset(ints, ci.orb_list), verbose=0) 
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
       
        # spin operators
        
        #
        # S+
        op = Dict{Tuple,Array}()
        for (fock,mat) in cluster_ops[ci.idx]["Ab"]
            dims = size(mat)
            op[fock] = zeros(dims[2:4]...)
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p,i,j] = mat[p,p,i,j]
                    end
                end
            end
        end
        cluster_ops[ci.idx]["S+"] = op
        
        #
        # S-
        op = Dict{Tuple,Array}()
        for (fock,mat) in cluster_ops[ci.idx]["Ba"]
            dims = size(mat)
            op[fock] = zeros(dims[2:4]...)
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p,i,j] = mat[p,p,i,j]
                    end
                end
            end
        end
        cluster_ops[ci.idx]["S-"] = op
        
        #
        # Sz
        op = Dict{Tuple,Array}()
        #
        # loop over fock-space transitions
        for (fock,basis) in cb
            focktrans = (fock,fock)

            sz = (fock[1] - fock[2]) / 2.0
            op[focktrans] = sz*Matrix(1.0I, size(cb[fock],2), size(cb[fock],2))
            op[focktrans] = reshape(op[focktrans],1,size(op[focktrans],1),size(op[focktrans],2))

        end
        cluster_ops[ci.idx]["Sz"] = op


        #
        # S2
        cluster_ops[ci.idx]["S2"] = FermiCG.tdm_S2(cb, subset(ints, ci.orb_list), verbose=0) 


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
            opstring != "S2" || continue
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
    #=}}}=#


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
        verbose == 0 || display(basis.ansatz)
        Hmap = LinearMap(ints, basis.ansatz)

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
"""
function tdm_S2(cb::ClusterBasis, ints; verbose=0)
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
        verbose == 0 || display(basis.ansatz)

        dicti[focktrans] = cb[fock]' * apply_S2_matrix(basis.ansatz, cb[fock].vectors)

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
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_c_a(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_c_b(basis_bra, basis_ket)
                end
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
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_cc_aa(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_cc_bb(basis_bra, basis_ket)
                end
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
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_ca_aa(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_ca_bb(basis_bra, basis_ket)
                end
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
                dicti[focktrans] = compute_operator_ca_ab(basis_bra, basis_ket)
                
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
                dicti[focktrans] = compute_operator_cc_ab(basis_bra, basis_ket)
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
                if spin_case == "alpha"
                    dicti[focktrans] = compute_operator_cca_aaa(basis_bra, basis_ket)
                else
                    dicti[focktrans] = compute_operator_cca_bbb(basis_bra, basis_ket)
                end
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
                    dicti[focktrans]     = compute_operator_cca_aba(basis_bra, basis_ket)
                    dictj[focktrans] = -permutedims(dicti[focktrans], [2,1,3,4,5])
                elseif spin_case == "beta"
                    dicti[focktrans]     = compute_operator_cca_abb(basis_bra, basis_ket)
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



"""
    function add_cmf_operators!(ops::Vector{ClusterOps}, bases::Vector{ClusterBasis}, ints, Da, Db; verbose=0)

Add effective local hamiltonians (local CASCI) type hamiltonians to a `ClusterOps` type for each `Cluster'
"""
function add_cmf_operators!(ops, bases, ints, Da, Db; verbose=0)
    #={{{=#
    n_clusters = length(bases)
    for ci_idx in 1:n_clusters
        cb = bases[ci_idx]
        ci = cb.cluster
        verbose == 0 || println()
        verbose == 0 || display(ci)
        norbs = length(cb.cluster)
        
        ints_i = subset(ints, ci.orb_list, Da, Db)
        #ints_i = form_casci_ints(ints, ci, Da, Db)


        dicti = Dict{Tuple,Array}()
        
        #
        # loop over fock-space transitions
        verbose == 0 || display(cb.cluster)
        for (fock,basis) in cb
            focktrans = (fock,fock)
            verbose == 0 || display(basis.ansatz)
            Hmap = LinearMap(ints_i, basis.ansatz)

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
#=}}}=#


"""
	form_schmidt_basis
thresh_orb      :   threshold for determining how many bath orbitals to include
thresh_schmidt  :   threshold for determining how many singular vectors to include for cluster basis

Returns new basis for the cluster
"""
function form_schmidt_basis(ints::InCoreInts, ci::MOCluster, Da, Db; 
        thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,do_embedding=true,
        eig_nr=1, eig_max_cycles=200,
        A::Type=FCIAnsatz)

    println()
    println("------------------------------------------------------------")
    @printf("Form Embedded Schmidt-style basis for Cluster %4i\n",ci.idx)
    D = Da + Db 

    # Form the exchange matrix
    K = zeros(size(ints.h1))
    @tensor begin
	K[q,r]  = ints.h2[p,q,r,s] * D[p,s]
    end

    no = size(ints.h1,1)
    ci_no = length(ci.orb_list)


    na_tot = Int(round(tr(Da)))
    nb_tot = Int(round(tr(Db)))
    println(" Number of electrons in full system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_tot,nb_tot)

    active = ci.orb_list

    backgr = Vector{Int}()
    for i in 1:no
    	if !(i in active)
    	    append!(backgr,i)
        end
    end

    println("active",active)
    println("backgr",backgr)

    K2 = zeros((ci_no,no-ci_no))

    for (pi,p) in enumerate(active)
    	for (qi,q) in enumerate(backgr)
	    K2[pi,qi] = K[p,q]
	end
    end

    println("The exchange matrix:")
    display(K2)
    F = svd(K2,full=true)

    @printf("\nSing. Val.\n")
    nkeep = 0
    for si in F.S
    	@printf("%16.12f\n",si)
        if si > thresh_orb
            nkeep += 1
        end
    end

    C = zeros(size(ints.h1))
    for (pi,p) in enumerate(active)
    	for (qi,q) in enumerate(active)
	    if pi==qi
	        C[p,qi] = 1
	    end
	end
    end

    #display(C)

    #display(F.Vt)
    #display(F.U)
    for (pi,p) in enumerate(backgr)
    	for (qi,q) in enumerate(backgr)
	    C[p,qi+length(active)] = F.Vt[qi,pi]
	end
    end
    #display(C)

    Cfrag = C[:,1:ci_no]
    Cbath = C[:,ci_no+1:ci_no+nkeep]
    Cenvt = C[:,ci_no+nkeep+1:end]
    
    @printf("Cfrag\n")
    display(Cfrag)
    @printf("\n NElec: %12.8f\n",(tr(Cfrag'*(Da+Db)*Cfrag)))
    @printf("Cbath\n")
    display(Cbath)
    @printf("\n NElec: %12.8f\n",(tr(Cbath'*(Da+Db)*Cbath)))
    @printf("Cenv\n")
    display(Cenvt)
    @printf("\n NElec: %12.8f\n",(tr(Cenvt'*(Da+Db)*Cenvt)))

    K2 = C'* K * C
    Da2 = C'* Da * C
    Db2 = C'* Db * C

    na = tr(Da2[1:ci_no+nkeep,1:ci_no+nkeep])
    nb = tr(Db2[1:ci_no+nkeep,1:ci_no+nkeep])

    println(" Number of electrons in Fragment+Bath system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na,nb)

    denvt_a = Cenvt*Cenvt'*Da*Cenvt*Cenvt'
    denvt_b = Cenvt*Cenvt'*Db*Cenvt*Cenvt'
    #println(denvt_a)

    na_env = tr(denvt_a)
    nb_env = tr(denvt_b)

    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_env,nb_env)

    na_envt = Int(round(tr(Cenvt'*Da*Cenvt)))
    nb_envt = Int(round(tr(Cenvt'*Db*Cenvt)))


    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_envt,nb_envt)
    #display(Da)

    # rotate integrals to current subspace basis
    denvt_a = C'*denvt_a*C
    denvt_b = C'*denvt_b*C
    ints2 = orbital_rotation(ints, C)

    #avoid very zero numbers in diagonalization
    denvt_a[abs.(denvt_a) .< 1e-15] .= 0
    denvt_b[abs.(denvt_b) .< 1e-15] .= 0
    #display(denvt_a)
    #display(denvt_a)

    # find closest idempotent density for the environment
    if do_embedding
        if size(Cenvt,2)>0

	    #eigenvalue 
	    EIG = eigen(denvt_a)
	    U = EIG.vectors
	    n = EIG.values

	    U = U[:, sortperm(n,rev=true)]
	    n = n[sortperm(n,rev=true)]
	    #println(n)
	    #display(U)

            for i in 1:nkeep
                @assert(n[i]>1e-14)
	    end

            denvt_a = U[:,1:na_envt] * U[:,1:na_envt]'

	    EIG = eigen(denvt_b)
	    U = EIG.vectors
	    n = EIG.values

	    U = U[:, sortperm(n,rev=true)]
	    n = n[sortperm(n,rev=true)]

            for i in 1:nkeep
                @assert(n[i]>1e-14)
	    end

            denvt_b = U[:,1:nb_envt] * U[:,1:nb_envt]'

	end
    #form ints in the cluster 
    no_range = collect(1:size(Cfrag,2)+size(Cbath,2))

    #ints_f = subset(ints2,collect(1:size(Cfrag,2)+size(Cbath,2)), denvt_a, denvt_b)

    ints_f = subset(ints2,no_range,denvt_a,denvt_b)
    #ints_f = FermiCG.form_1rdm_dressed_ints(ints2,no_range,denvt_a,denvt_b)

    else
        denvt_a *= 0 
        denvt_b *= 0 
        ints_f = form_casci_eff_ints(ints2,collect(1:size(Cfrag,2)+size(Cbath,2)), denvt_a, denvt_b)
    end

    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",tr(denvt_a),tr(denvt_b))

    na_actv = na_tot - na_envt
    nb_actv = nb_tot - nb_envt
    println(" Number of electrons in Fragment+Bath system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_actv,nb_actv)

    norb2 = size(ints_f.h1,1)

    ansatz = FCIAnsatz(norb2, na_actv, nb_actv)
    Hmap = LinearMap(ints_f, ansatz)
    v0 = svd(rand(ansatz.dim,eig_nr)).U
    davidson = FermiCG.Davidson(Hmap,v0=v0,max_iter=200, max_ss_vecs=20, nroots=eig_nr, tol=1e-8)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    @time e,v = BlockDavidson.eigs(davidson);

    solution = Solution(ansatz, e, v)
    ansatz = FCIAnsatz(norb2, na_actv, nb_actv)
    
    #solution = solve(ints_f, ansatz, SolverSettings(maxiter=200, nroots=eig_nr, tol=1e-8))
    #solution = solve(ints_f, ansatz, SolverSettings(maxiter=200, nroots=eig_nr, tol=1e-8))
    
    basis = svd_state(solution, length(active), nkeep, thresh_schmidt)

    return basis
end


"""
    compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                       clusters::Vector{MOCluster}, 
                                       rdm1::RDM1{T},
                                       delta_elec::Vector,
                                       ref_fock::FockConfig; 
                                       verbose=0, 
                                       max_roots=10, 
                                       A::Type=FCIAnsatz) where T

Return a Vector of `ClusterBasis` for each `Cluster`.
For each number of electrons specified by ref_fock +- 1->delta_elec (for each cluster), 
we solve the CASCI problem, collecting `max_roots` of the lowest energy eigenvectors for the half-filled (or of odd number nalpha = nbeta+1) level. Then we apply S^- and S^+ to generate the higher/lower m_s blocks directly. 

# Arguments
#
- `ints`: InCoreInts integrals
- `clusters`: Clusters 
- `verbose`: Print level
- `ref_fock`:  reference space for defining target focksectors with `delta_elec`
- `delta_elec`: number of electrons different from reference (init_fspace) for each cluster
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1`: background density matrix for embedding local hamiltonian 
- `A`: the type of Ansatz object used to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                            clusters::Vector{MOCluster}, 
                                            rdm1::RDM1{T},
                                            delta_elec::Vector,
                                            ref_fock::FockConfig; 
                                            verbose=0, 
                                            max_roots=10, 
                                            A::Type=FCIAnsatz) where T
    #={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{A,T}}()

    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)") 
    for ci in clusters
        verbose == 0 || display(ci)
        

        ints_i = subset(ints, ci, rdm1) 


        # 
        # Verify that density matrix provided is consistent with reference fock sectors
        occs = diag(rdm1.a)
        occs[ci.orb_list] .= 0
        na_embed = sum(occs)
        occs = diag(rdm1.b)
        occs[ci.orb_list] .= 0
        nb_embed = sum(occs)
        verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f\n", na_embed, nb_embed)


        delta_e_i = delta_elec[ci.idx] 

        #
        # Get list of Fock-space sectors for current cluster
        #
        ni = ref_fock[ci.idx][1] + ref_fock[ci.idx][2]  # number of electrons in ci
        sectors = []
        max_e = 2*length(ci)
        min_e = 0
        for nj in ni-delta_e_i:ni+delta_e_i
        
            nj <= max_e || continue
            nj >= min_e || continue

            naj = nj÷2 + nj%2
            nbj = nj÷2
            push!(sectors, (naj, nbj))
        end

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T) 
        for sec in sectors

            #
            # prepare for FCI calculation for give sector of Fock space
            ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            verbose == 0 || @printf(" Preparing to compute : \n")
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)

            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr 
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                #Hmat = Matrix(1.0I, nr, nr)
                F = eigen(Hmat)

                basis_i[sec] = Solution(ansatz, F.values[1:nr], F.vectors[:,1:nr])

                #display(e)
            else
                #
                # Do sparse build 
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr,package="arpack"))
            end

            #
            # Loop over spin-flips
            # 
            # s2 = s(s+1) 
            

            s2 = compute_s2(basis_i[sec])    

            nr = length(basis_i[sec].energies)
            #for r in 1:nr
            #    S = (-1 + sqrt(1+4*s2[r]))/2
            #    gr = 2*S+1 # Degeneracy
            #end
          
            #
            #   S-
            #
            # find how many applications of S- we need to try
           
            verbose == 0 || println(" Compute higher and lower Ms components")
            n_sm = minimum((sec[1], ansatz.no-sec[2]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for smi in 1:n_sm
                vi, ansatzi = apply_sminus(vi, ansatzi)

                verbose == 0 || display(ansatzi) 
                flush(stdout)

                if size(vi,2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi*vi))
                #ei = compute_energy(vi, ansatzi)
            
                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end
            #
            #   S+
            #
            # find how many applications of S+ we need to try
            
            n_sp = minimum((sec[2], ansatz.no-sec[1]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for spi in 1:n_sp
                vi, ansatzi = apply_splus(vi, ansatzi)
                
                verbose == 0 || display(ansatzi) 
                flush(stdout)

                if size(vi,2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi*vi))
                #ei = compute_energy(vi, ansatzi)
            
                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end

        end
           
        flush(stdout)
        if verbose > 0
            println()
            for (sec, sol) in basis_i    
                println()
                display(sol.ansatz)
                s2 = compute_s2(sol)    
                for i in 1:length(sol.energies)
                    @printf("   State %4i Energy: %12.8f S2: %12.8f\n",i, sol.energies[i], s2[i])
                end
                flush(stdout)
            end
        end

        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#

"""
    compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                       clusters::Vector{MOCluster}, 
                                       rdm1::RDM1{T},
                                       delta_elec::Vector,
                                       ref_fock::FockConfig,
                                       ansatze::Vector{<:Ansatz};
                                       verbose=0, 
                                       max_roots=10) where T

Return a Vector of `ClusterBasis` for each `Cluster`.
For each number of electrons specified by ref_fock +- 1->delta_elec (for each cluster), 
we solve the CASCI problem, collecting `max_roots` of the lowest energy eigenvectors for the half-filled (or of odd number nalpha = nbeta+1) level. Then we apply S^- and S^+ to generate the higher/lower m_s blocks directly. 

# Arguments
#
- `ints`: InCoreInts integrals
- `clusters`: Clusters 
- `verbose`: Print level
- `ref_fock`:  reference space for defining target focksectors with `delta_elec`
- `delta_elec`: number of electrons different from reference (init_fspace) for each cluster
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1`: background density matrix for embedding local hamiltonian 
- `A`: the type of Ansatz object used to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                            clusters::Vector{MOCluster}, 
                                            rdm1::RDM1{T},
                                            delta_elec::Vector,
                                            ref_fock::FockConfig,
                                            ansatze::Vector{<:Ansatz};
                                            verbose=0, 
                                            max_roots=10) where T
    #={{{=#
    # initialize output
    #
    #
    cluster_bases = Vector{ClusterBasis{<:Ansatz,T}}()
    fock_ansatze = ActiveSpaceSolvers.generate_cluster_fock_ansatze(ref_fock, clusters, ansatze, delta_elec)
    #for i in fock_ansatze
    #    display(i)
    #end

    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)") 

    for i in 1:length(clusters)
        ci = clusters[i]
        #verbose == 0 || display(ci)
        ints_i = subset(ints, ci, rdm1) 

        # 
        # Verify that density matrix provided is consistent with reference fock sectors
        occs = diag(rdm1.a)
        occs[ci.orb_list] .= 0
        na_embed = sum(occs)
        occs = diag(rdm1.b)
        occs[ci.orb_list] .= 0
        nb_embed = sum(occs)
        verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f\n", na_embed, nb_embed)


        #
        # Loop over sectors and do FCI or RASCI for each
        basis_i = ClusterBasis(ci, T=T, A=typeof(fock_ansatze[i][1]))
        for ansatz in fock_ansatze[i]
            sec = (ansatz.na, ansatz.nb)
            #
            # prepare for CI calculation for give sector of Fock space
            verbose == 0 || @printf(" Preparing to compute : \n")
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)

            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr 
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)
                basis_i[sec] = Solution(ansatz, F.values[1:nr], F.vectors[:,1:nr])
                #display(e)
            else
                #
                # Do sparse build 
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr, maxiter=1000, package="arpack", tol=1e-7))
            end



            #
            # Loop over spin-flips
            # 
            # s2 = s(s+1) 
            
            s2 = compute_s2(basis_i[sec])    

            nr = length(basis_i[sec].energies)
            #for r in 1:nr
            #    S = (-1 + sqrt(1+4*s2[r]))/2
            #    gr = 2*S+1 # Degeneracy
            #end
          
            #
            #   S-
            #
            # find how many applications of S- we need to try
           
            verbose == 0 || println(" Compute higher and lower Ms components")

            n_sm = minimum((sec[1], ansatz.no-sec[2]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for smi in 1:n_sm
                vi, ansatzi = apply_sminus(vi, ansatzi)

                verbose == 0 || display(ansatzi) 
                flush(stdout)

                if size(vi,2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi*vi))
                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end
            #
            #   S+
            #
            # find how many applications of S+ we need to try
            
            n_sp = minimum((sec[2], ansatz.no-sec[1]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for spi in 1:n_sp
                #display(ansatzi)
                vi, ansatzi = apply_splus(vi, ansatzi)
                
                verbose == 0 || display(ansatzi) 
                flush(stdout)

                if size(vi,2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi*vi))
            
                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end

        end
           
        flush(stdout)
        if verbose > 0
            println()
            for (sec, sol) in basis_i    
                println()
                display(sol.ansatz)
                s2 = compute_s2(sol)    
                for i in 1:length(sol.energies)
                    @printf("   State %4i Energy: %12.8f S2: %12.8f\n",i, sol.energies[i], s2[i])
                end
                flush(stdout)
            end
        end

        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#


"""
    compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{MOCluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing, T::Type=Float64)

Return a Vector of `ClusterBasis` for each `Cluster` 
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{MOCluster}`: Clusters 
- `verbose::Int`: Print level
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `delta_elec`: number of electrons different from reference (init_fspace)
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1a`: background density matrix for embedding local hamiltonian (alpha)
- `rdm1b`: background density matrix for embedding local hamiltonian (beta)
- `ansatze`: should be a list of Ansatz objects so that we know how to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{MOCluster}; 
                init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
                rdm1a=nothing, rdm1b=nothing, 
                ansatze=nothing,     
                T::Type=Float64, A::Type=FCIAnsatz)
#={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose == 0 || display(ci)
        
        if (rdm1a !== nothing && init_fspace == nothing)
            error(" Cant embed without init_fspace")
        end

        #
        # Get subset of integrals living on cluster, ci
        if rdm1a === nothing && rdm1b === nothing
            ints_i = subset(ints, ci.orb_list) 
        else
            ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 
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
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T) 
        for sec in sectors
            
            #
            # prepare for FCI calculation for give sector of Fock space
            ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)
            
            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr 
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)
                basis_i[sec] = Solution(ansatz, Vector{T}(F.values[1:nr]), Matrix{T}(F.vectors[:,1:nr]))

                #vecs = Matrix(1.0I, nr, nr)
                #energies = diag(vecs' * Hmat * vecs)
                #basis_i[sec] = Solution(ansatz, energies, vecs)

                #display(e)
            else
                #
                # Do sparse build 
                #if ansatz.dim > 3000
                #    display(norm(ints_i.h1))
                #    display(norm(ints_i.h2))
                #end
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr))
            end
            if verbose > 0
                state=1
                for ei in basis_i[sec].energies
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
                flush(stdout)
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#

"""
    compute_cluster_eigenbasis(        ints::InCoreInts{T}, 
                                       clusters::Vector{MOCluster}, 
                                       rdm1::RDM1{T},
                                       delta_elec::Vector,
                                       ref_fock::FockConfig,
                                       ansatze::Vector{<:Ansatz};
                                       verbose=0, 
                                       max_roots=10) where T

Return a Vector of `ClusterBasis` for each `Cluster`.
For each number of electrons specified by ref_fock +- 1->delta_elec (for each cluster)

# Arguments
#
- `ints`: InCoreInts integrals
- `clusters`: Clusters 
- `verbose`: Print level
- `ref_fock`:  reference space for defining target focksectors with `delta_elec`
- `delta_elec`: number of electrons different from reference (init_fspace) for each cluster
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1`: background density matrix for embedding local hamiltonian 
- `A`: the type of Ansatz object used to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis(    ints::InCoreInts{T}, 
                                        clusters::Vector{MOCluster}, 
                                        rdm1::RDM1{T},
                                        delta_elec::Vector,
                                        ref_fock::FockConfig,
                                        ansatze::Vector{<:Ansatz};
                                        verbose=0, 
                                        max_roots=10) where T
#={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{<:Ansatz,T}}()
    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)") 

    for i in 1:length(clusters)
        ci = clusters[i]
        verbose == 0 || display(ci)
        
        ints_i = subset(ints, ci, rdm1) 
        
        # Verify that density matrix provided is consistent with reference fock sectors
        occs = diag(rdm1.a)
        occs[ci.orb_list] .= 0
        na_embed = sum(occs)
        occs = diag(rdm1.b)
        occs[ci.orb_list] .= 0
        nb_embed = sum(occs)
        verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f\n", na_embed, nb_embed)
        
        delta_e_i = ()
        if isempty(delta_elec) == false
            delta_e_i = (ref_fock[ci.idx][1], ref_fock[ci.idx][2], delta_elec[i])
        end
        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)
        display(sectors)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T, A=typeof(ansatze[i]))
        for sec in sectors
            if typeof(ansatze[i]) == FCIAnsatz
                ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            elseif typeof(ansatze[i]) == RASCIAnsatz_2
                ansatz = RASCIAnsatz_2(length(ci), sec[1], sec[2], ansatze[i].ras_spaces, max_h=ansatze[i].max_h, max_p=ansatze[i].max_p)
            elseif typeof(ansatze[i]) == RASCIAnsatz
                ansatz = RASCIAnsatz(length(ci), sec[1], sec[2], ansatze[i].ras_spaces, max_h=ansatze[i].max_h, max_p=ansatze[i].max_p)
            elseif typeof(ansatze[i]) == DDCIAnsatz
                ansatz = DDCIAnsatz(length(ci), sec[1], sec[2], ansatze[i].ras_spaces, ex_level=ansatze[i].ex_level)
            end
            # prepare for CI calculation for give sector of Fock space
            verbose == 0 || @printf(" Preparing to compute : \n")
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)

            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr 
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)
                basis_i[sec] = Solution(ansatz, F.values[1:nr], F.vectors[:,1:nr])
                #vecs = Matrix(1.0I, nr, nr)
                #energies = diag(vecs' * Hmat * vecs)
                #basis_i[sec] = Solution(ansatz, energies, vecs)
                
            else
                #
                # Do sparse build 
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr))
            end

            if verbose > 0
                state=1
                for ei in basis_i[sec].energies
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
                flush(stdout)
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#



"""
    compute_cluster_est_basis(ints::InCoreInts, clusters::Vector{MOCluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing)

Return a Vector of `ClusterBasis` for each `Cluster`  using the Embedded Schmidt Truncation
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{MOCluster}`: Clusters 
- `Da`: background density matrix for embedding local hamiltonian (alpha)
- `Db`: background density matrix for embedding local hamiltonian (beta)
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `thresh_schmidt`: the threshold for the EST 
- `thresh_orb`: threshold for the orbital
- `thresh_ci`: threshold for the ci problem
"""
function compute_cluster_est_basis(ints::InCoreInts{T}, clusters::Vector{MOCluster},Da,Db; 
                thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,
                do_embedding=true,verbose=0,init_fspace=nothing,delta_elec=nothing,
                est_nr=1, est_max_cycles=200, est_thresh=1e-6, 
                A::Type=FCIAnsatz) where T
#={{{=#
    # initialize output
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose == 0 || display(ci)

        # Obtain the schmidt basis
        basis = FermiCG.form_schmidt_basis(ints, ci, Da, Db,thresh_schmidt=thresh_schmidt,
					   eig_nr=est_nr, eig_max_cycles=est_max_cycles, thresh_ci=est_thresh)

        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec[ci.idx])
        end

        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci) 


        #for (key, value) in basis
        #    basis_i[key] = value
	#    println(key)
	#    display(value)
        #end


        for sec in sectors
            if sec in keys(basis) 
                basis_i[sec] = Solution(FCIAnsatz(length(ci), sec[1], sec[2]), zeros(size(basis[sec],2)), basis[sec])
		#display(basis[sec])
		#st = "fock_"*string(ci.idx)*"_"*string(sec)
		#npzwrite(st, Matrix(basis[sec]))
            else
	    	#println(sec)
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#
    

