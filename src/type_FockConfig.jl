
"""
    config::NTuple{N,Tuple{Int16,Int16}}

Indexes a 'Change in fock space'. For instance, if `F1` and `F2` are distince `FockConfig` instances, 
then `TransferConfig` is `F1-F2`.

e.g., `config = ((2,3), (2,2), (3,2))` is a 3 cluster configuration with 2, 2, 3 α and 3, 2, 2 β electrons, respectively.  
"""
struct FockConfig{N} <: SparseIndex
    config::NTuple{N,Tuple{Int16,Int16}}
end

FockConfig(in::Vector{Tuple{T,T}}) where T = convert(FockConfig{length(in)}, in)


"""
    function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
"""
function Base.convert(::Type{FockConfig{N}}, in::Vector) where {N}
    return FockConfig{length(in)}(ntuple(i -> Tuple(convert.(Int16, in[i])), length(in)))
end


"""
    dim(fc::FockConfig, no)

Return total dimension of space indexed by `fc` on `no` orbitals
"""
function dim(fc::FockConfig, clusters)
    dim = 1
    for ci in clusters
        dim *= binomial(BigInt(length(ci)), fc[ci.idx][1]) * binomial(BigInt(length(ci)), fc[ci.idx][2])
    end
    return dim
end

"""
    function replace(tc::FockConfig, idx, fock)
"""
function replace(tc::FockConfig, idx, fock)
    new = [tc.config...]
    length(idx) == length(fock) || error("wrong dimensions")
    for i in 1:length(idx)
        new[idx[i]] = (convert(Int16, fock[i][1]), convert(Int16, fock[i][2]))
    end
    return FockConfig(new)
end

"""
    n_elec_a(fc::FockConfig)

Return number of alpha electrons in `fc`
"""
function n_elec_a(fc::FockConfig)
    return sum([i[1] for i in fc])
end

"""
    n_elec_b(fc::FockConfig)

Return number of beta electrons in `fc`
"""
function n_elec_b(fc::FockConfig)
    return sum([i[2] for i in fc])
end

"""
    n_elec(fc::FockConfig)

Return number of electrons in `fc`
"""
function n_elec(fc::FockConfig)
    return n_elec_a(fc) + n_elec_b(fc)
end


"""
    possible_spin_focksectors(clusters, ref_fock; verbose=1)

Generate list of focksectors needed to spin-complete ref_fock
"""
function possible_spin_focksectors(clusters, ref_fock; verbose=1)
    spaces = Vector{Vector{Tuple{Int,Int}}}([])
    for ci in clusters

        na = ref_fock[ci.idx][1]
        nb = ref_fock[ci.idx][2]
        
        spacesi = [(na, nb)]
        nai = na
        nbi = nb
        if na > nb 
            mult = na-nb + 1; 
            for m = 2:mult 
                nai -= 1
                nbi += 1
                push!(spacesi, (nai, nbi))
            end
        end
        if na < nb 
            mult = nb-na + 1; 
            for m = 2:mult 
                nai += 1
                nbi -= 1
                push!(spacesi, (nai, nbi))
            end
        end
        push!(spaces, spacesi)
    end
    out = Vector{typeof(ref_fock)}([])
    for (fi,f) in enumerate(Iterators.product(spaces...))
        focki = FockConfig([f...])
        FermiCG.n_elec_a(focki) == FermiCG.n_elec_a(ref_fock) || continue
        FermiCG.n_elec_b(focki) == FermiCG.n_elec_b(ref_fock) || continue
        verbose < 1 || display(focki)
        push!(out, focki)
    end

    return out
end



