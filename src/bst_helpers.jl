"""
    function add_double_excitons!(ts::BSTstate{T,N,R}, 
                              fock::FockConfig{N}) where {T,N,R}

Modify the current state by adding the "biexitonic" basis for the specified `FockConfig`. 
This basically, starts from a reference state where only the p-spaces are included,
and then adds the excited states. E.g., 
    |PPPP> += |QQPP> + |PQQP> + |PPQQ> + |QPPQ> 
"""

function add_double_excitons!(ts::BSTstate{T,N,R}, fock::FockConfig{N},num_states::Integer) where {T,N,R}
    ref_config = [ts.p_spaces[ci.idx][fock[ci.idx]] for ci in ts.clusters]
    num_clusters = length(ts.clusters)
    valid_pairs = [(i, j) for i in 1:num_clusters for j in i+1:num_clusters]

    for ci in ts.clusters
        
        conf_i = deepcopy(ref_config)
        # Check if there is a q space for this fock sector
        fock[ci.idx] in keys(ts.q_spaces[ci.idx].data) || continue

        conf_i[ci.idx] = ts.q_spaces[ci.idx][fock[ci.idx]]
        # Loop over clusters to set factors for double excitations
        for cj in ts.clusters
            conf_j = deepcopy(conf_i)
            # println(ci.idx, cj.idx)
            # Skip if the pair (ci.idx, cj.idx) is not in the list of valid pairs
            (ci.idx, cj.idx) in valid_pairs || continue

            # Check if there is a q space for this fock sector
            fock[cj.idx] in keys(ts.q_spaces[cj.idx].data) || continue
            conf_j[cj.idx] = ts.q_spaces[cj.idx][fock[cj.idx]]
            tconfig_j = TuckerConfig(conf_j)
            core = tuple([zeros(length.(tconfig_j)...) for r in 1:R]...)
            core,factors=tucker_initialize(core; num_roots=num_states)
            ts.data[fock][tconfig_j] = FermiCG.Tucker(FermiCG.Tucker(core, factors); R,T)
        end
    end
    return
end

"""
    function add_single_excitons!(ts::BSTstate{T,N,R}, 
                              fock::FockConfig{N}, 
                              cluster_bases::Vector{ClusterBasis}) where {T,N,R}

Modify the current state by adding the "single excitonic" basis for the specified `FockConfig`. 
This basically, starts from a reference state where only the p-spaces are included,
and then adds the excited states. E.g., 
    |PPPP> += |QPPP> + |PQPP> + |PPQP> + |PPPQ> 
"""
function add_single_excitons!(ts::BSTstate{T,N,R}, 
        fock::FockConfig{N},num_states::Integer) where {T,N,R}
    #={{{=#
    #length(size(v)) == 1 || error(" Only takes vectors", size(v))

    ref_config = [ts.p_spaces[ci.idx][fock[ci.idx]] for ci in ts.clusters]


    # println(ref_config)
    # println(TuckerConfig(ref_config))
    for ci in ts.clusters
        conf_i = deepcopy(ref_config)

        # Check to make sure there is a q space for this fock sector (e.g., (0,0) fock sector only has a P space 
        # since it is 1 dimensional)
        fock[ci.idx] in keys(ts.q_spaces[ci.idx].data) || continue

        conf_i[ci.idx] = ts.q_spaces[ci.idx][fock[ci.idx]]
        tconfig_i = TuckerConfig(conf_i)

        #factors = tuple([cluster_bases[j.idx][fock[j.idx]][:,tconfig_i[j.idx]] for j in ts.clusters]...)
        core = tuple([zeros(length.(tconfig_i)...) for r in 1:R]...)
        # factors = tuple([Matrix{T}(I, length(tconfig_i[j.idx]), length(tconfig_i[j.idx])) for j in ts.clusters]...)
        core,factors=tucker_initialize(core; num_roots=num_states)
        ts.data[fock][tconfig_i] = FermiCG.Tucker(FermiCG.Tucker(core, factors); R,T)
        # display(ts.data[fock][tconfig_i])
    end
    return
end
#=}}}=#