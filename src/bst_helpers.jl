"""
    function add_double_excitons!(ts::BSTstate{T,N,R}, 
                              fock::FockConfig{N}) where {T,N,R}

Modify the current state by adding the "biexitonic" basis for the specified `FockConfig`. 
This basically, starts from a reference state where only the p-spaces are included,
and then adds the excited states. E.g., 
    |PPPP> += |QQPP> + |PQQP> + |PPQQ> + |QPPQ> 
"""

function add_double_excitons!(ts::BSTstate{T,N,R}, fock::FockConfig{N}) where {T,N,R}
    ref_config = [ts.p_spaces[ci.idx][fock[ci.idx]] for ci in ts.clusters]

    for ci in ts.clusters
        conf_i = deepcopy(ref_config)

        # Check if there is a q space for this fock sector
        fock[ci.idx] in keys(ts.q_spaces[ci.idx].data) || continue

        conf_i[ci.idx] = ts.q_spaces[ci.idx][fock[ci.idx]]
        # Loop over clusters to set factors for double excitations
        for cj in ts.clusters
            # Skip if the cluster is the same as ci
            if ci.idx == cj.idx
                continue
            end

            # Check if there is a q space for this fock sector
            fock[cj.idx] in keys(ts.q_spaces[cj.idx].data) || continue
            conf_i[cj.idx] = ts.q_spaces[cj.idx][fock[cj.idx]]
            tconfig_j = TuckerConfig(conf_i)
            core = tuple([zeros(length.(tconfig_j)...) for r in 1:R]...)
            # factors = tuple([Matrix{T}(I, length(tconfig_j[j.idx]), length(tconfig_j[j.idx])) for j in ts.clusters]...)
            core,factors=tucker_initialize(core; num_roots=R)
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
        fock::FockConfig{N}) where {T,N,R}
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
        core,factors=tucker_initialize(core; num_roots=R)
        ts.data[fock][tconfig_i] = FermiCG.Tucker(FermiCG.Tucker(core, factors); R,T)
        # display(ts.data[fock][tconfig_i])
    end
    return
end
#=}}}=#