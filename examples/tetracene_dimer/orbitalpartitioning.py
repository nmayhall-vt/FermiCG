import numpy as np
import scipy
import warnings

def spade_partitioning(Cocc, Pv, S):
    pass


def dmet_clustering(Cocc, Cvir, frags, S):
    full = []
    [full.extend(i) for i in frags]
    nmo = Cocc.shape[1] + Cvir.shape[1]

    init_fspace = [] # Number of electrons in each cluster
    clusters = []

    Cenv_occ, Cact_occ, Cact_vir, Cenv_vir = dmet_active_space(Cocc, Cvir, full, S)
    
    nfrag = len(frags)
    print(" Partition %4i orbitals into a total of %4i fragments" %(nmo, nfrag))

    Clist = []

    orb_idx = 0
    for f in frags:
        _, fo, fv, _ = dmet_active_space(Cocc, Cvir, f, S)
        Clist.append(np.hstack((fo,fv)))

        ndocc_f = fo.shape[1]
        init_fspace.append((ndocc_f, ndocc_f))

        nmo_f = fo.shape[1] + fv.shape[1]
        clusters.append(list(range(orb_idx, orb_idx+nmo_f)))
        orb_idx += nmo_f

    
    Clist = sym_ortho(Clist, S)

    out = [Cenv_occ]
    [out.append(i) for i in Clist]
    out.append(Cenv_vir)

    print(" init_fspace = ", init_fspace)
    print(" clusters    = ", clusters)


    return out, init_fspace, clusters

def dmet_active_space(Cocc_in, Cvir_in, frag, S):
        
    X = scipy.linalg.sqrtm(S)
    Xinv = np.linalg.inv(X)

    Cocc = X@Cocc_in
    Cvir = X@Cvir_in

    nbas = S.shape[0]
    assert(Cocc.shape[0] == nbas)
    assert(Cvir.shape[0] == nbas)
    nmo = Cocc.shape[1] + Cvir.shape[1]

    nfrag = len(frag)

    print(" Create DMET active space by projecting %4i MOs onto %4i fragment orbitals" %(nmo, nfrag))
    # Occupieds
    _,s,V = np.linalg.svd(Cocc[frag,:], full_matrices=True)
    nocc = 0
    for si in s:
        if si > 1e-6:
            nocc += 1
        else:
            warnings.warn("Small singular values")

    Cact_occ = Cocc @ V[0:nocc,:].T
    Cenv_occ = Cocc @ V[nocc:,:].T

    # Virtuals
    _,s,V = np.linalg.svd(Cvir[frag,:], full_matrices=True)
    nocc = 0
    for si in s:
        if si > 1e-6:
            nocc += 1
        else:
            warnings.warn("Small singular values")
    Cact_vir = Cvir @ V[0:nocc,:].T
    Cenv_vir = Cvir @ V[nocc:,:].T

    # Un-Orthogonalize
    Cact_occ = Xinv@Cact_occ
    Cact_vir = Xinv@Cact_vir
    Cenv_occ = Xinv@Cenv_occ
    Cenv_vir = Xinv@Cenv_vir

    

    # print(Cenv.shape, Cact_occ.shape, Cact_vir.shape)

    print(" Dmet active space has the following dimensions:")
    print("   Environment (occupied)   : %5i" % Cenv_occ.shape[1])
    print("   Active (occupied)        : %5i" % Cact_occ.shape[1])
    print("   Active (virtual)         : %5i" % Cact_vir.shape[1])
    print("   Environment (virtual)    : %5i" % Cenv_vir.shape[1])
    return Cenv_occ, Cact_occ, Cact_vir, Cenv_vir


def canonicalize(orbital_blocks, F):
    """
    Given an AO Fock matrix, rotate each orbital block in `orbital_blocks` to diagonalize F
    """
    out = []
    for obi, ob in enumerate(orbital_blocks):
        fi = ob.T @ F @ ob
        fi = .5 * ( fi + fi.T )
        e, U =  np.linalg.eigh(fi)
        perm = e.argsort()
        e = e[perm]
        U = U[:,perm]
        out.append(ob @ U)
    return out

def extract_frontier_orbitals(orbital_blocks, F, dims):
    """
    Given an AO Fock matrix, split each orbital block into 3 spaces, NDocc, NAct, Nvirt

    `dims` = [(NDocc, NAct, Nvirt), (NDocc, NAct, Nvirt), ... ]
    `F`: the fock matrix  
    """
    NAOs = F.shape[0]
    tmp = canonicalize(orbital_blocks, F)
    env_blocks = []
    act_blocks = []
    vir_blocks = []
    for obi, ob in enumerate(tmp):
        assert(ob.shape[0] == NAOs)
        env_blocks.append(np.zeros((NAOs, 0)))
        act_blocks.append(np.zeros((NAOs, 0)))
        vir_blocks.append(np.zeros((NAOs, 0)))
    for obi, ob in enumerate(tmp):
        assert(np.sum(dims[obi]) == ob.shape[1])
        env_blocks[obi] = ob[:,0:dims[obi][0]]
        act_blocks[obi] = ob[:,dims[obi][0]:dims[obi][0]+dims[obi][1]]
        vir_blocks[obi] = ob[:,dims[obi][0]+dims[obi][1]:dims[obi][0]+dims[obi][1]+dims[obi][2]]

    return env_blocks, act_blocks, vir_blocks
    
"""
The `svd_subspace_partitioning` function appears to perform Singular Value Decomposition (SVD) based 
    subspace partitioning of orbitals. Here is a breakdown of the logic behind the function:

1. **Input Parameters:**
   - `orbitals_blocks`: List of orbital blocks, where each block is a matrix of orbitals. Common scenarios include `[Occ, Virt]` or `[Occ, Sing, Virt]`.
   - `Pv`: Projector matrix of shape `[AO, frag]` representing the orbitals to be partitioned.
   - `S`: Overlap matrix of the basis set.
  
2. **Initialization:**
   - `nfrag`: Number of fragments (fragments correspond to the columns of `Pv`).
   - `nbas`: Number of basis functions.
   - `nmo`: Total number of molecular orbitals, initialized to 0.
   - Loop over `orbitals_blocks` to check if the shape of each block matches the number of basis functions and update `nmo`.

3. **Transformation Matrix X:**
   - Calculate the square root of the overlap matrix `S` using `scipy.linalg.sqrtm` and store it in `X`.

4. **Partitioning the Orbitals:**
   - Construct the projector `P` by applying transformations to `Pv`.
   - Use SVD on the transformed orbitals (`X @ P @ S @ ob`), where `ob` is an orbital block.
   - Store the singular values, left singular vectors, and right singular vectors in `s`, `Clist`, and `Vob` respectively.

5. **Sorting Singular Values:**
   - Sort the singular values in descending order along with their corresponding indices.
   - Update the singular values `s` and the corresponding spaces.

6. **Constructing Total Singular Vectors Matrix:**
   - Concatenate the matrices in `Clist` horizontally to form the total singular vectors matrix `Ctot`.

7. **Printing Information:**
   - Display information about the singular values, their corresponding indices, and the orbital blocks they belong to.

8. **Active Space Dimensions:**
   - Display the dimensions of the active space for each orbital block.

9. **Return:**
   - Return lists `Cf` and `Ce` representing the fragment and environment orbitals respectively.

The function aims to partition the orbitals into fragments and environment based on their overlap with the projector `Pv` using SVD. The logic involves manipulatin
g the orbital matrices and singular value decomposition to identify the active space for each orbital block. The active space is divided into fragment and environment orbitals,
and the dimensions of the active space are printed for each orbital block.

"""
def svd_subspace_partitioning(orbitals_blocks, Pv, S):
    """
    Find orbitals that most strongly overlap with the projector, P,  by doing rotations within each orbital block. 
    [C1, C2, C3] -> [(C1f, C2f, C3f), (C1e, C2e, C3e)]
    where C1f (C2f) and C1e (C2e) are the fragment orbitals in block 1 (2) and remainder orbitals in block 1 (2).

    Common scenarios would be 
        `orbital_blocks` = [Occ, Virt]
        or 
        `orbital_blocks` = [Occ, Sing, Virt]
    
    P[AO, frag]
    O[AO, occupied]
    U[AO, virtual]
    """

    nfrag = Pv.shape[1]
    nbas = S.shape[0]
    assert(Pv.shape[0] == nbas)
    nmo = 0
    for i in orbitals_blocks:
        assert(i.shape[0] == nbas)
        nmo += i.shape[1]


    X = scipy.linalg.sqrtm(S)

    print(" Partition %4i orbitals into a total of %4i orbitals" %(nmo, Pv.shape[1]))
    P = Pv @ np.linalg.inv(Pv.T @ S @ Pv) @ Pv.T


    s = []
    Clist = []
    spaces = []
    Cf = []
    Ce = []
    for obi, ob in enumerate(orbitals_blocks):
        _,sob,Vob = np.linalg.svd(X @ P @ S @ ob, full_matrices=True)
        s.extend(sob)
        Clist.append(ob @ Vob.T)
        spaces.extend([obi for i in range(ob.shape[1])])
        Cf.append(np.zeros((nbas, 0)))
        Ce.append(np.zeros((nbas, 0)))

    spaces = np.array(spaces)
    s = np.array(s)

    # Sort all the singular values
    perm = np.argsort(s)[::-1]
    s = s[perm]
    spaces = spaces[perm]

    Ctot = np.hstack(Clist)
    Ctot = Ctot[:,perm]    

    print(" %16s %12s %-12s" %("Index", "Sing. Val.", "Space"))
    for i in range(nfrag):
        print(" %16i %12.8f %12s*" %(i, s[i], spaces[i]))
        block = spaces[i]
        Cf[block] = np.hstack((Cf[block], Ctot[:,i:i+1]))

    for i in range(nfrag, nmo):
        if s[i] > 1e-6:
            print(" %16i %12.8f %12s" %(i, s[i], spaces[i]))
        block = spaces[i]
        Ce[block] = np.hstack((Ce[block], Ctot[:,i:i+1]))

    print("  SVD active space has the following dimensions:")
    print(" %14s %14s %14s" %("Orbital Block", "Environment", "Active"))
    for obi,ob in enumerate(orbitals_blocks):
        print(" %14i %14i %14i" %(obi, Ce[obi].shape[1], Cf[obi].shape[1]))
        assert(abs(np.linalg.det(ob.T @ S @ ob)) > 1e-12)

    return Cf, Ce 

def svd_subspace_partitioning_orth(orbitals_blocks, frag, S):
    """
    Find orbitals that most strongly overlap with the atomic orbitals listed in `frag` by doing rotations within each orbital block. 
    [C1, C2, C3] -> [(C1f, C2f, C3f), (C1e, C2e, C3e)]
    where C1f (C2f) and C1e (C2e) are the fragment orbitals in block 1 (2) and remainder orbitals in block 1 (2).

    Common scenarios would be 
        `orbital_blocks` = [Occ, Virt]
        or 
        `orbital_blocks` = [Occ, Sing, Virt]
    
    frag[ao1, ao2, ...] 
    O[AO, occupied]
    U[AO, virtual]

    frag listed here are assumed to be orthogonalized AOs
    """

    print(" In svd_subspace_partitioning_orth")
    nbas = S.shape[0]
    assert(len(orbitals_blocks)) > 0

    I = np.eye(nbas)

    # Define projectors
    X = scipy.linalg.sqrtm(S)
    Xinv = np.linalg.inv(X)

    # acts, envs = svd_subspace_partitioning([X@o for o in orbitals_blocks], X[:,frag], I)
    # return [Xinv@o for o in acts], [Xinv@o for o in envs]
    
    return svd_subspace_partitioning(orbitals_blocks, X[:,frag], S)



def svd_subspace_partitioning_nonorth(orbitals_blocks, frag, S):
    """
    Find orbitals that most strongly overlap with the atomic orbitals listed in `frags` by doing rotations within each orbital block. 
    [C1, C2, C3] -> [(C1f, C2f, C3f), (C1e, C2e, C3e)]
    where C1f (C2f) and C1e (C2e) are the fragment orbitals in block 1 (2) and remainder orbitals in block 1 (2).

    Common scenarios would be 
        `orbital_blocks` = [Occ, Virt]
        or 
        `orbital_blocks` = [Occ, Sing, Virt]
    
    frags[ao1, ao2, ...] 
    O[AO, occupied]
    U[AO, virtual]

    NOTE: frags listed here are assumed to be the non-orthogonal AOs
    """


    print(" In svd_subspace_partitioning_nonorth")
    nbas = S.shape[0]
    assert(len(orbitals_blocks)) > 0

    I = np.eye(nbas)

    return svd_subspace_partitioning(orbitals_blocks, I[:,frag], S)





def sym_ortho(frags, S, thresh=1e-8):
    """
    Orthogonalize list of MO coefficients. 
    
    `frags` is a list of mo-coeff matrices, e.g., [C[ao,mo], C[ao, mo], ...]
    """
    Nbas = S.shape[1]
    
    inds = []
    Cnonorth = np.hstack(frags)
    shift = 0
    for f in frags:
        inds.append(list(range(shift, shift+f.shape[1])))
        shift += f.shape[1]
        
    
    Smo = Cnonorth.T @ S @ Cnonorth
    X = np.linalg.inv(scipy.linalg.sqrtm(Smo))
    # print(Cnonorth.shape, X.shape)
    Corth = Cnonorth @ X
    
    frags2 = []
    for f in inds:
        frags2.append(Corth[:,f])
    return frags2






if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass