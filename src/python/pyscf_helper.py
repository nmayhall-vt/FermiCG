import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

class PyscfHelper(object):
    """
    Pyscf is used to generate 
    integrals etc for the TPSCI program. this is a class which keeps info about pyscf mol info and HF stuff
    """

    def __init__(self):

        self.mf  = None
        self.mol = None

        self.h      = None
        self.g      = None
        self.n_orb  = None
        #self.na     = 0
        #self.nb     = 0
        self.ecore  = 0
        self.C      = None
        self.S      = None
        self.J      = None
        self.K      = None

    def init(self,molecule,charge,spin,basis_set,orb_basis='scf',cas=False,cas_nstart=None,cas_nstop=None,cas_nel=None,loc_nstart=None,loc_nstop=None):
    # {{{
        import pyscf
        from pyscf import gto, scf, ao2mo, molden, lo
        pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
        #PYSCF inputs
        print(" ---------------------------------------------------------")
        print("                      Using Pyscf:")
        print(" ---------------------------------------------------------")
        print("                                                          ")

        mol = gto.Mole()
        mol.atom = molecule

        mol.max_memory = 1000 # MB
        mol.symmetry = True
        mol.charge = charge
        mol.spin = spin
        mol.basis = basis_set
        mol.build()
        print("symmertry")
        print(mol.topgroup)

        #SCF 

        #mf = scf.RHF(mol).run(init_guess='atom')
        mf = scf.RHF(mol).run()
        #C = mf.mo_coeff #MO coeffs
        enu = mf.energy_nuc()
       
        print(mf.get_fock())
        print(np.linalg.eig(mf.get_fock())[0])
        
        if mol.symmetry == True:
            from pyscf import symm
            mo = symm.symmetrize_orb(mol, mf.mo_coeff)
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in range(len(osym)):
                print("%4d %8s %16.8f"%(i+1,osym[i],mf.mo_energy[i]))

        #orbitals and lectrons
        n_orb = mol.nao_nr()
        n_b , n_a = mol.nelec 
        nel = n_a + n_b
        self.n_orb = mol.nao_nr()


        json_mol = pyscf.gto.mole.dumps(mol)
        import json
        with open('data/mol.json', 'w') as fp:
            json.dump(json_mol, fp)
        
        if cas == True:
            cas_norb = cas_nstop - cas_nstart
            from pyscf import mcscf
            assert(cas_nstart != None)
            assert(cas_nstop != None)
            assert(cas_nel != None)
        else:
            cas_nstart = 0
            cas_nstop = n_orb
            cas_nel = nel

        ##AO 2 MO Transformation: orb_basis or scf
        if orb_basis == 'scf':
            print("\nUsing Canonical Hartree Fock orbitals...\n")
            C = cp.deepcopy(mf.mo_coeff)
            print("C shape")
            print(C.shape)

        elif orb_basis == 'lowdin':
            assert(cas == False)
            S = mol.intor('int1e_ovlp_sph')
            print("Using lowdin orthogonalized orbitals")

            C = lowdin(S)
            #end

        elif orb_basis == 'boys':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'boys2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.Boys(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'PM':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'PM2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ER':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ER2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.ER(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ibmo':
            loc_vstop =  loc_nstop - n_a
            print(loc_vstop)

            mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
            mo_vir = mf.mo_coeff[:,mf.mo_occ==0]
            c_core = mo_occ[:,:loc_nstart]
            iao_occ = lo.iao.iao(mol, mo_occ[:,loc_nstart:])
            iao_vir = lo.iao.iao(mol, mo_vir[:,:loc_vstop])
            c_out  = mo_vir[:,loc_vstop:]

            # Orthogonalize IAO
            iao_occ = lo.vec_lowdin(iao_occ, mf.get_ovlp())
            iao_vir = lo.vec_lowdin(iao_vir, mf.get_ovlp())

            #
            # Method 1, using Knizia's alogrithm to localize IAO orbitals
            #
            '''
            Generate IBOS from orthogonal IAOs
            '''
            ibo_occ = lo.ibo.ibo(mol, mo_occ[:,loc_nstart:], iao_occ)
            ibo_vir = lo.ibo.ibo(mol, mo_vir[:,:loc_vstop], iao_vir)

            C = np.column_stack((c_core,ibo_occ,ibo_vir,c_out))

        else: 
            print("Error:NO orbital basis defined")

        molden.from_mo(mol, 'orbitals.molden', C)

        if cas == True:
            print(C.shape)
            print(cas_norb)
            print(cas_nel)
            mycas = mcscf.CASSCF(mf, cas_norb, cas_nel)
            h1e_cas, ecore = mycas.get_h1eff(mo_coeff = C)  #core core orbs to form ecore and eff
            h2e_cas = ao2mo.kernel(mol, C[:,cas_nstart:cas_nstop], aosym='s4',compact=False).reshape(4 * ((cas_norb), )) 
            print(h1e_cas)
            print(h1e_cas.shape)
            #return h1e_cas,h2e_cas,ecore,C,mol,mf
            self.h = h1e_cas
            self.g = h2e_cas
            self.ecore = ecore
            self.mf = mf
            self.mol = mol
            self.C = cp.deepcopy(C[:,cas_nstart:cas_nstop])
            J,K = mf.get_jk()
            self.J = self.C.T @ J @ self.C
            self.K = self.C.T @ J @ self.C
            if 0:
                h = C.T.dot(mf.get_hcore()).dot(C)
                g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
                const,heff = get_eff_for_casci(cas_nstart,cas_nstop,h,g)
                print(heff)
                print("const",const)
                print("ecore",ecore)
                self.h = heff
                self.g = g

        elif cas==False:
            h = C.T.dot(mf.get_hcore()).dot(C)
            g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
            print(h)
            #return h, g, enu, C,mol,mf
            self.h = h
            self.g = g
            self.ecore = enu
            self.mf = mf
            self.mol = mol
            self.C = C
            J,K = mf.get_jk()
            self.J = self.C.T @ J @ self.C
            self.K = self.C.T @ J @ self.C
    # }}}

def run_fci_pyscf( h, g, nelec, ecore=0,nroots=1, conv_tol=None, max_cycle=None):
# {{{
    # FCI
    from pyscf import fci
    #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
    cisolver = fci.direct_spin1.FCI()
    if max_cycle != None:
        cisolver.max_cycle = max_cycle 
    if conv_tol != None:
        cisolver.conv_tol = conv_tol 
    efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots =nroots,verbose=100)
    fci_dim = ci.shape[0]*ci.shape[1]
    d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
    print(" PYSCF 1RDM: ")
    occs = np.linalg.eig(d1)[0]
    [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
    with np.printoptions(precision=6, suppress=True):
        print(d1)
    print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    #for i in range(0,nroots):
    #    print("FCI %10.8f"%(efci[i]))
    #exit()
    #fci_dim =1
            
    return efci,fci_dim
# }}}

def run_hci_pyscf( h, g, nelec, ecore=0, select_cutoff=5e-4, ci_cutoff=5e-4,nroots=1):
# {{{
    #heat bath ci
    from pyscf import mcscf
    from pyscf.hci import hci
    cisolver = hci.SCI()
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = ci_cutoff
    ehci, civec = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,verbose=4,nroots=nroots)
    hci_dim = civec[0].shape[0]
    print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
    print("HCI %10.8f"%(ehci))
    #for i in range(0,nroots):
    #    print("HCI %10.8f"%(ehci[i]))
    #hci_dim = 1
    return ehci,hci_dim
# }}}

def lowdin(S):
# {{{
    print("Using lowdin orthogonalized orbitals")
    #forming S^-1/2 to transform to A and B block.
    sal, svec = np.linalg.eigh(S)
    idx = sal.argsort()[::-1]
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal**-0.5
    sal = np.diagflat(sal)
    X = svec @ sal @ svec.T
    return X
# }}}

def reorder_integrals(idx,h,g):
# {{{
    h = h[:,idx] 
    h = h[idx,:] 

    g = g[:,:,:,idx] 
    g = g[:,:,idx,:] 
    g = g[:,idx,:,:] 
    g = g[idx,:,:,:] 
    return h,g
# }}}

def e1_order(h,cut_off):
# {{{
    hnew = np.absolute(h)
    hnew[hnew < cut_off] = 0
    np.fill_diagonal(hnew, 0)
    print(hnew)
    import scipy.sparse
    idx = scipy.sparse.csgraph.reverse_cuthill_mckee(
        scipy.sparse.csr_matrix(hnew))
    print(idx)
    idx = idx 
    hnew = hnew[:, idx]
    hnew = hnew[idx, :]
    print("New order")
    print(hnew)
    return idx
# }}}

def ordering(pmol,cas,cas_nstart,cas_nstop,loc_nstart,loc_nstop,ordering='hcore'):
# {{{
    loc_range = np.array(list(range(loc_nstart-cas_nstart,loc_nstop-cas_nstart)))
    #cas_range = range(cas_nstart,cas_nstop)
    out_range = np.array(list(range(loc_nstop-cas_nstart,cas_nstop-cas_nstart)))
    print(loc_range)
    print(out_range)

    h = cp.deepcopy(pmol.h)
    print(h)
    if ordering == 'hcore':
        print("Bonding Active Space")
        hl = h[:,loc_range]
        hl = hl[loc_range,:]
        print(hl)
        idl = e1_order(hl,cut_off = 1e-2)

        ho = h[:,out_range]
        ho = ho[out_range,:]
        print("Virtual Active Space")
        ido = e1_order(ho,cut_off = 1e-2)

        idl = idl 
        ido = ido + loc_nstop - cas_nstart 

    print(idl)
    print(ido)
    idx = np.append(idl,ido)
    print(idx)
    return idx
    # }}}

def mulliken_ordering(mol,norb,C):
# {{{
    """
    pyscf mulliken
    """
    S = mol.intor('int1e_ovlp_sph')
    mulliken = np.zeros((mol.natm,norb))
    for i in range(0,norb):
        Cocc = C[:,i].reshape(C.shape[0],1)
        temp = Cocc @ Cocc.T @ S   
        for m,lb in enumerate(mol.ao_labels()):
            mulliken[int(lb[0]),i] += temp[m,m]
    print(mulliken)
    return mulliken
# }}}

def block_order_mulliken(n_blocks,n_orb,mulliken,atom_block): 
# {{{
    blocks = [[] for i in range(n_blocks)]
    for i in range(0,n_orb):
        atom = mulliken[:,i].argmax(axis=0)

        for ind,bl in enumerate(atom_block):
            if atom in bl:
                #print(ind)
                blocks[ind].append(i)
        print(blocks)
    return blocks
# }}}

def get_eff_for_casci(n_start,n_stop,h,g):
# {{{
    const = 0
    for i in range(0,n_start):
        const += 2 * h[i,i]
        for j in range(0,n_start):
            const += 2 * g[i,i,j,j] -  g[i,j,i,j]

    eff = np.zeros((n_stop - n_start,n_stop - n_start))

    for l in range(n_start,n_stop):
        L = l - n_start
        for m in range(n_start,n_stop):
            M = m - n_start
            for j in range(0,n_start):
                eff[L,M] += 2 * g[l,m,j,j] -  g[l,j,j,m]
    return const, eff
# }}}


def get_pi_space(mol,mf,cas_norb,cas_nel,local=True):
# {{{
    from pyscf import mcscf, mo_mapping, lo, ao2mo
    # find the 2pz orbitals using mo_mapping
    ao_labels = ['C 2pz']
    pop = mo_mapping.mo_comps(ao_labels, mol, mf.mo_coeff)
    cas_list = np.sort(pop.argsort()[-cas_norb:])  #take the 2z orbitals and resort in MO order
    print('Population for pz orbitals', pop[cas_list])
    mo_occ = np.where(mf.mo_occ>0)[0]
    focc_list = list(set(mo_occ)-set(cas_list))
    focc = len(focc_list)

    # localize the active space
    if local:
        cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_list]).kernel(verbose=4)
        C = mf.mo_coeff 
        C[:,cas_list] = cl_a
    else:
        C = mf.mo_coeff
        mo_energy = mf.mo_energy[cas_list]

        if mol.symmetry == True:
            from pyscf import symm
            mo = symm.symmetrize_orb(mol, C[:,cas_list])
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in range(len(osym)):
                print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))

    # reorder the orbitals to get docc,active,vir ordering  (Note:sort mo takes orbital ordering from 1)
    mycas = mcscf.CASCI(mf, cas_norb, cas_nel)
    C = mycas.sort_mo(cas_list+1,mo_coeff=C)

    # Get the active space integrals and the frozen core energy
    h, ecore = mycas.get_h1eff(C)
    g = ao2mo.kernel(mol,C[:,focc:focc+cas_norb], aosym = 's4', compact = False).reshape(4*((cas_norb),))
    C = C[:,focc:focc+cas_norb]  #only carrying the active sapce orbs
    return h,ecore,g,C
# }}}

def get_ao_space(mol,mf,cas_norb,cas_nel,ao_label,local=False):
# {{{
    from pyscf import mcscf, mo_mapping, lo, ao2mo
    # find the 2pz orbitals using mo_mapping
    pop = mo_mapping.mo_comps(ao_labels, mol, mf.mo_coeff)
    cas_list = np.sort(pop.argsort()[-cas_norb:])  #take the 2z orbitals and resort in MO order
    print('Population for pz orbitals', pop[cas_list])
    mo_occ = np.where(mf.mo_occ>0)[0]
    focc_list = list(set(mo_occ)-set(cas_list))
    focc = len(focc_list)

    # localize the active space
    if local:
        cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_list]).kernel(verbose=4)
        C = mf.mo_coeff 
        C[:,cas_list] = cl_a
    else:
        C = mf.mo_coeff
        mo_energy = mf.mo_energy[cas_list]

        if mol.symmetry == True:
            from pyscf import symm
            mo = symm.symmetrize_orb(mol, C[:,cas_list])
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in range(len(osym)):
                print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))

    # reorder the orbitals to get docc,active,vir ordering  (Note:sort mo takes orbital ordering from 1)
    mycas = mcscf.CASCI(mf, cas_norb, cas_nel)
    C = mycas.sort_mo(cas_list+1,mo_coeff=C)

    # Get the active space integrals and the frozen core energy
    h, ecore = mycas.get_h1eff(C)
    g = ao2mo.kernel(mol,C[:,focc:focc+cas_norb], aosym = 's4', compact = False).reshape(4*((cas_norb),))
    C = C[:,focc:focc+cas_norb]  #only carrying the active sapce orbs
    return h,ecore,g,C
# }}}

def get_pi_space_local_split(mol,mf,cas_norb,cas_nel):
# {{{
    from pyscf import mcscf, mo_mapping, lo, ao2mo
    # find the 2pz orbitals using mo_mapping
    ao_labels = ['C 2pz']
    pop = mo_mapping.mo_comps(ao_labels, mol, mf.mo_coeff)
    cas_list = np.sort(pop.argsort()[-cas_norb:])  #take the 2z orbitals and resort in MO order
    print('Population for pz orbitals', pop[cas_list])
    mo_occ = np.where(mf.mo_occ>0)[0]
    focc_list = list(set(mo_occ)-set(cas_list))

    mo_vir = np.where(mf.mo_occ==0)[0]
    fvir_list = list(set(mo_vir)-set(cas_list))
    focc = len(focc_list)
    # localize the active space

    C = mf.mo_coeff
    ##### New stuff
    occ_l = cas_list[:cas_norb//2]
    occ_v = cas_list[cas_norb//2:]

    if 0:
        boys = lo.Boys(mol, mf.mo_coeff[:,occ_l])
        #boys.init_guess = None
        cl_a = boys.kernel()

        boys = lo.Boys(mol, mf.mo_coeff[:,occ_v])
        #boys.init_guess = None
        cl_b = boys.kernel()
    else:
        cl_a = lo.Boys(mol, mf.mo_coeff[:, occ_l]).kernel(verbose=4)
        cl_b = lo.Boys(mol, mf.mo_coeff[:, occ_v]).kernel(verbose=4)

    C[:,occ_l] = cl_a
    C[:,occ_v] = cl_b


    # reorder the orbitals to get docc,active,vir ordering  (Note:sort mo takes orbital ordering from 1)
    mycas = mcscf.CASCI(mf, cas_norb, cas_nel)
    C = mycas.sort_mo(cas_list+1,mo_coeff=C)
    # Get the active space integrals and the frozen core energy
    h, ecore = mycas.get_h1eff(C)
    g = ao2mo.kernel(mol,C[:,focc:focc+cas_norb], aosym = 's4', compact = False).reshape(4*((cas_norb),))
    C = C[:,focc:focc+cas_norb]  #only carrying the active sapce orbs
    return h,ecore,g,C
# }}}

def kekule_type_order(mol,h,g,C):
# {{{
    cas_norb = h.shape[0]
    mc = mulliken_ordering(mol, h.shape[0], C) # Reorder orbitals to same order as C atoms
    mc[abs(mc) < 0.2] = 0
    print(mc)
    atom_list = []
    for i in range(0,cas_norb):
        #print(mc[:,i])
        idx = np.array(np.where(mc[:,i]> 0.2)).tolist()
        print(i,idx)
        atom_list.append(idx[0])

    print(atom_list)

    blocks = []
    occ_list = []
    vir_list = []
    for i in range(0,cas_norb//2):
        for j in range(cas_norb//2,cas_norb):
            if atom_list[i] == atom_list[j]:
                blocks.append([i,j])
                occ_list.append(i)
                vir_list.append(j)

    print(blocks)
    print("Occ list:",occ_list)
    print("Vir list:",vir_list)

    idx = cp.deepcopy(occ_list)
    idx.extend(vir_list)

    # Reorder 
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx] # make sure u reorder this too
    #molden.from_mo(mol, 'cas.molden', C)
    #print(h)

    #dm_aa = np.zeros_like(h)
    #dm_bb = np.zeros_like(h)
    #for i in range(cas_nel//2):
    #    dm_aa[i,i] = 1
    #    dm_bb[i,i] = 1
    return h,g,C
    # }}}
