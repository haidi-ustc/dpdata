import numpy as np
from pprint import pprint
from dpdata.periodic_table import Element

def system_info (lines, type_idx_zero = False) :
    atom_types = []
    atom_names = []
    cell = []
    atom_numbs = None

    scf_miter = None
    scf_phi_miter = None
    md_scf_miter = None
    md_scf_phi_miter = None

    _atom_names = []
  
    for ii in lines: 
        if 'Super cell' in ii : 
            _ii=[float(k)*0.52917721067 for k in ii.split('=')[-1].split()]
            cell=[[_ii[0],0,0],[0,_ii[1],0],[0,0,_ii[2]]]
        elif ii.strip().startswith('SCF Outer MaxIter'):
            scf_miter = int(ii.split('=')[-1])
           # print(scf_miter,'1')
        elif ii.strip().startswith('SCF Phi MaxIter') :
            scf_phi_miter = int(ii.split('=')[-1])
           # print(scf_phi_miter,'2')
        elif ii.strip().startswith('MD SCF Phi MaxIter'):
            md_scf_phi_miter = int(ii.split('=')[-1])
           # print(md_scf_phi_miter,'3')
        elif ii.strip().startswith('MD SCF Outer MaxIter') :
            md_scf_miter = int(ii.split('=')[-1])
           # print(md_scf_miter,'4')
        elif 'NumAtom' in ii :
            _atom_numbs = int(ii.split('=')[-1])
            assert (_atom_numbs is not None) 
        elif 'Type =' in ii:
            _atom_names.append(Element.from_Z(int(ii.split()[2])).symbol)
        #elif ii.strip().startswith("Hybrid ACE"):
        #     hydrid=
        else:
           pass

    atom_names=[]
    for ii in _atom_names:
        if not ii in atom_names:
           atom_names.append(ii)
    
    assert (_atom_numbs == len(_atom_names)), "inconsistent numb atoms in pwdft output"
    atom_numbs =[_atom_names.count(ii) for ii in atom_names] 
    assert(atom_numbs is not None), "cannot find ion type info in pwddft output"
    for idx,ii in enumerate(atom_numbs) :
        for jj in range(ii) :
            if type_idx_zero :
                atom_types.append(idx)
            else :
                atom_types.append(idx+1)

    if md_scf_miter and  md_scf_phi_miter:
       return [cell, atom_numbs, atom_names, atom_types,  scf_miter, scf_phi_miter, md_scf_miter, md_scf_phi_miter]
    else:
       return [cell, atom_numbs, atom_names, atom_types,  scf_miter, scf_phi_miter]


def get_pwdft_block(fp) :
    blk = []
    for ii in fp :
        if not ii :
            return blk
        blk.append(ii.rstrip('\n'))
        if '! Total time for the SCF iteration' in ii:
            return blk
    return blk

# we assume that the force is printed ...
def get_frames (fname, begin = 0, step = 1, hydrid=True) :
    fp = open(fname)
    blk = get_pwdft_block(fp)
    ret = system_info(blk, type_idx_zero = True)
    if len(ret) == 8:
       md=True
    else:
       md=False

    cell, atom_numbs, atom_names, atom_types =ret[0],ret[1],ret[2],ret[3]
    ntot = sum(atom_numbs)

    all_coords = []
    all_cells = []
    all_energies = []
    all_forces = []
    all_virials = []    

    cc = 0
    while len(blk) > 0 :
        #print("-"*20+str(cc)+'-'*20)
        if cc >= begin and (cc - begin) % step == 0 :
            if cc==0:
                coord, _cell, energy, force, virial, is_converge = analyze_block(blk, ret, md, first_blk=True, hydrid=hydrid)
            else:
                coord, _cell, energy, force, virial, is_converge = analyze_block(blk, ret, md, first_blk=False, hydrid=hydrid)
            #pprint(coord)
            if is_converge : 
                if len(coord) == 0:
                    break
                all_coords.append(coord)

                if _cell:
                   all_cells.append(_cell)
                else:
                   all_cells.append(cell)

                all_energies.append(energy)
                all_forces.append(force)
                if virial is not None :
                    all_virials.append(virial)
        blk = get_pwdft_block(fp)
        cc += 1
        
    if len(all_virials) == 0 :
        all_virials = None
    else :
        all_virials = np.array(all_virials)
    fp.close()
    return atom_names, atom_numbs, np.array(atom_types), np.array(all_cells), np.array(all_coords), np.array(all_energies), np.array(all_forces), all_virials


def analyze_block(lines, ret, md, first_blk=False,hydrid=True) :
    coord = []
    cell = []
    energy = None
    force = []
    virial = None
    atom_names=[]
    _atom_names=[]
    # set convege condition
    if md:
       if first_blk:
          phi_miter = ret[5]
          scf_miter = ret[4]
       else: 
          phi_miter=ret[7]
          scf_miter=ret[6]
    else:
       phi_miter=ret[5]
       scf_miter=ret[4]
   
    #parsing lines
    is_converge = True
    is_converge_scf = True
    is_converge_phi = True
    for idx,ii in enumerate(lines) :
             
        if ii.strip().startswith('SCF iteration #'):
            scf_index = int(ii.split()[-1])
            if scf_index >= scf_miter:
                is_converge_scf = False
            
        if ii.strip().startswith('Phi iteration #'):
            phi_index = int(ii.split()[-1])
            if phi_index >= phi_miter:
                is_converge_phi = False

        elif '! Etot' in ii:
            if 'au' in ii:
                energy = float(ii.split()[3])*27.211386020632837
            else:
                energy = float(ii.split()[3])

        elif ii.strip().startswith('Type =') and 'Position' in ii:
            coord.append([float(i)*0.52917721067 for i in ii.split('=')[-1].split()])
        elif ii.strip().startswith('Type =') and 'Pos' in ii and 'Vel' in ii:
            coord.append([float(i)*0.52917721067 for i in ii.split()[5:8]])

        elif ii.startswith('atom') and 'force' in ii:
           force.append([float(i)*27.211386020632837/0.52917721067 for i in ii.split()[-3:]])

    if first_blk:
       if hydrid:
          if is_converge_scf and is_converge_phi:
             is_converge = True
          elif is_converge_scf and not is_converge_phi:
             is_converge = False
          elif not is_converge_scf and  is_converge_phi:
             is_converge = True
          else:
             is_converge = False
       else:
          if is_converge_scf:
             is_converge = True
    else:
       if hydrid:
          is_converge = True if is_converge_phi else False
       else:
          is_converge = True if is_converge_scf else False
    
    if not energy:
       is_converge = False

    if energy:
       assert((force is not None) and len(coord) > 0 )

    return coord, cell, energy, force, virial, is_converge

if __name__=='__main__':
  import sys
  ret=get_frames (sys.argv[1], begin = 0, step = 1)
  #print(ret)
