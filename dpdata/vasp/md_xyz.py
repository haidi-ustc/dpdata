#!/usr/bin/python3 

import numpy as np
from dpdata.periodic_table import Element

def from_system_data(system,  skip_zeros = True) :
    """
192
MD step # 1
     1         5.72429         9.91148         7.79881
     1         10.5529       -0.585524         9.39996
     1         1.10042         7.11083         10.5041

    """
    print(system)
    ele=[]
    for name,num in zip(system.get_atom_names(),system.get_atom_numbs()):
        ele.extend([name]*num)
    atomic_numb=np.array([Element(el).Z for el in ele])
    ret = ''
    for ii in range(len(system)): 
        sys=system.sub_system([ii])
        ret += '%d' % sum(sys.get_atom_numbs()) 
        ret += '\n'
        ret +="MD step # %d"% (ii+1)
        ret += '\n'
        atype = sys['atom_types']
        posis = sys.data['coords'][0]
        # atype_idx = [[idx,tt] for idx,tt in enumerate(atype)]
        # sort_idx = np.argsort(atype, kind = 'mergesort')
        sort_idx = np.lexsort((np.arange(len(atype)), atype))
        atype = atype[sort_idx]
        posis = posis[sort_idx]
        atm_z = atomic_numb[sort_idx]
        posi_list = []
        for Z, ii in  zip(atm_z,posis) :
            posi_list.append('%d  %15.10f %15.10f %15.10f' % \
                             (Z, ii[0], ii[1], ii[2])
            )
        posi_list.append('')
        ret += '\n'.join(posi_list)
    return ret
