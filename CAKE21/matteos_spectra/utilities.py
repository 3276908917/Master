import sys
import os
import baccoemu as bacco
import json
import numpy as np
import copy
from scipy.special import gamma, gammainc
from scipy.interpolate import interp1d

def ask_continue(text=None, abort=True):
    if (text!=None):
        print(text)
    ans = input('Do you want to continue anyway [y/n]? ')
    if ans=='y':
        return True
    elif ans=='n':
        if abort:
            sys.exit()
        return False
    else:
        print('Please insert only y or n')
        return ask_continue(abort=abort)

def mask_zeros(Pk, masking_var='pk', to_mask=['k', 'pk', 'shotnoise']):
    if type(Pk) == dict:
        mask = (Pk[masking_var]!=0)
        for masked_qty in to_mask:
            Pk[masked_qty] = Pk[masked_qty][mask]
    elif type(Pk) == tuple:
        mask = (Pk[0]!=0)
        result = [Pk[0][mask]]
        for x in Pk[1:]:
            result.append(x[mask])
        return result
    else:
        print('ERROR: Please use as an argument a tuple () or a dict {}')

def safe_diff(X, Y):
    diff = np.zeros(len(X))
    for i in range(len(diff)):
        if Y[i]==0:
            diff[i]=0
        else:
            diff[i]=(X[i]-Y[i])/Y[i]
    return diff

def myfind_BinA(A, B):
    return (A[:, None] == B).argmax(axis=0)

def read_sim_and_get_Pk(sim_idx, snapnum, kmin=0.01, kmax=1.6, nbins=200, t_sim_idx=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        ngrid=512, space='all', sigma12_corr=False, save_mem=True, pair_idx=['A', 'B']):
    #Check that you are asking for something already implemented
    assert space in ['all', 'real', 'redshift', 'scaled', 'vel', 'vel-real']
    #Set Cosmology
    cosmo_params = cosmo_dict['Columbus_%d_%d' %(sim_idx, snapnum)]
    cosmo = bacco.Cosmology(**cosmo_params)
    h = cosmo.pars['hubble']
    #Check k range for Pk (in Mpc units)
    npart = 1500
    boxsize = 1000
    ngrid = ngrid
    if kmin is not None:
        kmin /= h
        if kmin < 2*np.pi/boxsize:
            print('WARNING: the given kmin [%f] is lower than the fundamental mode [%f]' %(kmin, 2*np.pi/boxsize))
    if kmax is not None:
        kmax /= h
        if kmax > kmin*max(ngrid, npart):
            print('WARNING: the given kmax [%f] is higher than the maximum possible mode [%f]' %(kmax, kmin*max(ngrid, npart)))
    #Read simulation
    for L in pair_idx:
        params = {'basedir': '/ptmp/anruiz/Columbus/Columbus_%d_%s/snapshots' %(sim_idx, L),
                  'snapnum': snapnum,
                  'dm_file': 'snapdir_00%d/snapshot_Columbus_%d_%s_00%d' %(snapnum, sim_idx, L, snapnum),
                  'sim_cosmology': cosmo,
                  'sim_format': 'gadget4'}
        sim = bacco.Simulation(**params)
        #Get Pk (in h/Mpc units)
        if space in ['real', 'all']:
            Pk = sim.get_Power(kmin=kmin, kmax=kmax, nbins=nbins, ngrid=ngrid)
            to_save = np.column_stack((Pk['k'], Pk['pk']))
            np.savetxt('Pks/Pk_Columbus_%d_%s_00%d.txt' %(sim_idx, L, snapnum), to_save)
        if space in ['redshift', 'all']:
            Pk = sim.get_Power(kmin=kmin, kmax=kmax, nbins=nbins, ngrid=ngrid, zspace=True)
            to_save = np.column_stack((Pk['k'], Pk['pk']))
            np.savetxt('Pks/Pk-zspace_Columbus_%d_%s_00%d.txt' %(sim_idx, L, snapnum), to_save)
        if space in ['vel', 'all']:
            Pk = sim.get_thetaPower(kmin=kmin, kmax=kmax, nbins=nbins, ngrid=ngrid, 
                                log_binning = True, interlacing = True, deposit_method = 'tsc')
            to_save = np.column_stack((Pk['k'], Pk['pk']))
            np.savetxt('Pks/test_Pk-vel_Columbus_%d_%s_00%d.txt' %(sim_idx, L, snapnum), to_save)
        if sigma12_corr and (space == 'scaled'):
            for i in t_sim_idx:
                target_cosmo = bacco.Cosmology(**cosmo_dict['Columbus_%d_%d' %(i, snapnum)])
                Pk = sim.get_Power(kmin=kmin, kmax=kmax, nbins=nbins, ngrid=ngrid, 
                                   zspace=True, sigma12_corr=True, target_cosmo=target_cosmo)
                to_save = np.column_stack((Pk['k'], Pk['pk']))
                np.savetxt('Pks/Pk-zspace_rescaled_Columbus_%d_%s_00%d.txt' %(i, L, snapnum), to_save)
        if save_mem:
            del sim
        else:
            return sim
        
def read_Pk(sim_indices, snapnum, pair_idx=None, space='real', path = 'Pks'):
    if (not hasattr(sim_indices, '__iter__')):
        sim_indices = [sim_indices]
    assert space in ['real', 'zspace', 'rescaled']
    if space == 'real':
        label = ''
    elif space == 'zspace':
        label = '-zspace'
    else:
        label = '-zspace_rescaled'
    k_arr = []
    Pk_arr = []
    for sim_idx in sim_indices:
        if space != 'rescaled':
            h = cosmo_dict['Columbus_%d_%d' %(sim_idx, snapnum)]['hubble']
        else:
            h = cosmo_dict['Columbus_0_0']['hubble']
        kA, PkA = np.loadtxt(path+'/Pk'+label+'_Columbus_%d_A_00%d.txt' %(sim_idx, snapnum), unpack=True)
        kB, PkB = np.loadtxt(path+'/Pk'+label+'_Columbus_%d_B_00%d.txt' %(sim_idx, snapnum), unpack=True)
        if np.sum(kA-kB) > 1e-8:
            print('WARNING: There is something wrong, the k arrays are not identical')
        Pk = (PkA+PkB)/2.
        #Mask zeros
        kA = kA[Pk!=0]
        Pk = Pk[Pk!=0]
        #In Mpc units!
        k_arr.append(kA*h)
        Pk_arr.append(Pk/h**3)
    return k_arr, Pk_arr
    

def make_cosmo_suite(fname='/home/lfinkbei/Documents/Master/' + \
    'CAKE21/matteos_spectra/cosmology_Columbus.dat', cosmo_suite_name='my_cosmo_suite.py', use_sigma8=True):
    f = open(fname)
    lines = f.readlines()
    f.close()
    labels = (lines.pop(0)).split()
    
    """
    for n in range(len(lines)):
        for i in range(len(labels)):
            if 'Columbus' in lines[n].split()[i]:
                print(lines[n].split()[i])
            else:
                print(float(lines[n].split()[i]))
    """
    
    def to_float(string):
        if string == '-':
            return np.nan
        return float(string)
    
    cosmo_suite = [{labels[i]: (lines[n].split()[i] if 'Columbus' in lines[n].split()[i]
                 else to_float(lines[n].split()[i])) for i in range(len(labels))} for n in range(len(lines))]
    cosmo_dict = {}
    for cosmo in cosmo_suite:
        for snapnum in range(5):
            cosmo_params ={}
            cosmo_params['omega_cdm'] = cosmo['OmC']
            cosmo_params['omega_de'] = cosmo['OmL']
            cosmo_params['omega_baryon'] = cosmo['OmB']
            cosmo_params['hubble'] = cosmo['h']
            cosmo_params['ns'] = cosmo['n_s']
            if use_sigma8:
                cosmo_params['A_s'] = None
                cosmo_params['sigma8'] = cosmo['sigma8']
            else:
                cosmo_params['A_s'] = cosmo['A_s']
                cosmo_params['sigma8'] = None               
            cosmo_params['expfactor'] = 1/(1+cosmo['z(%d)' %snapnum]) #Warning!! May differ from the actual expfactor (check header)
            cosmo_params['tau'] = 0.0952 #
            w0 = cosmo['w0']
            wa = cosmo['wa']
            if w0!=-1 or (wa!=0):
                cosmo_params['de_model'] = 'w0wa'
                cosmo_params['w0'] = w0
                cosmo_params['wa'] = 0.0 if np.isnan(wa) else wa
            cosmo_dict[cosmo['Name']+'_%d' %snapnum] = cosmo_params
    return cosmo_dict

def mem_from_npart(n, u='gb', per_CPU=True):
    PartAllocFactor = 2.
    TreeAllocFactor = 0.7
    PMGRID = 2*n
    m1 = PartAllocFactor*(68+TreeAllocFactor*64)
    m2 = PMGRID**3 * 12 - 16
    memtot = m1*(n**3) + m2
    cf = {'b':1., 'kb':1024., 'mb':(1024.)**2, 'gb':(1024.)**3, 'tb':(1024.)**4}
    if per_CPU:
        nCPU = 48*int(np.ceil(n/768))
        return memtot/(nCPU*cf[u])
    else:
        return memtot/cf[u]

### Things for free when importing utilities ###
cosmo_dict = make_cosmo_suite()
