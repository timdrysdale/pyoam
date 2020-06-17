# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:44:43 2015

@author: tdd
"""
'''
Here are the first few lines from a CST farfield result file

Theta [deg.]  Phi   [deg.]  Abs(E   )[V/m   ]   Abs(Theta)[V/m   ]  Phase(Theta)[deg.]  Abs(Phi  )[V/m   ]  Phase(Phi  )[deg.]  Ax.Ratio[      ]    
------------------------------------------------------------------------------------------------------------------------------------------------------
-180.000         -90.000           7.429e-001          7.429e-001             221.796          4.669e-010             107.878          3.162e+002     
-175.000         -90.000           7.041e-001          7.041e-001              29.998          4.654e-006             197.307          3.162e+002     


And here are the first few lines from an far field source file in ASCII, which seem esaier to produce on demand
from the GUI


// CST Farfield Source File
 
// Version:
3.0 

// Data Type
Farfield 

// #Frequencies
1 

// Position
0.000000e+000 0.000000e+000 3.098270e-003 

// zAxis
0.000000e+000 0.000000e+000 1.000000e+000 

// xAxis
1.000000e+000 0.000000e+000 0.000000e+000 

// Radiated/Accepted/Stimulated Power , Frequency 
4.915340e-001 
4.989917e-001 
5.000000e-001 
6.000000e+010 


// >> Total #phi samples, total #theta samples
361 181

// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi): 
   0.000     0.000   1.00598219e-009  -1.84997406e-009 -4.05856590e+001   1.64062271e+001
   0.000     1.000  -1.65680267e-005   1.83749216e-005 -4.01599121e+001   1.64335918e+001

'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import sys
import os.path

import scipy.interpolate as sci 


def read_CST_ffs_ASCII(filename):
    
    # Open file
   
    f = open(filename, 'r')
    header= {}
    
    # Read and ignore header lines
    for i in np.arange(31):     
        header = np.append(header,f.readline())
    

    nums = header[29].strip()
    columns = nums.split()
    numTheta = int(columns[1]) #checked
    numPhi = int(columns[0])        
    
    #header3 = f.readline()
    Phi = np.array([])  ##Note that these are reversed compared to the other file
    Theta = np.array([]) 
    RET = np.array([])
    IET = np.array([])
    REP = np.array([])
    IEP = np.array([])
    
    # Loop over lines and extract variables of interest
    for line in f:
        line = line.strip()
        columns = line.split()
        Phi     = np.append(Phi,float(columns[0]))
        Theta   = np.append(Theta,float(columns[1]))
        RET = np.append(RET,float(columns[2]))
        IET = np.append(IET,float(columns[3]))
        REP = np.append(REP,float(columns[4]))
        IEP = np.append(IEP,float(columns[5]))
        

    f.close()
    

    
    return(Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi)


def subsample_CST_ffs_ASCII(filename,max_theta):
    
    # Open file
    fo_name = '%s-%d-%s'%('sub', max_theta, filename) 
    fo = open(fo_name,'w')   
   
    f = open(filename, 'r')
    header= {}
    
    # Read and ignore header lines
    for i in np.arange(31):     
        header = np.append(header,f.readline())
    

    nums = header[29].strip()
    columns = nums.split()
    numTheta = int(columns[1]) #checked
    numPhi = int(columns[0])        
    
    #header3 = f.readline()
    Phi = np.array([])  ##Note that these are reversed compared to the other file
    Theta = np.array([]) 
    RET = np.array([])
    IET = np.array([])
    REP = np.array([])
    IEP = np.array([])
    
    # Loop over lines and extract variables of interest
    for line in f:
        line = line.strip()
        columns = line.split()
        this_phi =  float(columns[0])
        this_theta = float(columns[1])
        if this_theta <= max_theta:
            Phi     = np.append(Phi,float(columns[0]))
            Theta   = np.append(Theta,float(columns[1]))
            RET = np.append(RET,float(columns[2]))
            IET = np.append(IET,float(columns[3]))
            REP = np.append(REP,float(columns[4]))
            IEP = np.append(IEP,float(columns[5]))
         

    f.close()
    
        
    header[29] = '%d %d\n'%(len(set(Phi)),len(set(Theta)))
    
    for i in np.arange(32):     
        fo.write('%s'%str(header[i]))
        
    for (p,t,rt,it,rp,ip) in zip(Phi,Theta,RET,IET,REP,IEP):
        fo.write('%f %f %e %e %e %e\n'%(p,t,rt,it,rp,ip))
    
    fo.close()    
    
    return(Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi)



def downsample_CST_ffs_ASCII(filename,N):
    
    # Open file
    fo_name = '%s%s'%('small',filename) 
    fo = open(fo_name,'w')   
   
    f = open(filename, 'r')
    header= {}
    
    # Read and ignore header lines
    for i in np.arange(31):     
        header = np.append(header,f.readline())
    

    nums = header[29].strip()
    columns = nums.split()
    numTheta = int(columns[1]) #checked
    numPhi = int(columns[0])        
    
    #header3 = f.readline()
    Phi = np.array([])  ##Note that these are reversed compared to the other file
    Theta = np.array([]) 
    RET = np.array([])
    IET = np.array([])
    REP = np.array([])
    IEP = np.array([])
    
    # Loop over lines and extract variables of interest
    for line in f:
        line = line.strip()
        columns = line.split()
        this_phi =  float(columns[0])
        this_theta = float(columns[1])
        if this_phi%N == 0:
            if this_theta%N == 0:
                Phi     = np.append(Phi,float(columns[0]))
                Theta   = np.append(Theta,float(columns[1]))
                RET = np.append(RET,float(columns[2]))
                IET = np.append(IET,float(columns[3]))
                REP = np.append(REP,float(columns[4]))
                IEP = np.append(IEP,float(columns[5]))
         

    f.close()
    
        
    header[29] = '%d %d\n'%(len(set(Phi)),len(set(Theta)))
    
    for i in np.arange(32):     
        fo.write('%s'%str(header[i]))
        
    for (p,t,rt,it,rp,ip) in zip(Phi,Theta,RET,IET,REP,IEP):
        fo.write('%f %f %e %e %e %e\n'%(p,t,rt,it,rp,ip))
    
    fo.close()    
    
    return(Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi)


def read_CST_farfield(filename):
    #(Theta,Phi,AbsE,AbsTheta,PhaseTheta,AbsPhi,PhasePhi,AxRatio)=read_CST_farfield(filename)
    # Open file
    f = open(filename, 'r')
    
    # Read and ignore header lines
    header1 = f.readline()
    header2 = f.readline()
    #header3 = f.readline()
    Theta = np.array([]) 
    Phi = np.array([]) 
    AbsE = np.array([])
    AbsTheta = np.array([])   
    PhaseTheta = np.array([])
    AbsPhi = np.array([])
    PhasePhi = np.array([])
    AxRatio = np.array([])
    
    # Loop over lines and extract variables of interest
    for line in f:
        line = line.strip()
        columns = line.split()
        Theta   = np.append(Theta,float(columns[0]))
        Phi     = np.append(Phi,float(columns[1]))
        AbsE    = np.append(AbsE,float(columns[2]) )       
        AbsTheta    = np.append(AbsTheta,float(columns[3]))
        PhaseTheta  = np.append(PhaseTheta,float(columns[4]))
        AbsPhi      = np.append(AbsPhi,float(columns[5]))
        PhasePhi    = np.append(PhasePhi,float(columns[6])) 
        AxRatio     = np.append(AxRatio,float(columns[7]))
        
    f.close()
    
    return(Theta,Phi,AbsE,AbsTheta,PhaseTheta,AbsPhi,PhasePhi,AxRatio)

def plot_AbsE(Theta,Phi,AbsE):
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    ThetaRad = Theta*np.pi/180.
    PhiRad = Phi*np.pi/180.

    radius = AbsE / np.max(AbsE)
    
    zs = radius * np.sin(ThetaRad)      
    rxy  = radius * np.cos(ThetaRad)
    xs = rxy * np.cos(PhiRad)
    ys = rxy * np.sin(PhiRad)    
    '''
    ax.scatter(xs, ys, zs, c='r', marker='o')
     
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    '''

    surf = ax.plot_trisurf(xs, ys, zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()
    
    plt.show() # or:
    # fig.savefig('3D.png')
        
    
    
def cartesian_coords(Theta=0,Phi=0,radius=1):
    '''
    Given Theta and Phi in radians, get Cartesian location
    #TESTED 20/8/15
    '''
    ThetaRad = Theta*np.pi/180.
    PhiRad = Phi*np.pi/180. 
    
    z = radius * np.cos(ThetaRad)      
    rxy  = radius * np.sin(ThetaRad)
    x = rxy * np.cos(PhiRad)
    y = rxy * np.sin(PhiRad)
    
    return(x,y,z)    
    
def cartesian_weightings(Theta=0,Phi=0):
 
    '''
    Calculate coefficients to allow us to convert Ephi and Etheta into Ex,Ey,Ez    
    Given at some point 0 ... 
    A0 = AbsTheta*np.exp(1j*w*t+PhaseTheta*np.pi/180.)
    B0 = AbsPhi*np.exp(ij*w*t+PhasePhi*np.pi/180.)

    Ex0 = A0*tx + B0*px
    Ey0 = A0*ty + B0*py
    Ez0 = A0*tz + B0*pz
    
    such that (necessary but not sufficient condition) 
    1 = (px**2+py**2+pz**2)**0.5
    1 = (tx**2+ty**2+pz**2)**0.5 
    
    Theta is the angle from the Z-axis
    
    When Theta=0, and Phi=0, E_Theta points in +x dir
    When Theta=90 and Phi=0, E_phi points in the +y dir
    
    Theta   Phi     E_theta   E_phi
    0       0       +x        +y    
    pi/2    0       -z        +y
    pi      0       -x        +y
    3pi/2   0       +z        +y
    
    0       pi/2    +y        -x    
    pi/2    pi/2    -z        -x
    pi      pi/2    -y        -x
    3pi/2   pi/2    +z        -x

    0       pi      -x        -y    
    pi/2    pi      -z        -y
    pi      pi      -x        -y
    3pi/2   pi      +z        -y  
       
    0       3pi/2   -y        +x    
    pi/2    3pi/2   -z        +x
    pi      3pi/2   +y        +x
    3pi/2   3pi/2   +z        +x

    Tested 20 Aug 2015 
    '''    
    ThetaRad = Theta*np.pi/180.
    PhiRad = Phi*np.pi/180.
    
    tx = np.cos(ThetaRad)*np.cos(PhiRad) 
    ty = np.cos(ThetaRad)*np.sin(PhiRad)
    tz = np.sin(-ThetaRad)
    
    px = np.sin(-PhiRad)
    py = np.cos(PhiRad)
    pz = 0
    
    
    return(tx,ty,tz,px,py,pz)
    
def test_plotting():
    '''
    This will plot the Abs and Phase of Ephi and Etheta from a CST ASCII FFS
    file, if the images don't already exist, and it will load the data from an 
    npz if it can, but if not, from the original file.
    This was for efficiency during prototyping
    '''
    outfile = 'ffs.npz'    
    
    pngfileAbs = 'AbsE.png'   
    pngfilePhase = 'PhaseE.png'
    
    if os.path.isfile(pngfileAbs) == False: 
        if os.path.isfile(outfile) == False: 
            filename = 'plasmon_circular_rings_final_design_shifted_7rings_0.0mm_FARFIELDS1deg.txt'
            print('loading data from %s'%filename)
            (Phi,Theta,RET,IET,REP,IEP) = read_CST_ffs_ASCII(filename)
            np.savez(outfile,Phi=Phi,Theta=Theta,RET=RET,IET=IET,REP=REP,IEP=IEP)    
        else: 
            print('loading data from %s'%outfile)        
            npzfile = np.load(outfile)
            Phi = npzfile['Phi']
            Theta = npzfile['Theta']
            RET = npzfile['RET']
            IET = npzfile['IET']
            REP = npzfile['REP']
            IEP = npzfile['IEP']    
            
            
        AbsE = (RET**2+IET**2 + REP**2 + IEP**2)**0.5    
        plot_AbsE(Theta,Phi,20*np.log10(AbsE))
        plt.savefig(pngfileAbs,dpi=300)
    
        PhaseE = np.arctan2(IET,RET) + np.arctan2(IEP,REP)
        plot_AbsE(Theta,Phi,PhaseE)
        plt.savefig(pngfilePhase)    

def E_cartesian(Theta,Phi,AbsTheta,PhaseTheta,AbsPhi,PhasePhi):
        
    (x,y,z) = cartesian_coords(Theta,Phi,radius)
    
    print('E_cartesian not finished')
    
    return(x,y,z,Ex,Ey,Ez) 
    
def test_cartesian():
    print('testing cartesian coords, plotting the weightings as vectors')
    print('should give the directions of Ephi and Etheta')
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')    
    ax.set_xlim3d(-5,5)
    ax.set_ylim3d(-5,5)
    ax.set_zlim3d(-5,5)
  
    phislist = [0,0,0,0,0,0,0,0,0,0,0,0,
                90,90,90,90,90,90,90,90,90,90,
                45,45,45,45,45,45,45,45,45,45,
                135,135,135,135,135,135,135,135,135,135]
                
    thetaslist = [0,30,60,90,120,150,180,210,240,270,300,330,
                  30,60,90,120,150,210,240,270,300,330,
                  30,60,90,120,150,210,240,270,300,330,
                  30,60,90,120,150,210,240,270,300,330]    
    
    radius = 5
    
    for (phi,theta) in zip(phislist,thetaslist):
        
        [tx,ty,tz,px,py,pz] = cartesian_weightings(Phi=phi,Theta=theta)
   
        (x,y,z) = cartesian_coords(Theta=theta,Phi=phi,radius=radius)
        
        ax.plot([x,x+tx],[y,y+ty],[z,z+tz],'r')
        ax.plot([x,x+px],[y,y+py],[z,z+pz],'b')

        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('ETheta in Red, Ephi in Blue')
    plt.show()    

def test_CST_farfield():
    
    (Theta,Phi,
     AbsE,
     AbsTheta,PhaseTheta,
     AbsPhi,PhasePhi,
     AxRatio)  = read_CST_farfield('farfield (f=60) [1].txt')
     
    plot_AbsE(Theta,Phi,AbsE)
    return()
    
def get_subset_ffs_data(verbose=False):    
    outfile = 'ffs_subset.npz'
    m = 1000
    if os.path.isfile(outfile) == False: 
        filename = 'plasmon_circular_rings_final_design_shifted_7rings_0.0mm_FARFIELDS1deg.txt'
        if verbose==True:
            print('loading data from %s'%filename)
        (Phi,Theta,RET,IET,REP,IEP) = read_CST_ffs_ASCII(filename)
        np.savez(outfile,Phi=Phi[0:m],Theta=Theta[0:m],RET=RET[0:m],IET=IET[0:m],REP=REP[0:m],IEP=IEP[0:m])    
    else: 
        if verbose==True:
            print('loading data from %s'%outfile)        
        npzfile = np.load(outfile)
        Phi = npzfile['Phi']
        Theta = npzfile['Theta']
        RET = npzfile['RET']
        IET = npzfile['IET']
        REP = npzfile['REP']
        IEP = npzfile['IEP']     
        return(Phi,Theta,RET,IET,REP,IEP)
        
def get_thetaphi(dx,dy,dz):
    '''
    now we go in reverse, given the offsets from the source to the far field
    point, where the far field point is dx further along the x axis than the 
    source...
    '''
    theta = np.arctan((dx**2+dy**2)**0.5/dz)
    phi = np.arctan2(dy,dx)
    
    #if theta < 0:
    #    theta = theta + 2.*np.pi
    if phi < 0:
        phi = phi + 2.* np.pi
 
    
    return(theta,phi)

def deg(rad):
    return(rad*180./np.pi)      

def rad(deg):
    return(deg*np.pi/180.)
    
def get_thetaphi_deg(dx,dy,dz):
    (theta,phi) = get_thetaphi(dx,dy,dz)
    return(deg(theta),deg(phi))   
     
def get_fullset_ffs_data(verbose=False):    
    outfile = 'ffs.npz'
    if os.path.isfile(outfile) == False: 
        filename = 'plasmon_circular_rings_final_design_shifted_7rings_0.0mm_FARFIELDS1deg.txt'
        if verbose==True:
            print('loading data from %s'%filename)
        (Phi,Theta,RET,IET,REP,IEP) = read_CST_ffs_ASCII(filename)
        np.savez(outfile,Phi=Phi,Theta=Theta,RET=RET,IET=IET,REP=REP,IEP=IEP)    
    else: 
        if verbose==True:
            print('loading data from %s'%outfile)        
        npzfile = np.load(outfile)
        Phi = npzfile['Phi']
        Theta = npzfile['Theta']
        RET = npzfile['RET']
        IET = npzfile['IET']
        REP = npzfile['REP']
        IEP = npzfile['IEP']     
        return(Phi,Theta,RET,IET,REP,IEP)   

def put_ffs_into_npz(filename, outfile, verbose = False):    
    
    if os.path.isfile(outfile) == False: 
        if verbose==True:
            print('loading data from %s'%filename)
        (Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi) = read_CST_ffs_ASCII(filename)
        np.savez(outfile,Phi=Phi,Theta=Theta,RET=RET,IET=IET,REP=REP,IEP=IEP,
                 numPhi=numPhi,numTheta=numTheta)    
    else: 
        print('outfile already exists')
    return()

def put_ffstxt_into_npz(filename, outfile, verbose = False):    
    
    if os.path.isfile(outfile) == False: 
        if verbose==True:
            print('loading data from %s'%filename)
        
        (Theta,Phi,AbsE,AbsTheta,PhaseTheta,AbsPhi,PhasePhi,AxRatio)=read_CST_farfield(filename)
        
        np.savez(outfile,Theta=Theta,Phi=Phi,AbsE=AbsE,AbsTheta=AbsTheta,PhaseTheta=PhaseTheta,AbsPhi=AbsPhi,PhasePhi=PhasePhi,AxRatio=AxRatio)    
    else: 
        print('outfile already exists')
    return()

def get_ffs_from_npz(outfile, verbose = False):    
    if verbose==True:
            print('loading data from %s'%outfile)        
    npzfile = np.load(outfile)
    Phi = npzfile['Phi']
    Theta = npzfile['Theta']
    RET = npzfile['RET']
    IET = npzfile['IET']
    REP = npzfile['REP']
    IEP = npzfile['IEP'] 
    numTheta = npzfile['numTheta']
    numPhi = npzfile['numPhi']    
    return(Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi)        

def get_ffstxt_from_npz(outfile, verbose = False):    
    if verbose==True:
            print('loading data from %s'%outfile)        
    npzfile = np.load(outfile)
    Phi = npzfile['Phi']
    Theta = npzfile['Theta']
    AbsE = npzfile['AbsE']
    AbsTheta = npzfile['AbsTheta']
    PhaseTheta = npzfile['PhaseTheta']
    AbsPhi = npzfile['AbsPhi']
    PhasePhi = npzfile['PhasePhi'] 
    AxRatio = npzfile['AxRatio']
        
    return(Theta,Phi,AbsE,AbsTheta,PhaseTheta,AbsPhi,PhasePhi,AxRatio) 
    
def farfield_projection(Phi=0,Theta=0,RET=0,IET=0,REP=0,IEP=0,
                        radius = 1, xmin=-1, xmax=1,ymin=-1,ymax=1,
                        xnum = 30, ynum = 30, wavelength = 3e8/60e9,
                        ffz = 2,fig_file_stub='farfield_projection',
                        outfile='farfield_projection.npz',
                        numTheta=181,numPhi=361):
    '''
    We can simply project the farfield rather than repeating N2FF transform
    this is because we know the source, and we know the value in any particular 
    angle. We should scale for distance changes though, so we still
    use a Green's function to propagate them.
    
    '''
    
    [tx,ty,tz,px,py,pz] = cartesian_weightings(Phi=Phi,Theta=Theta)
    (sx,sy,sz) = cartesian_coords(Theta=Theta,Phi=Phi,radius=radius) 
    
    ET = RET + 1j*IET
    EP = REP + 1j*IEP
   
    Ex = ET*tx + EP*px
    Ey = ET*ty + EP*py
    Ez = ET*tz + EP*pz
    
    xlist = np.linspace(xmin,xmax,xnum,endpoint=True)
    ylist = np.linspace(ymin,ymax,ynum,endpoint=True)
    
    (xmesh,ymesh) = np.meshgrid(xlist,ylist)
    
    xmeshlist = np.squeeze(np.reshape(xmesh,(1,-1)))     
    ymeshlist = np.squeeze(np.reshape(ymesh,(1,-1))) 
        
    jk = 2.0j*np.pi/wavelength
    
    '''
    Set up for interpolation
    '''
           
    ThetaBlock = np.reshape(Theta,(numPhi,numTheta))
    PhiBlock = np.reshape(Phi,(numPhi,numTheta))
    ExBlock =  np.reshape(Ex,(numPhi,numTheta))   
    EyBlock =  np.reshape(Ey,(numPhi,numTheta))    
    EzBlock =  np.reshape(Ez,(numPhi,numTheta))        
    
    '''
    Get rid of the lower half of the data, it uses too many spline coeffs
    '''       
    mT = np.ceil(numTheta/2)
    ThetaBlock = ThetaBlock[:,0:mT]
    PhiBlock = PhiBlock[:,0:mT]
    ExBlock = ExBlock[:,0:mT]
    EyBlock = EyBlock[:,0:mT]
    EzBlock = EzBlock[:,0:mT]
 
    ThetaList = ThetaBlock[0,:]
    PhiList   = PhiBlock[:,0]
    
    '''
    Fix the interpolation - limit calculation to subset of points that will be used?
    '''
    
    Ex2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(ExBlock))
    Ey2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(EyBlock))
    Ez2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(EzBlock))
    Ex2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(ExBlock))
    Ey2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(EyBlock))
    Ez2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(EzBlock))       
    
    '''
    Ex2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(ExBlock))
    Ey2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(EyBlock))
    Ez2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(EzBlock))
    Ex2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(ExBlock))
    Ey2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(EyBlock))
    Ez2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(EzBlock))       
    
    Ex2dR = sci.interp2d(PhiBlock,ThetaBlock,np.real(ExBlock),kind='cubic')
    Ey2dR = sci.interp2d(PhiBlock,ThetaBlock,np.real(EyBlock),kind='cubic')
    Ez2dR = sci.interp2d(PhiBlock,ThetaBlock,np.real(EzBlock),kind='cubic')
    Ex2dI = sci.interp2d(PhiBlock,ThetaBlock,np.imag(ExBlock),kind='cubic')
    Ey2dI = sci.interp2d(PhiBlock,ThetaBlock,np.imag(EyBlock),kind='cubic')
    Ez2dI = sci.interp2d(PhiBlock,ThetaBlock,np.imag(EzBlock),kind='cubic')   
    '''    
     
    Exff = np.zeros((xnum*ynum),dtype='complex128')
    Eyff = np.zeros((xnum*ynum),dtype='complex128')
    Ezff = np.zeros((xnum*ynum),dtype='complex128')
     
    xcount = -1
    for (ffx,ffy) in zip(xmeshlist,ymeshlist):
       xcount = xcount + 1

       dr = (ffx**2+ffy**2+ffz**2)**0.5  - radius
       prop = 1 #np.exp(jk*dr)#/dr
       (theta,phi) = get_thetaphi_deg(ffx,ffy,ffz)

       Exff[xcount] = (complex(Ex2dR(phi,theta))+1j*complex(Ex2dI(phi,theta)))*prop
       Eyff[xcount] = (complex(Ey2dR(phi,theta))+1j*complex(Ey2dI(phi,theta)))*prop
       Ezff[xcount] = (complex(Ez2dR(phi,theta))+1j*complex(Ez2dI(phi,theta)))*prop
           
    
    Eabs = np.abs((Exff**2 + Eyff**2 + Ezff**2)**0.5)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    ax.scatter(xmeshlist,ymeshlist,Eabs)
    surf = ax.plot_trisurf(xmeshlist, ymeshlist, Eabs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    ax.view_init(elev=90., azim=0) 
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))    
    plt.savefig('%s_abs.png'%fig_file_stub,dpi=300)
    plt.show()
    
    EPhase = np.angle((Exff + Eyff + Ezff),deg=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    surf = ax.plot_trisurf(xmeshlist, ymeshlist, EPhase, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.view_init(elev=90., azim=0)    
    plt.savefig('%s_phase.png'%fig_file_stub,dpi=300)
    plt.show()
    
    np.savez(outfile,Exff=Exff,Eyff=Eyff,Ezff=Ezff,xmeshlist=xmeshlist,
             ymeshlist=ymeshlist,Eabs=Eabs,EPhase=EPhase)
    
    return()
    
def farfield_projection_phi(Phi=0,Theta=0,RET=0,IET=0,REP=0,IEP=0,
                        radius = 1, xmin=-1, xmax=1,ymin=-1,ymax=1,
                        xnum = 50, ynum = 50, wavelength = 3e8/60e9,
                        ffz = 2,fig_file_stub='farfield_projection',
                        outfile='farfield_projection.npz',
                        numTheta=181,numPhi=361):
    '''
    We can simply project the farfield rather than repeating N2FF transform
    this is because we know the source, and we know the value in any particular 
    angle. We should scale for distance changes though, so we still
    use a Green's function to propagate them.
    
    '''
    
    [tx,ty,tz,px,py,pz] = cartesian_weightings(Phi=Phi,Theta=Theta)
    (sx,sy,sz) = cartesian_coords(Theta=Theta,Phi=Phi,radius=radius) 
    
    ET = RET + 1j*IET
    EP = REP + 1j*IEP

    EPp  = np.reshape(REP,(numPhi,numTheta))
    EPa  = np.reshape(REP,(numPhi,numTheta))
   
    Ex =  EP*px
    Ey =  EP*py
    Ez =  EP*pz
    
    
        
    
    xlist = np.linspace(xmin,xmax,xnum,endpoint=True)
    ylist = np.linspace(ymin,ymax,ynum,endpoint=True)
    
    (xmesh,ymesh) = np.meshgrid(xlist,ylist)
    
    xmeshlist = np.squeeze(np.reshape(xmesh,(1,-1)))     
    ymeshlist = np.squeeze(np.reshape(ymesh,(1,-1))) 
        
    jk = 2.0j*np.pi/wavelength
    
    '''
    Set up for interpolation
    '''
           
    ThetaBlock = np.reshape(Theta,(numPhi,numTheta))
    PhiBlock = np.reshape(Phi,(numPhi,numTheta))
    ExBlock =  np.reshape(Ex,(numPhi,numTheta))   
    EyBlock =  np.reshape(Ey,(numPhi,numTheta))    
    EzBlock =  np.reshape(Ez,(numPhi,numTheta))        
    
    ThetaList = ThetaBlock[0,:]
    PhiList   = PhiBlock[:,0]        
    
    Ex2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(ExBlock))
    Ey2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(EyBlock))
    Ez2dR = sci.RectBivariateSpline(PhiList,ThetaList,np.real(EzBlock))
    Ex2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(ExBlock))
    Ey2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(EyBlock))
    Ez2dI = sci.RectBivariateSpline(PhiList,ThetaList,np.imag(EzBlock))   
        
     
    Exff = np.zeros((xnum*ynum),dtype='complex128')
    Eyff = np.zeros((xnum*ynum),dtype='complex128')
    Ezff = np.zeros((xnum*ynum),dtype='complex128')
     
    xcount = -1
    for (ffx,ffy) in zip(xmeshlist,ymeshlist):
       xcount = xcount + 1

       dr = (ffx**2+ffy**2+ffz**2)**0.5  - radius
       prop = np.exp(jk*dr)/dr
       (theta,phi) = get_thetaphi_deg(ffx,ffy,ffz)

       Exff[xcount] = (complex)(Ex2dR(phi,theta)+Ex2dI(phi,theta))*prop
       Eyff[xcount] = (complex)(Ey2dR(phi,theta)+Ey2dI(phi,theta))*prop
       Ezff[xcount] = (complex)(Ez2dR(phi,theta)+Ez2dI(phi,theta))*prop
           
    
    Eabs = np.abs((Exff**2 + Eyff**2 + Ezff**2)**0.5)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    ax.scatter(xmeshlist,ymeshlist,Eabs)
    surf = ax.plot_trisurf(xmeshlist, ymeshlist, Eabs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))    
    plt.savefig('%s_abs.png'%fig_file_stub,dpi=300)
    plt.show()
    
    EPhase = np.angle((Exff + Eyff + Ezff),deg=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    surf = ax.plot_trisurf(xmeshlist, ymeshlist, EPhase, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))    
    plt.savefig('%s_phase.png'%fig_file_stub,dpi=300)
    plt.show()
    
    np.savez(outfile,Exff=Exff,Eyff=Eyff,Ezff=Ezff,xmeshlist=xmeshlist,
             ymeshlist=ymeshlist,Eabs=Eabs,EPhase=EPhase)
    
    return()    
    
def unwrap(phase):
    # naive test to see if we have radians or degrees
    degrees = False    
    if np.max(np.abs(phase)) > (2. * np.pi):
        degrees = True
    
    #choose whether to fix degrees or radian phase jumps
    if degrees == True:
        jump_fix = 360 
    else:
        jump_fix = 2. * np.pi
    
    #set detection threshhold
    jump_size = 0.8 * jump_fix
    
    #find jumps
    jump_locs = np.where(np.abs(np.diff(phase))>jump_size)    
    
    fixed_phase = phase + 0 #fool spyders optimisation software into copying the object

    #repair the jumps - CHECK FOR  FENCEPOST ERRORS!
    if np.size(jump_locs) > 0:    
        for this_jump in np.nditer(jump_locs):
            if this_jump < (np.size(phase)-1):
                N = this_jump + 1
                fixed_phase[N:] = fixed_phase[N:] + jump_fix * np.sign(np.diff(phase)[N])
        
    return(fixed_phase)  

        
if __name__ == "__main__":

    #test_plotting()
    #test_cartesian()
    '''
    filename = 'TDD_OAM_test_bed_03_farfield.txt'
    outfile = 'TDD_OAM_test_bed_03_farfield.npz'
    ffoutfile='TDD_OAM_TB3_FF.npz'
    verbose = True

    put_ffs_into_npz(filename, outfile, verbose=verbose)  
    
    (Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi) = get_ffs_from_npz(outfile)  
    
    if os.path.isfile(ffoutfile) == False: 
        farfield_projection(Phi=Phi,Theta=Theta,RET=RET,IET=IET,REP=REP,IEP=IEP,
                        radius = 1, xmin=-2, xmax=2,ymin=-2,ymax=2,
                        xnum = 120, ynum = 120, wavelength = 3e8/60e9,
                        ffz = 2,fig_file_stub='TDD_OAM_TB3_FF',
                        outfile=ffoutfile,
                        numPhi=numPhi,numTheta=numTheta)    
    
    ff = np.load(ffoutfile)
    
    #x = np.reshape(ff['xmeshlist'],(numPhi,numTheta))
    #y = np.reshape(ff['ymeshlist'],(numPhi,numTheta))
    Eabs = np.reshape(ff['Eabs'],(120,120))
    plt.imshow(Eabs)
    plt.savefig('TDD_OAM_TBS_Eabs.png',dpi=300)
    plt.show()
    
    EPhase = np.reshape(ff['EPhase'],(120,120))
    plt.imshow(EPhase)
    plt.savefig('TDD_OAM_TBS_EPhase.png',dpi=300)
    plt.show()  
    '''
    
    filename = 'OAM_attempt_09_farfield.txt'
    outfile = 'OAM_attempt_09_farfield_0.3_phi.npz'
    ffoutfile='OAM_attempt_09_farfield_FF_0.3_phi.npz'
    verbose = True

    put_ffs_into_npz(filename, outfile, verbose=verbose)  
    
    (Phi,Theta,RET,IET,REP,IEP,numTheta,numPhi) = get_ffs_from_npz(outfile)  
    
    if os.path.isfile(ffoutfile) == False: 
        farfield_projection_phi(Phi=Phi,Theta=Theta,RET=RET,IET=IET,REP=REP,IEP=IEP,
                        radius = 1, xmin=-0.3, xmax=0.3,ymin=-0.3,ymax=0.3,
                        xnum = 120, ynum = 120, wavelength = 3e8/60e9,
                        ffz = 2,fig_file_stub='OAM_09_FF',
                        outfile=ffoutfile,
                        numPhi=numPhi,numTheta=numTheta)    
    
    ff = np.load(ffoutfile)
    
    Exff = np.reshape(ff['Exff'],(120,120))
    Eyff = np.reshape(ff['Eyff'],(120,120))
    
    Einplane = (Exff**2 + Eyff**2)**0.5
    Einplane_phase = np.angle((Exff),deg=True)
    Einplane_abs   = np.abs(Einplane)
    plt.imshow(Einplane_abs)
    plt.savefig('OAM_09_inplane_Eabs.png',dpi=300)
    plt.show()
    
    plt.imshow(Einplane_phase)
    plt.savefig('OAM_09_inplane_Ephase.png',dpi=300)
    plt.show()
    
