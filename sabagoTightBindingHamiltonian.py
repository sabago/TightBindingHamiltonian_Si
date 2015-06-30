# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:50:53 2015

This is a code to plot the band structure of diamond structure Si run in the
UBUNTU Python IDE, "SPYDER" which uses the "numpy" and "matplotlib" modules 
among others.
The code involves generation of the tight binding Hamiltonian of diamond 
structure Si from first principles, solving for and sorting for the real 
parts of the eigenvalues of the hamiltonian that were plotted against kpoints.
@author: sabago
"""

import numpy as np
import scipy.linalg 
import matplotlib.pyplot as plt  
from math import pi,cos,sin 

def k_points(n):
    # Define a set of k points along the Brillouin zone boundary  
    #of Si in the Diamond crystal shape from
    # W (0.5,0.25,0.75) to Gamma (0,0,0) to X (0.5,0.,0.5), back to W
    #then to L(0.5,0.5,0.5) and back to Gamma Space the points
    # evenly, based on the scaling parameter n, in this case =10.
    kpoints = []
    step = 0.5/float(n)
    kx,ky,kz = 0.50,0.25,0.75    # Start at the W point (1/2,1/4,3/4)
    kpoints.append((kx,ky,kz))
    for i in range(n): # Move to the Gamma point (0,0,0)
        kx,ky,kz = kx-step,ky-0.5*step,kz-1.5*step
        kpoints.append((kx,ky,kz))
    for i in range(n): # Now go to the X point (0.5,0,0.5)
        kx,kz = kx+step,kz+step
        kpoints.append((kx,ky,kz))
    for i in range(n): # Now go back to W
        ky,kz = ky + 0.5*step, kz + 0.5*step    
        kpoints.append((kx,ky,kz))
    for i in range(n): # Now go to the L point (0.5,0.5,0.5)
        ky,kz = kx+0.5*step,kz-0.5*step
        kpoints.append((kx,ky,kz))
    for i in range(n): # Now go back to Gamma
        kx,ky,kz = kx-step, ky-step, kz-step    
        kpoints.append((kx,ky,kz))
    return kpoints 
    
def set_phases(kpoint):
    #Defining the tight-binding Energy Dispersion "phases"
    kx,ky,kz = kpoint
    kxp,kyp,kzp = kx*pi/2.,ky*pi/2.,kz*pi/2.# The a's cancel here
    g1_real = cos(kxp)*cos(kyp)*cos(kzp)
    g1_imag = -sin(kxp)*sin(kyp)*sin(kzp)
    g2_real = -cos(kxp)*sin(kyp)*sin(kzp)
    g2_imag = sin(kxp)*cos(kyp)*cos(kzp)
    g3_real = -sin(kxp)*cos(kyp)*sin(kzp)
    g3_imag = cos(kxp)*sin(kyp)*cos(kzp)
    g4_real = -sin(kxp)*sin(kyp)*cos(kzp)
    g4_imag = cos(kxp)*cos(kyp)*sin(kzp)
    
    # "c" stands for the complex conjugate
    g1,g1c = g1_real+g1_imag*1j,g1_real-g1_imag*1j
    g2,g2c = g2_real+g2_imag*1j,g2_real-g2_imag*1j
    g3,g3c = g3_real+g3_imag*1j,g3_real-g3_imag*1j
    g4,g4c = g4_real+g4_imag*1j,g4_real-g4_imag*1j
    return (g1,g1c,g2,g2c,g3,g3c,g4,g4c)
    
    #Defining parameters for the 8x8 tight-binding Hamiltonian of Si 
    #in the Diamond Crystal Structure
    #experimental voltage (energy) values from 
    #http://www.sciencedirect.com/science/article/pii/S002236979700190X#
    
def diag(H):
    # Make the diagonal elements which have self terms, considered p and s orbitals
        e_s = -4.15
        e_p = 3.05
        H[0,0] = H[1,1] = e_s
        H[2,2] = H[3,3] = H[4,4] = H[5,5] = H[6,6] = H[7,7] = e_p
        return H
    
def off_diag(H,phases):
        g1,g1c,g2,g2c,g3,g3c,g4,g4c = phases
    # Make the off-diagonal parts which have terms related to coupling
        e_ss = -6.78
        e_sp = 5.91
        e_xx = 2.05
        e_xy = 4.26
        H[1,0] = e_ss*g1c
        H[0,1] = e_ss*g1

        H[2,1] = -e_sp*g2
        H[1,2] = -e_sp*g2c
        H[3,1] = -e_sp*g3
        H[1,3] = -e_sp*g3c
        H[4,1] = -e_sp*g4
        H[1,4] = -e_sp*g4c

        H[5,0] = e_sp*g2c
        H[0,5] = e_sp*g2
        H[6,0] = e_sp*g3c
        H[0,6] = e_sp*g3
        H[7,0] = e_sp*g4c
        H[0,7] = e_sp*g4

        H[5,2] = e_xx*g1c
        H[2,5] = e_xx*g1
        H[6,2] = e_xy*g4c
        H[2,6] = e_xy*g4
        H[7,2] = e_xy*g3c
        H[2,7] = e_xy*g3

        H[5,3] = e_xy*g4c
        H[3,5] = e_xy*g4
        H[6,3] = e_xx*g1c
        H[3,6] = e_xx*g1
        H[7,3] = e_xy*g2c
        H[3,7] = e_xy*g2

        H[5,4] = e_xy*g3c
        H[4,5] = e_xy*g3
        H[6,4] = e_xy*g2c
        H[4,6] = e_xy*g2
        H[7,4] = e_xx*g1c
        H[4,7] = e_xx*g1
        return H
    #Putting the Hamiltonian together
def TB_Hamiltonian(kpoint):
    phase_factors = set_phases(kpoint)
    H = np.zeros((8,8), dtype = complex)
    
    H = diag(H)
    H = off_diag(H,phase_factors)
    #print(H)
    return H
    #Purely Python related function
def sort_eig(E):
    # This is trickier than it sounds, since NumPy doesn't define
    # sort on Complex numbers. Convert to a normal python array of
    # reals, and sort.
    enarray = []
    for en in E: enarray.append(en.real)
    enarray = np.sort(enarray)
    #print(enarray)
    return enarray
    

#Running the code by getting the real parts of the eigenvalues of the tight banding Hamiltonian   
    
    
plt.figure(figsize = (6,5))
kpoints = k_points(10)
energies = []

for kpoints in kpoints:
        H = TB_Hamiltonian(kpoints)
        #print(H)

        eigval , eigvec =  scipy.linalg.eig(H) 
        #print(eigval)
        energyarray = sort_eig(eigval)
       # energyarray.sort()
        energies.append(energyarray)
        print(energies)
#energy=[1,2,3]
kpoints=range(51)
plt.plot(kpoints, energies, 'or', label ='Exp. Data')
xticks = np.linspace(0,51, 6)
labels = ["W (0.5,0.25,0.75)", "G(0,0,0)", "X(0.5,0,0.5)", "W(0.5,0.25,0.75)", "L(0.5,0.5,0.5)", "G(0,0,0)"]
plt.xticks(xticks, labels, rotation = 'vertical')
plt.xlabel("K-Points")
plt.ylabel("Energy (eV)")
plt.title('Bandstructure of Si (diamond structure)')
plt.tick_params(axis='y',which='minor',bottom='off')
ml = MultipleLocator(1)
plt.axes().yaxis.set_minor_locator(ml)
plt.tight_layout()
plt.savefig('Si bandstructure from Hamiltonian.pdf')
