#!/usr/bin/env python

################################################################################
#  Copyright (C) 2016 Jose M. Llorens
#
# This file is part of QWVAREX
#
# QWVAREX is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# QWVAREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
################################################################################

import scipy as sp
import scipy.constants as spc

def Gaussian1d(z, a, z0=0, power = 1.0):
    """Auxiliar function to get a real wave function based on a Gaussian 
    function normalized to the modulus squared integral.

    Keyword arguments:
       z -- array containing the spatial 1D mesh
       a -- Gaussian width
       z0 -- Gaussian maximum position
       power -- Computes t Gaussian1d**power
    """

    return ( sp.exp(-(z-z0)**2/(2.0*a**2))/sp.sqrt(a*sp.sqrt(sp.pi)) )**power


def G_int(x_in):
    """Polynomial approximation to G(x) (Coulomb integral)
    Defined in Apendix A of Mares and Chuang, J. Appl. Phys. 74, 1388 (1993)

    Keyword arguments:
       x_in -- Normalized 2|ze-zh|/lambda
    """

    
    A = -8.9707E-1
    B = -2.5262E-1
    C = 2.2576E-1
    D = 3.2373E-2
    E = -4.1369E-4
    
    G = sp.zeros_like(x_in)
    G[x_in < 0] = float('nan')
    G[x_in == 0] = 1.0
    
    ind = (x_in > 0) * (x_in <= 6.8)
    x = x_in[ind]
    aux = sp.log(x/2.0)    
    G[ind] = 1.0 + A * x + (B * aux + C) * x**2 +\
                            D * x**3 + E * x**4 * aux
    ind = x_in > 6.8    
    x = x_in[ind]
    G[ind] = 1.0/x - 3.0/x**3 + 45.0/x**5 - 1575.0/x**7
    
    return G
    
def X_binding(par, qw):
    """Computation of the exciton binding energy given the electron and hole
    wave function and the variational parameter par
    Defined in Eq. 15(b) of Mares and Chuang, J. Appl. Phys. 74, 1388 (1993)

    Keyword arguments:
       par -- variational parameter in [nm]
       qw -- Object containing the electronic structure related magnitudes. See QW.
    """
    
    lam = par * 1.0E-9    
    Ry = spc.physical_constants['Rydberg constant times hc in eV'][0]
    R0 = Ry*qw.mu/qw.eps_r**2
    aB = spc.physical_constants['Bohr radius'][0]   
    ax = aB*qw.eps_r/qw.mu
    beta = ax/lam  
    C0 = R0*beta**2
    C1 = -R0*4.0*beta
    
    Ze = sp.zeros([1,qw.grid.shape[0]])
    Ze[0,:] = qw.grid[:]
    Zh = sp.zeros([qw.grid.shape[0],1])
    Zh[:,0] = qw.grid[:]
    X = 2.0*sp.absolute(Ze-Zh)/par
    Fe = sp.zeros([1,qw.grid.shape[0]])
    Fe[0,:] = qw.Elec.wf[:]**2
    Fh = sp.zeros([qw.grid.shape[0],1])
    Fh[:,0] = qw.Hole.wf[:]**2
    Int = Fe * Fh * G_int(X)
    Val = sp.trapz( sp.trapz(Int, Ze.flatten() ), Zh.flatten())
    
    return C0 + C1 * Val 


class state():
    """Simple class to define a quantum well state

    Parameters:
       n -- [Integer] arbitrary quantum number
       E -- [float] State energy level
       wf -- [1D array] Wave function array
    """

    n = 1
    E = 1.2
    wf = sp.r_[0.0,0.0]
    
class Mat_AlGaAs():
    """Simple class to define the basic parameters for the AlGaAs ternary.
    Values extracted from: Mares and Chuang, J. Appl. Phys. 74, 1388 (1993)

    Parameters:
       x -- [float] Aluminium content
    Methods:
       Eg() -- Returns the bandgap at Gamma
       me() -- Electron effective mass
       mhh() -- Heavy hole effective mass
       mlh() -- Light hole effective mass
    """
    
    def __init__(self, x = 0):    
        self.x = x
    
    def Eg(self):
        return 1.426 + 1.247 * self.x
    def me(self):
        return 0.0665 + 0.0835 * self.x
    def mhh(self):
        return 0.34 + 0.42 * self.x
    def mlh(self):
        return 0.094 + 0.043 * self.x
        

class qw():
    """Simple class to encapsulate the quantum well magnitudes

    Parameters:
       grid -- [1D array] spatial mesh
       mu -- [float] Reduced mass
       eps_r -- [float] Relative permittivity
       Elec -- [state] Electron state
       Hole -- [state] Hole state
    """
    
    grid = sp.r_[0.0,0.0]
    # Reduced effective mass    
    mu = 1.0
    # Relative permitivitty
    eps_r = 10.0
    Elec = state()
    Hole = state()
    
def main():
    """Example on how to compute the exciton bingind energy.
    """
   
    import scipy.optimize as spopt


    # Case of a GaAs QW
    GaAs = Mat_AlGaAs(x=0.0)
    
    # Initialize the QW
    QW = qw()
    
    # Trivial uniform mesh
    z = sp.linspace(-10.0, 10.0, 200)

    # Assign values to the QW object
    QW.grid = z
    QW.mu = 1.0/(1.0/GaAs.me() + 1.0/GaAs.mhh())
    
    QW.Elec.n = 1
    QW.Elec.E = 1.0
    QW.Elec.wf = Gaussian1d(z, 1.0, 0.0)
    
    QW.Hole.n = 1
    QW.Hole.E = 1.0
    QW.Hole.wf = Gaussian1d(z, 0.8, 0.0)
    
    # Optionally, plot the wf distribution
    # sp.savetxt('wf.dat',sp.c_[z, QW.Elec.wf, QW.Hole.wf])
    
    # Optionally, plot the function to be minimized
    # lam = sp.linspace(0.1, 20)
    # eb = ''
    # for l in lam:
        # X = X_binding(l,QW)
        # eb += 2*'%.8E '%(l, X)+'\n'
    # open('Ex_lam.dat','w').write(eb)
    
    # Locate the minimum using the Brent method
    xopt, fval, ierr, nf = spopt.fminbound(X_binding,0.0,25.0,args=(QW,), full_output=True)
    
    print "Exciton wave function extension: %.3f nm"%(xopt)
    print "Exciton wave function binding energy: %.3f meV"%(fval*1000.0)
    print "Error code: %i"%(ierr)
    print "Number of function evaluations: %i"%(nf)
    
    
if __name__ == "__main__":
    main()

