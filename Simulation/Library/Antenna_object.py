{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

import numpy as np
import matplotlib.pyplot as plt
from mathematical_utilities import *

########################################## Fonction for antenna ################################################

def sigma_from_HPBW_deg(hpbw_deg):
    # HPBW ≈ 2*sqrt(2 ln2) * sigma
    return np.deg2rad(hpbw_deg) / (2 * np.sqrt(2*np.log(2)))

def compute_directivity_separable(sigma_theta, sigma_phi, ntheta=721, nphi=721):
    """
    sigma_theta/sigma_phi in radians (std dev of Gaussian).
    Returns: D, GdBi_peak, theta_grid(rad), phi_grid(rad), G_lin (ntheta x nphi)
    """
    # grids
    thetas = np.linspace(0, np.pi, ntheta)          # 0..pi
    phis = np.linspace(-np.pi, np.pi, nphi)         # -pi..pi
    
    # 1D profiles (max = 1 at 0)
    f_theta = np.exp(-thetas**2 / (2*sigma_theta**2))
    f_phi = np.exp(-phis**2 / (2*sigma_phi**2))
    
    # intégrales séparées
    int_phi = np.trapezoid(f_phi, phis)                         # ∫_{-π}^{π} f_phi dφ
    int_theta = np.trapezoid(f_theta * np.sin(thetas), thetas)  # ∫_0^{π} f_theta sinθ dθ
    Omega = int_phi * int_theta
    D = 4*np.pi / Omega
    GdBi_peak = 10 * np.log10(D)
    
    # construction 2D (θ x φ)
    F = np.outer(f_theta, f_phi)   # shape (ntheta, nphi)
    G_lin = D * F                  # G(θ,φ) linéaire
    G_dBi = 10 * np.log10(G_lin)
    
    return D, GdBi_peak, thetas, phis, G_lin, G_dBi


########################################## Fonction to define antenna object ##########################################


class Antenna:
    def __init__(self, azimut, elevation, frequence, plageDeFrequence = [], antennaType = "OMNI", ouverture = [45,45],\
                 antennaPattern = [], GaindBi = 0, direction = [1,0,0], position=[0,0,0]):
        """
        azimut          : vecteur d'azimut de l'antenne
        elevation       : vecteur d'elevation de l'antenne
        antennaType     : str ("OMNI", "GAUSSIAN", "CUSTOM")
        antennaPattern  : diagramme de rayonnement de l'antenne de dimension 
        GaindBi(float)  : Gain de l'antenne en dBi
        """
        # Physical parameters
        self.antennaType = antennaType.upper()
        if self.antennaType == "OMNI":
            antennaPattern = 10*np.log10(np.ones((len(azimut), len(elevation))))
        elif self.antennaType == "GAUSSIAN":
            if isinstance(frequence, int):
                nf = 1
            elif isinstance(frequence, list):
                [_, nf] = np.shape(ouverture)

            
            if nf == 1:
                sigma_theta = sigma_from_HPBW_deg(ouverture[0][0])
                sigma_phi = sigma_from_HPBW_deg(ouverture[1][0])
                sigmaAzimut_3dB      = ouverture[0][0]/2/np.sqrt(2*np.log(2))
                sigmaElevation_3dB   = ouverture[1][0]/2/np.sqrt(2*np.log(2))
                
                _, GdBi_peak, _, _, _, _ = compute_directivity_separable(sigma_theta, sigma_phi,
                                                                                          ntheta=361, nphi=721)
                
                azimutCurve          = np.exp(-azimut**2/2/sigmaAzimut_3dB**2)
                elevationCurve       = np.exp(-elevation**2/2/sigmaElevation_3dB**2)
                antennaPattern       = 10*np.log10(elevationCurve[np.newaxis].T*azimutCurve+1e-12).T+GdBi_peak
                self.Gain_dBi = GdBi_peak

            elif nf==len(frequence) and nf>1:
                antennaPattern = np.zeros((len(azimut), len(elevation), nf))
                Gain_dBi = np.zeros((nf,1))
                for ind in range(nf):
                    # rad
                    sigma_theta = sigma_from_HPBW_deg(ouverture[0][ind])
                    sigma_phi = sigma_from_HPBW_deg(ouverture[1][ind])

                    #deg
                    sigmaAzimut_3dB      = ouverture[0][ind]/2/np.sqrt(2*np.log(2))
                    sigmaElevation_3dB   = ouverture[1][ind]/2/np.sqrt(2*np.log(2))

                    _, GdBi_peak, _, _, _, _ = compute_directivity_separable(sigma_theta, sigma_phi,
                                                                                          ntheta=361, nphi=721)
        
                    azimutCurve          = np.exp(-azimut**2/2/sigmaAzimut_3dB**2)
                    elevationCurve       = np.exp(-elevation**2/2/sigmaElevation_3dB**2)
                    antennaPattern[:,:,ind]= 10*np.log10(elevationCurve[np.newaxis].T*azimutCurve+1e-12).T+GdBi_peak
                    Gain_dBi[ind] = GdBi_peak
                self.Gain_dBi = Gain_dBi
            else:
                raise ValueError("L'ouverture 3dB doit etre un vecteur de la forme [2,1] ou [2,len(frequence])")
        elif self.antennaType == "CUSTOM":
                if self._verificationAntennaPattern(azimut, elevation, frequence, antennaPattern):
                    self.Gain_dBi = GaindBi
                    pass

        # attribution de la plage de fonctionnement de l'antenne
        if plageDeFrequence == []:
            if isinstance(frequence, int):
                self.plageDeFrequence = [frequence-0.1*frequence, frequence-0.1*frequence]
            else:
                self.plageDeFrequence = [np.min(frequence)-0.1*np.min(frequence), np.max(frequence)+0.1*np.max(frequence)]
        else:
            self.plageDeFrequence = plageDeFrequence
        
        self.azimut         = azimut
        self.elevation      = elevation
        self.antennaPattern = antennaPattern #self._verificationAntennaPattern(azimut, elevation, frequence, antennaPattern)
        self.ouverture      = ouverture
        self.frequence      = frequence
        self.direction      = direction
        self.position       = position

    def _verificationAntennaPattern(self, azimut, elevation, frequence, antennaPattern):
        if isinstance(frequence, list):
            [Naz, Nel, Nf] = np.shape(antennaPattern)
            if len(azimut)==Naz and len(elevation)==Nel and len(frequence)==Nf:
                return True
            else:
                raise ValueError("Les dimensions Azimut/elevation/frequence/antennaPattern ne coincidents pas")
        elif isinstance(frequence, int):
            [Naz, Nel] = np.shape(antennaPattern)
            if len(azimut)==Naz and len(elevation)==Nel:
                return True
            else:
                raise ValueError("Les dimensions Azimut/elevation/frequence/antennaPattern ne coincidents pas")
    
    def rotationAntenna(self, shift):
        """"
        effectue un decalage du diagramme de rayonnement en fonction des valeurs d'entrée 
        shift(array) = [shiftAzimut, shiftElevation]
        """
        if isinstance(self.frequence, int):
            patt = self.antennaPattern
            az = self.azimut
            el = self.elevation
        
            ref0az = np.argmin(np.abs(az))
            ref0el = np.argmin(np.abs(el))
        
            [shiftAz,shiftEl] = shift
            if shiftAz>180:
                shiftAz = shiftAz-360

            # shift par rapport à la reference
            indShiftAz = np.argmin(np.abs(az-shiftAz))        
            indShiftEl = np.argmin(np.abs(el-shiftEl))

            # decalage du diagramme de rayonnemnt
            self.antennaPattern = np.roll(np.roll(patt, shift=indShiftAz-ref0az, axis=1), shift=indShiftEl-ref0el, axis=0)
            #self.direction = np.round(R_y(np.deg2rad(shiftEl))@(R_z(np.deg2rad(shiftAz))@self.direction),3)

            x = np.cos(np.deg2rad(shiftAz))*np.cos(np.deg2rad(shiftEl))
            y = np.sin(np.deg2rad(shiftAz))*np.cos(np.deg2rad(shiftEl))
            z = np.sin(np.deg2rad(shiftEl))
            vect = np.stack((x,y,z), axis=-1)
            self.direction = np.round(vect, 3)
            
        else:
            pattern = self.antennaPattern
            for ff in range(len(self.frequence)):
                patt = pattern[:,:,ff]
                az = self.azimut
                el = self.elevation
            
                ref0az = np.argmin(np.abs(az))
                ref0el = np.argmin(np.abs(el))
            
                [shiftAz,shiftEl] = shift

                if shiftAz>180:
                    shiftAz = shiftAz-360

    
                # shift par rapport à la reference
                indShiftAz = np.argmin(np.abs(az-shiftAz))        
                indShiftEl = np.argmin(np.abs(el-shiftEl))
    
                # decalage du diagramme de rayonnemnt
                pattern[:,:,ff] = np.roll(np.roll(patt, shift=indShiftAz-ref0az, axis=1), shift=indShiftEl-ref0el, axis=0)
            self.antennaPattern = pattern
            #self.direction = np.round(R_y(np.deg2rad(shiftEl))@(R_z(np.deg2rad(shiftAz))@self.direction))
            x = np.cos(np.deg2rad(shiftAz))*np.cos(np.deg2rad(shiftEl))
            y = np.sin(np.deg2rad(shiftAz))*np.cos(np.deg2rad(shiftEl))
            z = np.sin(np.deg2rad(shiftEl))
            vect = np.stack((x,y,z), axis=-1)
            self.direction = np.round(vect, 3)

    
    def plot_pattern(self, ax=None, vminColorbar = -20):
        """Affiche le diagramme de rayonnement de l'antenne en 2D"""
        if ax is None:
            fig = plt.figure()
        if isinstance(self.frequence, int):
            nf = 1
        elif isinstance(self.frequence, list):
            nf = len(self.frequence)
            
        if nf == 1 or  self.antennaType == "OMNI":
            ax = fig.add_subplot(111)
            if self.antennaType == "OMNI":
                pcm = ax.pcolormesh(self.antennaPattern)
            else:
                vmax = np.max(self.antennaPattern)
                pcm = ax.pcolormesh(self.azimut, self.elevation, self.antennaPattern, vmin=vmax+vminColorbar, vmax=vmax) 
            ax.set_title(f"fréquence : {self.frequence}Hz")
            ax.set_xlabel(r"azimut($\theta$)(°)")
            ax.set_ylabel(r"elevation($\phi$)(°)")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label('dB')
        else:
            for ind_f in range(len(self.frequence)):
                ax = fig.add_subplot(nf, 1, ind_f+1)
                vmax = np.max(self.antennaPattern[:,:,ind_f])
                pcm = ax.pcolormesh(self.azimut, self.elevation,self.antennaPattern[:,:,ind_f], vmin=vmax+vminColorbar, vmax=vmax)
                ax.set_title(f"fréquence : {self.frequence[ind_f]}Hz")
                ax.set_xlabel(r"azimut($\theta$)(°)")
                ax.set_ylabel(r"elevation($\phi$)(°)")
                cb = fig.colorbar(pcm, ax=ax)
                cb.set_label('dB')
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def plot_pattern3D(self, fig=None, ax=None, vminColorbar = -20):
        """Affiche le diagramme de rayonnement de l'antenne en 2D"""
        if fig is None:
            fig = plt.figure()

            
        if isinstance(self.frequence, int):
            nf = 1
        elif isinstance(self.frequence, list):
            nf = len(self.frequence)

        # projection polar plot
        THETA, PHI = np.meshgrid(self.azimut, self.elevation)
        TH = np.deg2rad(THETA)
        PH = np.deg2rad(PHI)
        
        
        if nf == 1 or  self.antennaType == "OMNI":
            ax = fig.add_subplot(111, projection='3d')
            
            if self.antennaType == "OMNI":
                # Coordonnées cartésiennes
                power = self.antennaPattern
                X = np.cos(TH) * np.cos(PH)
                Y = np.sin(TH) * np.cos(PH)
                Z = np.sin(PH)
                pcm = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8)
            else:
                # Coordonnées cartésiennes
                power = 10**((self.antennaPattern-self.Gain_dBi)/10)
                X = power * np.cos(TH) * np.cos(PH)  + self.position[0]
                Y = power * np.sin(TH) * np.cos(PH)  + self.position[1]
                Z = power * np.sin(PH) + self.position[2]
                vmax = np.max(self.antennaPattern)
                pcm = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(power/np.max(power)), rstride=2, cstride=2, alpha=0.8,\
                                      vmin=vmax+vminColorbar, vmax=vmax)

            ax.set_title(f"fréquence : {self.frequence}Hz\n puissance linéaire normalisée")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim([-1+self.position[0],1+self.position[0]])
            ax.set_ylim([-1+self.position[1],1+self.position[1]])
            ax.set_zlim([-1+self.position[2],1+self.position[2]])
            plt.subplots_adjust(hspace=0.5)
            return ax
        else:
            for ind_f in range(len(self.frequence)):
                ax = fig.add_subplot(nf, 1, ind_f+1, projection='3d')

                # Coordonnées cartésiennes
                if self.antennaType == "OMNI": 
                    X = np.cos(TH) * np.cos(PH)
                    Y = np.sin(TH) * np.cos(PH)
                    Z = np.sin(PH)
                else:
                    power = 10**((self.antennaPattern[:,:,ind_f]-self.Gain_dBi[ind_f])/10)
                    X = power * np.cos(TH) * np.cos(PH) + self.position[0]
                    Y = power * np.sin(TH) * np.cos(PH) + self.position[1]
                    Z = power * np.sin(PH) + self.position[2]
                
                vmax = np.max(self.antennaPattern[:,:,ind_f])
                pcm = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(power/np.max(power)), rstride=2, cstride=2, alpha=0.8,\
                                      vmin=vmax+vminColorbar, vmax=vmax)
                ax.set_title(f"fréquence : {self.frequence[ind_f]}Hz\n puissance linéaire normalisée")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-1+self.position[0],1+self.position[0]])
                ax.set_ylim([-1+self.position[1],1+self.position[1]])
                ax.set_zlim([-1+self.position[2],1+self.position[2]])
                plt.subplots_adjust(hspace=0.5)
                return ax

    def plot_pattern3D_simple(self, fig=None, ax=None, vminColorbar = -20):
        """Affiche le diagramme de rayonnement de l'antenne en 2D"""
        if fig is None:
            fig = plt.figure()

        if isinstance(self.frequence, int):
            nf = 1
        elif isinstance(self.frequence, list):
            nf = len(self.frequence)

        # projection polar plot
        THETA, PHI = np.meshgrid(self.azimut, self.elevation)
        TH = np.deg2rad(THETA)
        PH = np.deg2rad(PHI)
        
        
        if nf == 1 or  self.antennaType == "OMNI":
            ax = fig.add_subplot(111, projection='3d')
            
            if self.antennaType == "OMNI":
                # Coordonnées cartésiennes
                power = self.antennaPattern
                X = np.cos(TH) * np.cos(PH)
                Y = np.sin(TH) * np.cos(PH)
                Z = np.sin(PH)
                pcm = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8)
            else:
                # Coordonnées cartésiennes
                power = 10**((self.antennaPattern-self.Gain_dBi)/10)
                X = power * np.cos(TH) * np.cos(PH)  + self.position[0]
                Y = power * np.sin(TH) * np.cos(PH)  + self.position[1]
                Z = power * np.sin(PH) + self.position[2]
                vmax = np.max(self.antennaPattern)
                pcm = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(power/np.max(power)), rstride=2, cstride=2, alpha=0.8,\
                                      vmin=vmax+vminColorbar, vmax=vmax)

            ax.set_title(f"fréquence : {self.frequence}Hz\n puissance linéaire normalisée")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim([-1+self.position[0],1+self.position[0]])
            ax.set_ylim([-1+self.position[1],1+self.position[1]])
            ax.set_zlim([-1+self.position[2],1+self.position[2]])
            plt.subplots_adjust(hspace=0.5)
            return ax
        else:
            for ind_f in range(len(self.frequence)):
                ax = fig.add_subplot(nf, 1, ind_f+1, projection='3d')

                # Coordonnées cartésiennes
                if self.antennaType == "OMNI": 
                    X = np.cos(TH) * np.cos(PH)
                    Y = np.sin(TH) * np.cos(PH)
                    Z = np.sin(PH)
                else:
                    power = 10**((self.antennaPattern[:,:,ind_f]-self.Gain_dBi[ind_f])/10)
                    X = power * np.cos(TH) * np.cos(PH) + self.position[0]
                    Y = power * np.sin(TH) * np.cos(PH) + self.position[1]
                    Z = power * np.sin(PH) + self.position[2]
                
                vmax = np.max(self.antennaPattern[:,:,ind_f])
                pcm = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(power/np.max(power)), rstride=2, cstride=2, alpha=0.8,\
                                      vmin=vmax+vminColorbar, vmax=vmax)
                ax.set_title(f"fréquence : {self.frequence[ind_f]}Hz\n puissance linéaire normalisée")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-1+self.position[0],1+self.position[0]])
                ax.set_ylim([-1+self.position[1],1+self.position[1]])
                ax.set_zlim([-1+self.position[2],1+self.position[2]])
                plt.subplots_adjust(hspace=0.5)
                return ax