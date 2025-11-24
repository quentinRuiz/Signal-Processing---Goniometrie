{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

# Import module

import numpy as np
import matplotlib.pyplot as plt
import sys
from mathematical_utilities import *
from Antenna_object import *
import copy

# Inner functions 
def check_vector_or_matrix(x, sizeDim2):
    """ 
    verification pour savoir si l'instance d'entrée est un vecteur ou une matrice.

    """
    arr = np.array(x)

    if arr.ndim == 1 and arr.shape[0] == 2:
        return "vecteur"
    elif arr.ndim == 2 and arr.shape == (2, sizeDim2):
        return "matrice"
    else:
        return f"invalide (shape={arr.shape}, required dimension (2,) or (2, Nant))"


def limAxisPattern(limAxis, positionAnt):
    """
    definition des limites pour l'affichage (min/max) en fonction de la positions des antennes entrantes
    limAxis(array) : limites actuelles du plot
    positionAnt(array): position de l'antenne en entrée

    output(array):
    limAxis(array) : actualisation des limites en fonctions de la position de l'antennes entrantes

    """
    [posMinX, posMinY, posMinZ] = limAxis[0,:]
    [posMaxX, posMaxY, posMaxZ] = limAxis[1,:]

    # X axis
    if positionAnt[0]<posMinX:
            posMinX = positionAnt[0]
    if positionAnt[0]>posMaxX:
        posMaxX = positionAnt[0]

    # Y axis
    if positionAnt[1]<posMinY:
        posMinY = positionAnt[1]
    if positionAnt[1]>posMaxY:
        posMaxY = positionAnt[1]

    # Z axis
    if positionAnt[2]<posMinZ:
        posMinZ = positionAnt[2]
    if positionAnt[2]>posMaxZ:
        posMaxZ = positionAnt[2]

    return np.vstack(([posMinX, posMinY, posMinZ], [posMaxX, posMaxY, posMaxZ]))


# Class Passive system 
class AntennaArray:
    def __init__(self, array_type="ULA", Nant=8, Spacement=0.5, Nant_x=4, Nant_y=4, Radius=1.0, origine = [0,0,0]):
        """
        array_type : str ("ULA", "UCA", "URA")
        M          : nb antennes (ULA, UCA)
        d          : espacement (en λ)
        Mx, My     : dimensions (URA)
        R          : rayon (UCA)
        
        """
        # Physical parameters
        self.array_type = array_type.upper()
        if self.array_type == "ULA" or self.array_type == "UCA":
            self.Nant = Nant
        elif self.array_type == "URA":
            self.Nant = Nant_x*Nant_y
            
        self.Spacement = Spacement
        self.Nant_x = Nant_x
        self.Nant_y = Nant_y
        self.Radius = Radius
        self.positions = self._generate_positions() + origine
        self.origineSysteme = origine
        self.orientationSystème = [0,0,1]

        # electronic parameters
        self.FrequencySample = 0
        self.RxGain = 0
        self.PhaseError = []
        self.GainError = []

        # Array pattern parameters
        self.NumberFrequence = 0
        self.azimut = []
        self.elevation   = []
        self.thetaResolution = []
        self.phiResolution = []
        self.AntenaPattern = []
        self.angularShiftAntenna = []

    # Fonctions pour les parametres physiques du systèmes    
    def _generate_positions(self):
        """Génère les coordonnées (x,y,z) en fonction du type de réseau"""
        if self.array_type == "ULA":
            return np.array([[m*self.Spacement, 0, 0] for m in range(self.Nant)])
        
        elif self.array_type == "UCA":
            return np.array([[self.Radius*np.cos(2*np.pi*m/self.Nant),
                              self.Radius*np.sin(2*np.pi*m/self.Nant),
                              0] for m in range(self.Nant)])
        
        elif self.array_type == "URA":
            return np.array([[mx*self.Spacement, my*self.Spacement, 0] 
                             for mx in range(self.Nant_x) 
                             for my in range(self.Nant_y)])
        else:
            raise ValueError("Type de réseau non reconnu (ULA, UCA, URA)")

    def change_origine(self, origine):
        """
        changer l'origine du système
        """
        self.positions = self.positions - self.origineSysteme + origine
        self.origineSysteme = origine

    def rotation_array(self, angles):
        """ 
        effectuer une rotation complete du système autour d'un axe
        angles(array): angle pour chaque axe de la forme [alpha, beta, gamma] exprimé en degrés
        """
        position = self.positions - self.origineSysteme
        position = Rotation(position, np.deg2rad(angles))
        self.positions = position + self.origineSysteme

        self.orientationSystème = Rotation(self.orientationSystème, np.deg2rad(angles))

    # Fonctions pour les parametres electroniques
    def define_electronic_parameters(self, FrequencySample=0, RxGain=0, PhaseError=0, GainError=0):
        """
        Appliquer les parametres electroniques au système
        FrequencySample(int) en Hertz
        RxGain(Float) : en dB
        PhaseError(float/array): en degrés
        GainError(float/array): en dB
        """
        self.FrequencySample = FrequencySample
        
        if isinstance(RxGain, int) or len(RxGain)==self.Nant:
            self.RxGain = RxGain
        else:
            raise ValueError("Phase Error est une erreur constante pour toutes les antennes ou un vecteur de valeurs pour chaque antenne")
        
        if isinstance(PhaseError, int) or len(PhaseError)==self.Nant:
            self.PhaseError = PhaseError
        else:
            raise ValueError("Phase Error est une erreur constante pour toutes les antennes ou un vecteur de valeurs pour chaque antenne")
        
        if isinstance(GainError, int)or len(GainError)==self.Nant:
            self.GainError = GainError
        else:
            raise ValueError("Gain Error est une erreur constante pour toutes les antennes ou un vecteur de valeurs pour chaque antenne")
        
    # definition des antennes sur le systeme
    def get_numberAntenna(self):
        " obtenir le nombre d'antennes du systeme"
        print("Antenna number on the system: ", Nant)
        return self.Nant

    def _checkNumberFrequencies(self):
        if isinstance(self.Antenna, list):
            if isinstance(self.Antenna[0].frequence, list):
                valuesFreq = self.Antenna[0].frequence
                for ff in range(1,self.Nant):
                    freq = self.Antenna[ff].frequence
                    if sorted(valuesFreq) != sorted(freq):
                        raise ValueError("Les frequences ne sont pas identiques pour chaque antennes !")
                self.NumberFrequence = len(valuesFreq)
            else:
                self.NumberFrequence = 1
        else:
            self.NumberFrequence = 1


    def AntennaAttribution(self, antenna: Antenna):
        """
        Attribution des antennes avec leurs parametres au systemes
        antenna(Antenna) : objet ou liste de taille egale au nombre d'antenne d'objet Antenna 
        Dans le cas d'un seul objet, l'antenne est identique sur toute les positions d'antennes du systeme. Dans le cas d'une liste, 
        chaque antenne est attribué en fonction de l'antenne dans la liste et de son index.
        """
        # cas pour une entrée d'uen seule antenne, repetition sur toutes les antennes
        Nant = self.Nant
        if isinstance(antenna, Antenna):
            if Nant == 1:
                self.Antenna = antenna
                self.Antenna.position = self.positions
            else:
                self.Antenna = [copy.deepcopy(antenna) for _ in range(Nant)]
                for ff in range(Nant):
                    self.Antenna[ff].position = self.positions[ff,:]
                self._checkNumberFrequencies()
            
            self.azimut     = antenna.azimut
            self.elevation = antenna.elevation
            self.angularShiftAntenna = np.zeros((2,Nant))
        elif isinstance(antenna, list):
            if len(antenna)== Nant:
                for ind in range(Nant):
                    if isinstance(antenna[ind], Antenna):
                        pass
                    else:
                        raise ValueError(f" Dans la liste d'antenne, l'index {ind} ne correspond pas à une antenne")
                self.Antenna = antenna
                self._checkNumberFrequencies()
                self.azimut     = antenna.azimut
                self.elevation = antenna.elevation
                self.angularShiftAntenna = np.zeros((2,Nant))
                for ff in range(Nant):
                    self.Antenna[ff].position = self.positions[ff,:]
                
            else:
                raise ValueError("Le nombre d'antennes dans la liste ne correspond pas aux nombre d'antennes du système")
        

    def applyAngularShiftAntenna(self,shift):
        """
        Applique un decalage de l'orientation en azimut et en elevation pour toute les antennes,
        ou pour chaque antenne en focntion de la l'entrée.
        shift(Array or vect): [azimut, elevation] sous la forme de (2,) ou (2,Nant) en taille de dimension
        """
        output = check_vector_or_matrix(shift,self.Nant)
        if output =="vecteur":
            for ff in range(self.Nant):
                ant = self.Antenna[ff]
                ant.rotationAntenna(shift)
        elif output=="matrice":
            for ff in range(self.Nant):
                ant = self.Antenna[ff]
                ant.rotationAntenna(shift[:,ff])
        else:
            raise ValueError(output)
        self.angularShiftAntenna = shift
    
        
    # Affichage
    def plot_array(self, ax=None):
        """Affiche les positions des antennes en 3D"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        
        X, Y, Z = self.positions[:,0], self.positions[:,1], self.positions[:,2]
        ax.scatter(X, Y, Z, c='r', s=50)
        ax.set_xlabel("x (λ)")
        ax.set_ylabel("y (λ)")
        ax.set_zlabel("z (λ)")
        ax.set_title(f"Array type : {self.array_type}")
        plt.show()

    def plotSystemePattern3D(self, frequencePlot, fig = None, ax=None, scale = None, vminColorbar = -20, specificRadiusOMNI = 0.3):
        """
        Afficher les diagramme de rayonnement de toutes les antennes du systeme en 3D à leur position. 
        frequencePlot(int): frequence du systeme 
        fig(figure)
        ax(ax)
        scale(array): echelle pour le plot, en automatique par default, sinon ajouter matrice sous la forme (2,3)
        vminColorbar(float): dynamique des diagrammes rayonnements
        specificRadiusOMNI(float): rayon des antennes omni.
    
        """
        if fig == None and ax == None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  # si 3D
        
        if scale ==None:
            limAxis = np.zeros((2,3))
        
        # check frequence to plot
        if not isinstance(frequencePlot, int):
            raise ValueError("frequencePlot doit correspondre à la valeur d'une frequence dans la liste des frequences du systeme \
            sous forme d'entier")
        else:
            for ff in range(self.Nant):
                ant = self.Antenna[ff]
        
                # check frequency list in antenna
                if isinstance(ant.frequence, int):
                    nf = 1
                    indFreq = 0
                elif isinstance(ant.frequence, list):
                    nf = len(ant.frequence)
                    indFreq = np.argmin(np.abs(frequencePlot-np.array(ant.frequence)))
            
                # projection polar plot
                THETA, PHI = np.meshgrid(ant.azimut, ant.elevation)
                TH = np.deg2rad(THETA)
                PH = np.deg2rad(PHI)
                if ant.antennaType == "OMNI":
                    X = specificRadiusOMNI*np.cos(TH) * np.cos(PH) + ant.position[0]
                    Y = specificRadiusOMNI*np.sin(TH) * np.cos(PH) + ant.position[1]
                    Z = specificRadiusOMNI*np.sin(PH) + ant.position[2]
                    pcm = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8,facecolors=plt.cm.jet(np.ones(TH.shape)))
                    if scale==None:
                        limAxis = limAxisPattern(limAxis, ant.position)
            
                
                elif nf == 1 and ant.antennaType !="OMNI":
                    # Coordonnées cartésiennes
                    power = 10**((ant.antennaPattern-ant.Gain_dBi)/10)
                    X = specificRadiusOMNI*power*np.cos(TH) * np.cos(PH) + ant.position[0]
                    Y = specificRadiusOMNI*power*np.sin(TH) * np.cos(PH) + ant.position[1]
                    Z = specificRadiusOMNI*power*np.sin(PH) + ant.position[2]
                    vmax = np.max(ant.antennaPattern)
                    pcm = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(power/np.max(power)), rstride=2, cstride=2, alpha=0.8,\
                                          vmin=vmax+vminColorbar, vmax=vmax)
            
                    if scale==None:
                        limAxis = limAxisPattern(limAxis, ant.position)
            
                    
                else :
                    # Coordonnées cartésiennes
                    power = 10**((ant.antennaPattern[:,:,indFreq]-ant.Gain_dBi[indFreq])/10)
                    X = specificRadiusOMNI*power * np.cos(TH) * np.cos(PH)  + ant.position[0]
                    Y = specificRadiusOMNI*power * np.sin(TH) * np.cos(PH)  + ant.position[1]
                    Z = specificRadiusOMNI*power * np.sin(PH) + ant.position[2]
                    vmax = np.max(ant.antennaPattern)
                    pcm = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(power/np.max(power)), rstride=2, cstride=2, alpha=0.8,\
                                          vmin=vmax+vminColorbar, vmax=vmax)
            
                    if scale==None:
                        limAxis = limAxisPattern(limAxis, ant.position)
            
                    
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if scale == None:
                ax.set_xlim([-1+limAxis[0,0],1+limAxis[1,0]])
                ax.set_ylim([-1+limAxis[0,1],1+limAxis[1,1]])
                ax.set_zlim([-1+limAxis[0,2],1+limAxis[1,2]])
            else:
                ax.set_xlim([scale[0,0],scale[0,0]])
                ax.set_ylim([scale[0,1],scale[1,1]])
                ax.set_zlim([scale[0,2],scale[1,2]])
            plt.subplots_adjust(hspace=0.5)
    
    
            return fig, ax

    
    def steeringVector(self):
        """
        Calcul du steering vector pour le systeme en fonction des antennes. 
    
        """
        azimut = self.azimut
        elevation = self.elevation
        Nant = self.Nant
        nf   = self.NumberFrequence
        if nf == 1:
            steeringVect = np.zeros((len(azimut), len(elevation),Nant), dtype = complex)
            k = 2*np.pi*self.Antenna[0].frequence/3e8
            # Génération des vecteurs d'onde k pour toutes les directions
            for n in range(Nant):
                    #theta, phi = np.meshgrid(np.deg2rad(azimut), np.deg2rad(elevation-self.angularShiftAntenna[1,n]), indexing="ij")  # grilles d'angles
                    theta, phi = np.meshgrid(np.deg2rad(azimut), np.deg2rad(elevation), indexing="ij")  # grilles d'angles

                    kx = k * np.cos(theta) * np.cos(phi)
                    ky = k * np.sin(theta) * np.cos(phi)
                    kz = k * np.sin(phi)
                    phase = kx * (self.positions[n,0]-self.origineSysteme[0]) + ky * (self.positions[n,1]-self.origineSysteme[1]) \
                            + kz * (self.positions[n,2]-self.origineSysteme[2])
                    if self.Antenna[n].antennaType == "OMNI":
                        steeringVect[:,:, n] =  np.exp(-1j * phase) 
                    else:
                        steeringVect[:,:, n] =  10**(self.Antenna[n].antennaPattern/10)*np.exp(-1j * phase)

        else:
            steeringVect = np.zeros((len(azimut), len(elevation),Nant, nf), dtype = complex)
            k = 2*np.pi*np.array(self.Antenna[0].frequence)/3e8
            # Génération des vecteurs d'onde k pour toutes les directions
            for indF in range(nf):
                for n in range(Nant):
                        #theta, phi = np.meshgrid(np.deg2rad(azimut), np.deg2rad(elevation-self.angularShiftAntenna[1,n]), indexing="ij")  # grilles                         d'angles
                        theta, phi = np.meshgrid(np.deg2rad(azimut), np.deg2rad(elevation), indexing="ij")  # grilles                         d'angles
                        kx = k[indF] * np.cos(theta) * np.cos(phi)
                        ky = k[indF] * np.sin(theta) * np.cos(phi)
                        kz = k[indF] * np.sin(phi)
                        phase = kx * (self.positions[n,0]-self.origineSysteme[0]) + ky * (self.positions[n,1]-self.origineSysteme[1])\
                                + kz * (self.positions[n,2]-self.origineSysteme[2])
                        if self.Antenna[n].antennaType == "OMNI":
                            steeringVect[:,:, n, indF] =  np.exp(-1j * phase) 
                        else:
                            steeringVect[:,:, n, indF] =  10**(self.Antenna[n].antennaPattern[:,:,indF]/10)*np.exp(-1j * phase)
        self.steeringVector = steeringVect
    
        