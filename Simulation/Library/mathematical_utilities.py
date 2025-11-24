{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

import numpy as np


# Rotation
def R_x(alpha):
    """ Rotation autour de l'axe d'un angle alpha(radian)"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])

def R_y(beta):
    """ Rotation autour de l'axe d'un angle beta(radian)"""
    return np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

def R_z(gamma):
    """ Rotation autour de l'axe d'un angle gamma(radian)"""
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [0, 0, 1]
    ])

def Rotation(position, angles):
    """ 
    Rotation autour des axe x,y,z(radian)
    positions((Nx3) : coordonnées des points à appliquer la transformation
    angles(3x1)     : Vecteur d'angles sous la forme [alpha, beta, gamma] en radian, pour effectuer les rotations autour de x,y,z.
    """
    
    if angles[0]!=0:
        position = position @ R_x(angles[0])
    if angles[1]!=0:
        position = position @ R_y(angles[1])
    if angles[2]!=0:
        position = position @ R_z(angles[2])
    return position

def cartesian_to_spherical(points, origin=(0,0,0)):
    """
    Convertit un ensemble de points 3D en coordonnées sphériques 
    par rapport à une origine donnée.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Tableau des points (x,y,z).
    origin : tuple (x0, y0, z0)
        Point de référence.

    Returns
    -------
    r : ndarray (N,)
        Distances radiales.
    az : ndarray (N,)
        Azimuts en radians.
    el : ndarray (N,)
        Élévations en radians.
    """
    points = np.asarray(points)
    origin = np.asarray(origin)

    # vecteurs relatifs
    dxyz = points - origin

    dx, dy, dz = dxyz[:,0], dxyz[:,1], dxyz[:,2]

    # distance radiale
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # azimut
    az = np.arctan2(dy, dx)

    # élévation
    el = np.arctan2(dz, np.sqrt(dx**2 + dy**2))

    return r, az, el


        
