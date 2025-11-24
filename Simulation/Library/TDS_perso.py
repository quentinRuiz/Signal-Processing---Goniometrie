{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
import numpy as np
from scipy import signal 
from scipy.fft import fft, ifft, fftshift, ifftshift
import pandas as pd

#Fonction
def add_awgn(signal, fs, noise_floor_dBm, R=50):
    """
    add noise to a signal with desired noise floor in dBm
    input:
    - signal (ndarray) : signal to add noise
    - fs (int) : frequency sample
    - noise_floor_dBm (int) : noise floor in dBm 
    - R (int) : resistance of the input circuit
    output :
    noisy signal 
    """
    BW = fs  # largeur bande = fs
    Pn_W = 10**((noise_floor_dBm + 10*np.log10(BW) - 30)/10)
    noise_std = np.sqrt(Pn_W * R)
    noise = (np.random.normal(0, noise_std, size=signal.shape) +
             1j * np.random.normal(0, noise_std, size=signal.shape)) / np.sqrt(2)
    return signal + noise


def normalizeFFT(signal, frequencySample, R = 50):
    """
    FFT for analyse signal in dBm
    input:
    - signal (ndarray): signal to analyse
    - freqency sample (intà)
    - R (int) : resistance of the input circuit
    output:
    psd (ndarray) : spectral density of the signal 
    psd_dBm (ndarray): psd in dBm
    """
    Nfft = len(signal)
    fftsig = fftshift(fft(signal)/Nfft)
    psd = abs(fftsig)**2/R
    psd = psd/1e-3
    psd_dBm = 10*np.log10(psd+10e-12)
    return psd , psd_dBm   




def generationDataWIMAX(FFT_size, NumberBloc, NpilotSubcarrier,NdataSubcarrier, NguardBand):
    """
    Random Generation data for WIMAX, with uniform repartition of pilot, data subcarrier , DC null
    input:
    - FFT_size(int) : size of the FFT
    - NumberBloc(int): number of data block to implement
    - NpilotSubcarrier(int):  total number of pilot subcarrier
    - NguardBand(int): total number of the guardband

    output:
    - prepData2signal(ndarray: (FFT_size, NumberBloc)): data in the freqeuncy domain
    - pilotValue(ndarray) : value of the pilot subcarrier 
    - dataValue(ndarray): value of the data subcarrier
    - pilot_position(array): position of the pilot subcarrier
    - data_position(array): position of the data subcarrier
    
    """
    data = np.zeros((FFT_size,NumberBloc), dtype=complex)
    
    # pilot subcarrier BPSK
    # pilot position
    pilot_space = (FFT_size-NguardBand)/NpilotSubcarrier
    pilot_position = np.arange(NguardBand/2, FFT_size-NguardBand/2, pilot_space).astype(int)
    
    # pilot data
    bitsSubcarrier = np.random.randint(0, 2, (NpilotSubcarrier, NumberBloc))
    pilotValue = 2 * bitsSubcarrier - 1
    data[pilot_position,:] = pilotValue
    
    # data subcarrier QPSK
    # data position
    data_position = np.arange(NguardBand/2, FFT_size-NguardBand/2, 1).astype(int)
    data_position = np.setdiff1d(data_position, pilot_position)
    bits = np.random.randint(0, 4, (NdataSubcarrier, NumberBloc))  # 2 bits par symbole
    dataValue = np.exp(1j*2*np.pi*(bits+0.5)/4)
    data[data_position,:] = dataValue
    
    
    # remove DC carrier 
    prepData2signal = ifftshift(data, axes=0)
    prepData2signal[0,:] = 0

    return prepData2signal, pilotValue, dataValue, pilot_position, data_position
def PAPR_signal(signal):
    """ return the PAPR of the signal"""
    return 10 * np.log10(np.max(np.abs(signal)**2)/np.mean(np.abs(signal)**2))

def tapered_window(N, T):
    """ Creation of the N tapering window, with raised cosinus for the T first and last samples """
    fade = 0.5 * (1 - np.cos(np.pi * np.arange(T) / T))
    window = np.ones(N)
    window[:T] = fade           # Fade-in
    window[-T:] = fade[::-1]    # Fade-out
    return window

def clipping_signal(signal, papr_target_db):
    """
    Clipping th signal accorind with a threshold
    input:
    - signal(ndarray): signal to apply the clipping
    - papr_target_db(int): value of the treshold in dB

    output:
    clipped(ndarray): signal clipped
    """
    P_avg = np.mean(np.abs(signal)**2, axis=0)
    papr_target_lin = 10**(papr_target_db / 10)
    A_clip = np.sqrt(papr_target_lin * P_avg)
    
    clipped = np.zeros((signal.shape), dtype=complex)
    for i in range(len(A_clip)):
        clipped[:,i] = np.where(np.abs(signal[:,i]) > A_clip[i], A_clip[i] * signal[:,i] / np.abs(signal[:,i]), signal[:,i])
    return clipped



def add_Cyclic_Prefix(signal, sizeCP):
    """ Add the cyclic prefix """ 
    return np.concatenate((signal[-sizeCP:,:], signal), 0)


# creation of the function to generate the totale signal
def generationWIMAX(bandwidth, NumberBloc, papr_target_db = 6):
    """
    Generation of a random signal based on the WIMAX protocole
    the data are randomly generated for the pilot(BPSK) and for the data(QPSK). The DC of signal is null. 
    The signal is transform in the temporal domain, a clipping effect is applied, a Cyclic prefix added, then a 
    tapered window is applied to smooth the signal.
    input:
    - bandwidth(int): desired signal bandwidth in MHz
    - NumberBloc(int): number of successive OFDM block 
    - papr_target_db(optionnal: float): threshold for the PAPR, 6dB by default

    Output:
    - signalWithCpFilt(array): generated signal
    - timeVect(array): temporal vector of the signal
    - tableParams(dataframe): table of signal parameters
    - tableDataSignal(dataframe): table of the data of the signal

    """
    

    if bandwidth == 1.25e6:
        FFT_size = 128
        NdataSubcarrier = 72
        NpilotSubcarrier = 12
        NguardBand = 44
        GuardTime = 1/8
        oversamplingRate = 28/25
        
    elif bandwidth == 5e6:
        FFT_size = 512
        NdataSubcarrier = 360
        NpilotSubcarrier = 60
        NguardBand = 92
        GuardTime = 1/8
        oversamplingRate = 28/25
        
    elif bandwidth == 10e6:
        FFT_size = 1024
        NdataSubcarrier = 720
        NpilotSubcarrier = 120
        NguardBand = 184
        GuardTime = 1/8
        oversamplingRate = 28/25
        
    elif bandwidth == 20e6:
        FFT_size = 2048
        NdataSubcarrier = 1440
        NpilotSubcarrier = 240
        NguardBand = 368
        GuardTime = 1/8
        oversamplingRate = 28/25
    else:
        raise ValueError("The value of the bandwidth must be : 1.25e6, 5e6, 10e6 or 20e6")

    fs = round(bandwidth*oversamplingRate*1e-6, 3) # frequency sample
    df  = fs/FFT_size
    
    # print informations
    symbolDuration = (FFT_size*(1+GuardTime))/(bandwidth*oversamplingRate)
    print("fs = : " ,fs , "MHz")
    print("Bloc Symbol duration :", round(symbolDuration*1e6, 1), "micros")

    prepData2signal, pilotVal,dataVal, pilot_pos, data_pos = generationDataWIMAX(FFT_size, NumberBloc, NpilotSubcarrier,NdataSubcarrier, NguardBand)

    # signal to temporal domain 
    signal = fftshift(ifft(prepData2signal, axis=0))
    
    # apply clipping to the signal 
    clippedSignal = clipping_signal(signal, papr_target_db)
    
    # Add cyclic prefix 
    size_CP = int(GuardTime*FFT_size)
    CP_time = size_CP*fs
    clippedSignalWithCP = add_Cyclic_Prefix(clippedSignal, size_CP)

    # time of the total length of the symbol
    t_symb = len(clippedSignalWithCP[:,0])/fs
    
    # parallele to serial
    signalSerial = clippedSignalWithCP.T.reshape(-1,1, order='C') # transposition to get the right value
    
    # windowing the signal
    size_tapering = int(len(signalSerial)/64)
    window = tapered_window(len(signalSerial), size_tapering)
    
    # apply window to the total signal
    signalWithCpFilt = signalSerial*window[:,np.newaxis]
    t_trame = len(signalWithCpFilt)/fs

    # temporal vector
    timeVect = np.arange(0,len(signalWithCpFilt), 1)/(fs*1e6)
    
    # Parameters table 
    dataSignal = {
        'Parameters': [
            'All data',
            'pilot value',
            'data value',
            'pilot position',
            'data position'
        ],
        'Valeur': [
            prepData2signal,
            pilotVal,
            dataVal,
            pilot_pos,
            data_pos
        ]
    }
    
    params = {
        'Parameters': [
            'fs(MHz)',
            'bandwidth(MHz)',
            'Nfft',
            'Ndata subcarrier',
            'Npilot subcarrier',
            'Nguard subcarrier',
            'subcarrier space',
            'Duration of symbols without CP(s)',
            'Size CP',
            'CP Duration(s)',
            'Duration of symbols with CP(s)(s)',
            'Duration of the trame',
            'CP factor'
        ],
        'Valeur': [
            fs,
            bandwidth,
            FFT_size,
            NdataSubcarrier,
            NpilotSubcarrier,
            NguardBand,
            df,
            symbolDuration,
            size_CP,
            CP_time,
            t_symb,
            t_trame,
            GuardTime
        ]
    }
    tableParams = pd.DataFrame(params)
    tableDataSignal = pd.DataFrame(dataSignal)

    return signalWithCpFilt,timeVect, tableParams, tableDataSignal



def beamforming_1D(Rxx, steeringVect):
    """
    Beamforming 1D
    input: 
    - Rxx(ndarray): covariance matrix of the signal 
    - steering Vector(array)
    output:
    - beamNorm(array): normalized beamforming estimation
    - beam(array):  beamforming estimation
    """
    beam = steerVect.conj()@Rxx@steerVect.T
    beam = np.abs(np.sum(beam, axis=1))
    beamNorm = beam/np.max(beam) # normalisation
    return beamNorm, beam

def CAPON_beamformer_1D(Rxx, steeringVect):
    """
    Capon beamformer 1D
    input: 
    - Rxx(ndarray): covariance matrix of the signal 
    - steering Vector(array)
    output:
    - caponNorm(array): normalized Capon beamformer estimation
    - capon(array):  Capon beamformer estimation
    """
    Rxx_inv = np.linalg.pinv(Rxx + 1e-16*np.eye(Rxx.shape[0]))
    capon = np.zeros(len(steeringVect), dtype="complex")
    for it in range(len(steeringVect)):
        capon[it] = 1/(steerVect[it,:].conj()@Rxx_inv@steerVect[it,:].T)
    capon = np.abs(capon)
    caponNorm = capon/np.max(capon) # normalisation
    return caponNorm, capon

def MUSIC_beamformer_1D(Rxx, steeringVect, nsource):
    """
    MUSIC beamformer 1D
    input: 
    - Rxx(ndarray): covariance matrix of the signal 
    - steering Vector(array)
    - nsource(int): number of source
    output:
    - musicNorm(array): normalized MUSIC beamformer estimation
    - music(array):  MUSIC beamformer estimation
    """
    eigVal, eigVect = np.linalg.eig(Rxx)
    idx = np.argsort(eigVal)[::-1]
    eigVect = eigVect[:, idx]
    EE = eigVect[:, nsource:]@eigVect[:, nsource:].T.conj()
    music = np.zeros(len(thetaVect), dtype="complex")
    for it in range(len(thetaVect)):
        music[it] = 1/(steerVect[it,:].conj()@EE@steerVect[it,:].T)
    music = np.abs(music) # normalisation
    musicNorm = music/np.max(music) # normalisation
    return musicNorm, music

def champ_libre_mimo(tx_positions, rx_positions, freq):
    """
    Calcule la matrice de canal MIMO en champ libre avec atténuation et déphasage.
    
    Paramètres :
    - tx_positions : array (N_tx, 3) positions des antennes TX (x, y, z)
    - rx_positions : array (N_rx, 3) positions des antennes RX (x, y, z)
    - freq : fréquence en Hz
    
    Retour :
    - H : matrice MIMO complexe (N_rx x N_tx)
    - distances : matrice des distances (N_rx x N_tx)
    - retards : matrice des retards en secondes (N_rx x N_tx)
    """
    c = 3e8  # Vitesse de la lumière (m/s)
    wavelength = c / freq
    
    N_rx = rx_positions.shape[1]
    N_tx = tx_positions.shape[1]
    
    H = np.zeros((N_rx, N_tx), dtype=complex)
    distances = np.zeros((N_rx, N_tx))
    retards = np.zeros((N_rx, N_tx))
    
    for i in range(N_rx):
        for j in range(N_tx):
            d_ij = np.linalg.norm(rx_positions[:,i] - tx_positions[:,j])
            distances[i, j] = d_ij
            tau_ij = d_ij / c
            retards[i, j] = tau_ij
            
            attenuation = wavelength / (4 * np.pi * d_ij)
            phase = -2 * np.pi * freq * tau_ij
            H[i, j] = attenuation * np.exp(1j * phase)
    
    return H, distances, retards
