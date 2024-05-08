import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


#import tensorflow as tf
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.special as sp

def FiniteDifference(f,coord, order, params):
    N_theta, N_phi = f.shape
    result = f * 0
    step_size = params["step"]
    d_theta = step_size[0]
    d_phi = step_size[1]
    if coord == "theta" and order == 1:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta+1,i_phi] - f[i_theta-1,i_phi])/(2*d_theta)
        return result
    elif coord == "phi" and order == 1:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta,i_phi+1] - f[i_theta,i_phi-1])/(2*d_phi)
        return result
    elif coord == "theta" and order == 2:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta+1,i_phi] - 2*f[i_theta,i_phi] + f[i_theta-1,i_phi])/(d_theta*d_theta)
        return result
    elif coord == "phi" and order == 2:
        for i_theta in range(1,N_theta-1):
            for i_phi in range(1,N_phi-1):
                result[i_theta,i_phi] = (f[i_theta,i_phi+1] - 2*f[i_theta,i_phi] + f[i_theta,i_phi-1])/(d_phi*d_phi)
        return result

def hankel(params,derivative,conj):
    k = params["k"]
    r = params["r"]
    l = params["l"]
    if conj == True:
        i = -1j
    else:
        i = 1j
    if derivative == False:
        return (-i)**(l+1)*np.exp(i*k*r) / (k*r)
    elif derivative == True:
        return (-i)**(l)*np.exp(i*k*r)*(r+i) / (k*r*r)

def GetSphericalComponent(params, SphHarmDict, FieldType, Component):
    hbar = 1.054571817*10**(-34)
    const = hbar / np.sqrt(params["l"]*(params["l"]+1))

    if Component == "theta":

        if FieldType == "electric":

            Field = params["aE"]*1j*hankel(params,True,False)*SphHarmDict["D"]["1"]["theta"]/params["k"]
            Field = Field + params["aH"]*hankel(params,False,True)*SphHarmDict["D"]["1"]["phi"]/np.sin(params["theta"])
            Field = Field * -1j * const
            return Field
        
        elif FieldType == "magnetic":

            Field = params["aH"]* -1j *hankel(params,True,True)*SphHarmDict["D"]["1"]["theta"]/params["k"]
            Field = Field + params["aE"]*hankel(params,False,False)*SphHarmDict["D"]["1"]["phi"]/np.sin(params["theta"])
            Field = Field * 1j * const
            return Field

    elif Component == "phi":

        if FieldType == "electric":

            Field = params["aE"]*1j*hankel(params,True,False)*SphHarmDict["D"]["1"]["phi"]/(params["k"] * np.sin(params["theta"]))
            Field = Field + params["aH"]*hankel(params, False, True)*SphHarmDict["D"]["1"]["theta"]
            Field = Field * -1j * const
            return Field
        
        elif FieldType == "magnetic":

            Field = params["aH"]*1j*hankel(params,True,True)*SphHarmDict["D"]["1"]["phi"]/(params["k"] * np.sin(params["theta"]))
            Field = Field + params["aE"]*hankel(params, False, False)*SphHarmDict["D"]["1"]["theta"]
            Field = Field * -1j * const
            return Field

    elif Component == "r":

        if FieldType == "electric":
            Field = params["aE"]*1j*hankel(params,False,False)/(params["k"]*params["r"])
            Field = Field * (SphHarmDict["D"]["2"]["theta"] + SphHarmDict["D"]["2"]["phi"]/(np.sin(params["theta"])**2))
            Field = -1j * const * Field
            return Field

        elif FieldType == "magnetic":
            Field = params["aE"]*1j*hankel(params,False,True)/(params["k"]*params["r"])
            Field = Field * (SphHarmDict["D"]["2"]["theta"] + SphHarmDict["D"]["2"]["phi"]/(np.sin(params["theta"])**2))
            Field = 1j * const * Field
            return Field

def cartesian_projection(F,params):
    PHI = params["phi"]
    THETA = params["theta"]
    F_r , F_theta, F_phi = F

    x_hat_r = np.sin(THETA)*np.cos(PHI); x_hat_theta = np.cos(THETA)*np.cos(PHI); x_hat_phi = -np.sin(PHI)
    z_hat_r = np.sin(THETA)*np.sin(PHI); z_hat_theta = np.cos(THETA)*np.sin(PHI); z_hat_phi = np.cos(PHI)
    y_hat_r = np.cos(THETA); y_hat_theta = -np.sin(THETA); y_hat_phi = 0

    F_x = (x_hat_r * F_r) + (x_hat_theta * F_theta) + (x_hat_phi * F_phi)
    F_y = (y_hat_r * F_r) + (y_hat_theta * F_theta) + (y_hat_phi * F_phi)
    F_z = (z_hat_r * F_r) + (z_hat_theta * F_theta) + (z_hat_phi * F_phi)
    return [F_x, F_y, F_z]

def GetFieldComponent(params):
    SphHarm = sp.sph_harm(params["m"],params["l"],params["theta"],params["phi"])
    #Calculate Derivatives of Spherical Harmonic, Y
    SphHarmFirstDerivPhi   = FiniteDifference(SphHarm,'phi', 1, params)
    SphHarmFirstDerivTheta  = FiniteDifference(SphHarm,'theta', 1, params)
    SphHarmSecondDerivPhi   = FiniteDifference(SphHarm,'phi', 2, params)
    SphHarmSecondDerivTheta = FiniteDifference(SphHarm,'theta', 2, params)
    FirstDerivatives = {"phi":SphHarmFirstDerivPhi, "theta":SphHarmFirstDerivTheta}
    SecondDerivatives = {"phi":SphHarmSecondDerivPhi, "theta":SphHarmSecondDerivTheta}
    DerivativeDict = {"1": FirstDerivatives, "2": SecondDerivatives}
    SphHarmDict = {"Y": SphHarm, "D": DerivativeDict}

    ElecFieldR     = GetSphericalComponent(params, SphHarmDict, "electric", "r")
    ElecFieldTheta = GetSphericalComponent(params, SphHarmDict, "electric", "theta")
    ElecFieldPhi   = GetSphericalComponent(params, SphHarmDict, "electric", "phi")
    MagnFieldR     = GetSphericalComponent(params, SphHarmDict, "magnetic", "r")
    MagnFieldTheta = GetSphericalComponent(params, SphHarmDict, "magnetic", "theta")
    MagnFieldPhi   = GetSphericalComponent(params, SphHarmDict, "magnetic", "phi")

    ElecFieldCart = cartesian_projection([ElecFieldR, ElecFieldTheta, ElecFieldPhi], params)
    MagnFeildCart = cartesian_projection([MagnFieldR, MagnFieldTheta, MagnFieldPhi], params)

    return [ElecFieldCart, MagnFeildCart]




d_theta = 0.01
d_phi = 0.01
theta = phi = np.arange(0,np.pi, d_theta)
phi = np.arange(0,2*np.pi, d_phi)
dims = [len(theta), len(phi)]
PHI,THETA = np.meshgrid(phi, theta)
params = {
    "l":1,
    "m":1,
    "phi":PHI,
    "theta":THETA,
    "r":1,
    "k":1,
    "step": [d_theta, d_phi],
    "aE":1,
    "aH":1
}
GetFieldComponent(params)


