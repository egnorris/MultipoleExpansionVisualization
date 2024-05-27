import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


#import tensorflow as tf
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.special as sp
from matplotlib.widgets import Slider
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
    SphHarm = sp.sph_harm(params["m"],params["l"],params["phi"],params["theta"])
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

def magnitude(z):
  x = np.real(z)
  y = np.imag(z)
  return np.sqrt(x**2 + y**2)

def phase(z):
  x = np.real(z)
  y = np.imag(z)
  return np.arctan2(y,x)

def FieldPlotting(E,H, filename):
    E_x, E_y, E_z = E
    H_x, H_y, H_z = H
    E_x = np.nan_to_num(E_x); E_y = np.nan_to_num(E_y); E_z = np.nan_to_num(E_z)
    H_x = np.nan_to_num(H_x); H_y = np.nan_to_num(H_y); H_z = np.nan_to_num(H_z)
    m = np.max([magnitude(E_x), magnitude(E_y), magnitude(E_z), magnitude(H_x), magnitude(H_y), magnitude(H_z)])
    fig, ax = plt.subplots(2,6,figsize=(60,10))
    plt.set_cmap(plt.get_cmap('inferno'))
    ax[0, 0].set_title("E_x Magnitude"); ax[0, 0].set_ylabel("θ"); ax[0, 0].get_xaxis().set_visible(False)
    ax[0, 1].set_title("H_x Magnitude"); ax[0, 1].get_xaxis().set_visible(False); ax[0, 1].get_yaxis().set_visible(False)
    ax[0, 2].set_title("E_y Magnitude"); ax[0, 2].get_xaxis().set_visible(False); ax[0, 2].get_yaxis().set_visible(False)
    ax[0, 3].set_title("H_y Magnitude"); ax[0, 3].get_xaxis().set_visible(False); ax[0, 3].get_yaxis().set_visible(False)
    ax[0, 4].set_title("E_z Magnitude"); ax[0, 4].get_xaxis().set_visible(False); ax[0, 4].get_yaxis().set_visible(False)
    ax[0, 5].set_title("H_z Magnitude"); ax[0, 5].get_xaxis().set_visible(False); ax[0, 5].get_yaxis().set_visible(False)
    ax[1, 0].set_title("E_x Phase"); ax[1, 0].set_xlabel("ɸ"); ax[1, 0].set_ylabel("θ")
    ax[1, 1].set_title("H_x Phase"); ax[1, 1].get_yaxis().set_visible(False); ax[1, 1].set_xlabel("ɸ")
    ax[1, 2].set_title("E_y Phase"); ax[1, 2].get_yaxis().set_visible(False); ax[1, 2].set_xlabel("ɸ")
    ax[1, 3].set_title("H_y Phase"); ax[1, 3].get_yaxis().set_visible(False); ax[1, 3].set_xlabel("ɸ")
    ax[1, 4].set_title("E_z Phase"); ax[1, 4].get_yaxis().set_visible(False); ax[1, 4].set_xlabel("ɸ")
    ax[1, 5].set_title("H_z Phase"); ax[1, 5].get_yaxis().set_visible(False); ax[1, 5].set_xlabel("ɸ")
    bounds = [0,2*np.pi,0,np.pi]
    im0 = ax[0, 0].imshow(magnitude(E_x) / m,extent=bounds); im1 = ax[0, 1].imshow(magnitude(H_x) / m,extent=bounds)
    im2 = ax[0, 2].imshow(magnitude(E_y) / m,extent=bounds); im3 = ax[0, 3].imshow(magnitude(H_y) / m,extent=bounds)
    im4 = ax[0, 4].imshow(magnitude(E_z) / m,extent=bounds); im5 = ax[0, 5].imshow(magnitude(H_z) / m,extent=bounds)
    plt.colorbar(im5, ax=ax[0, 5], orientation="vertical")
    plt.set_cmap(plt.get_cmap('hsv'))
    im0 = ax[1, 0].imshow(phase(E_x),extent=bounds); im1 = ax[1, 1].imshow(phase(H_x),extent=bounds)
    im2 = ax[1, 2].imshow(phase(E_y),extent=bounds); im3 = ax[1, 3].imshow(phase(H_y),extent=bounds)
    im4 = ax[1, 4].imshow(phase(E_z),extent=bounds); im5 = ax[1, 5].imshow(phase(H_z),extent=bounds)
    im0.set_clim(-np.pi, np.pi); im1.set_clim(-np.pi, np.pi); im2.set_clim(-np.pi, np.pi)
    im3.set_clim(-np.pi, np.pi); im4.set_clim(-np.pi, np.pi); im5.set_clim(-np.pi, np.pi)
    plt.colorbar(im5, ax=ax[1, 5], orientation="vertical", ticks = [-3, -2, -1, 0, 1, 2, 3])
    plt.savefig(filename, dpi = 300)




def GetField(aE0, aE1, aE2, wl):
    d_theta = 0.03
    d_phi = 0.03
    theta = np.arange(0,np.pi, d_theta)
    phi = np.arange(0,2*np.pi, d_phi)
    dims = [len(theta), len(phi)]
    PHI,THETA = np.meshgrid(phi, theta)
    wavelength = wl*10**(-9)
    params = {
    "l":1,
    "m":1,
    "phi":PHI,
    "theta":THETA,
    "r":300*10**(-6),
    "k":2*np.pi / wavelength,
    "step": [d_theta, d_phi],
    "aE":aE0,
    "aH":aE0
    }
    E0, _ = GetFieldComponent(params)
    params = {
    "l":2,
    "m":1,
    "phi":PHI,
    "theta":THETA,
    "r":300*10**(-6),
    "k":2*np.pi / wavelength,
    "step": [d_theta, d_phi],
    "aE":aE1,
    "aH":aE1
    }
    E1, _ = GetFieldComponent(params)
    params = {
    "l":2,
    "m":2,
    "phi":PHI,
    "theta":THETA,
    "r":300*10**(-6),
    "k":2*np.pi / wavelength,
    "step": [d_theta, d_phi],
    "aE":aE2,
    "aH":aE2
    }
    E2, _ = GetFieldComponent(params)
    E0x, E0y, E0z = E0
    E1x, E1y, E1z = E1
    E2x, E2y, E2z = E2
    E = [E0x + E1x + E2x, E0y + E1y + E2y, E0z + E1z + E2z]
    return E

   
idx0 = 0
fig, ax = plt.subplots(2,3)
plt.set_cmap(plt.get_cmap('inferno'))
E_x, E_y, E_z = GetField(0.5, 0, 0, 500)
l1 = ax[0,0].imshow(magnitude(E_x))
m1 = ax[0,1].imshow(magnitude(E_y))
n1 = ax[0,2].imshow(magnitude(E_z))
plt.set_cmap(plt.get_cmap('hsv'))
l2 = ax[1,0].imshow(phase(E_x))
m2 = ax[1,1].imshow(phase(E_y))
n2 = ax[1,2].imshow(phase(E_z))
a0idx = plt.axes([0.25, 0.10, 0.65, 0.03])
a1idx = plt.axes([0.25, 0.075, 0.65, 0.03])
a2idx = plt.axes([0.25, 0.05, 0.65, 0.03])
awlidx = plt.axes([0.25, 0.025, 0.65, 0.03])
slida0 = Slider(
    ax=a0idx,
    label="First Dipole",
    valmin=0,
    valmax=1,
    valinit=0.5,
    valfmt="%f"
)

slida1 = Slider(
    ax=a1idx,
    label="First Quadrupole",
    valmin=0,
    valmax=1,
    valinit=0,
    valfmt="%f"
)

slida2 = Slider(
    ax=a2idx,
    label="Second Quadrupole",
    valmin=0,
    valmax=1,
    valinit=0,
    valfmt="%f"
)

slidwl = Slider(
    ax=awlidx,
    label="Wavelength",
    valmin=400,
    valmax=800,
    valinit=500,
    valstep=1,
    valfmt="%d nm"
)

def update(val):
    aE0 = slida0.val
    aE1 = slida1.val
    aE2 = slida2.val
    wl = slidwl.val
    E_x, E_y, E_z = GetField(aE0, aE1, aE2, wl)
    l1.set_data(magnitude(E_x))
    m1.set_data(magnitude(E_y))
    n1.set_data(magnitude(E_z))
    l2.set_data(phase(E_x))
    m2.set_data(phase(E_y))
    n2.set_data(phase(E_z))
    ax[0,0].set_title("Ex Magntiude")
    ax[0,1].set_title("Ey Magntiude")
    ax[0,2].set_title("Ez Magntiude")
    ax[1,0].set_title("Ex Phase")
    ax[1,1].set_title("Ey Phase")
    ax[1,2].set_title("Ez Phase")
    fig.canvas.draw_idle()
slida0.on_changed(update)
slida1.on_changed(update)
slida2.on_changed(update)
slidwl.on_changed(update)
plt.show()