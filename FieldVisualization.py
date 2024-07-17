import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


#import tensorflow as tf
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.io import savemat

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from keras.models import Model

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
    plt.close()

def GetField(aE,aH, wavelength, d_theta, d_phi):
    theta = np.arange(0,np.pi, d_theta)
    phi = np.arange(0,2*np.pi, d_phi)
    dims = [len(theta), len(phi)]
    PHI,THETA = np.meshgrid(phi, theta)
    wavelength = wavelength*10**(-9)
    params = {
        "l":1,
        "m":1,
        "phi":PHI,
        "theta":THETA,
        "r":300*10**(-6),
        "k":2*np.pi / wavelength,
        "step": [d_theta, d_phi],
        "aE":aE[0],
        "aH":aH[0]}
    E0, H0 = GetFieldComponent(params)
    params["l"] = 2
    params["aE"] = aE[1]; params["aH"] = aH[1]
    E1, H1 = GetFieldComponent(params)
    params["m"] = 2
    params["aE"] = aE[2]; params["aH"] = aH[2]
    E2, H2 = GetFieldComponent(params)
    E0x, E0y, E0z = E0
    H0x, H0y, H0z = H0
    E1x, E1y, E1z = E1
    H1x, H1y, H1z = H1
    E2x, E2y, E2z = E2
    H2x, H2y, H2z = H2
    E = [E0x + E1x + E2x, E0y + E1y + E2y, E0z + E1z + E2z]
    H = [H0x + H1x + H2x, H0y + H1y + H2y, H0z + H1z + H2z]
    return [E, H]

def generate_predictions(shapes,model_dir):
    print(model_dir)
    with tf.device('/cpu:0'):
        current_model = tf.keras.models.load_model(model_dir)
        return current_model.predict(np.asarray(shapes))

def load_testing_data(term,kind):
        with open("/media/work/evan/deep_learning_data/{}_{}_cutoff1000_test.pkl".format(term,kind), 'rb') as input:
            return pkl.load(input)

def get_multipole_term(field_type, component, model):
    model_path = model
    shapes = load_testing_data("X_img", "{}_{}".format(field_type, component))
    spectra = load_testing_data("y_labels", "{}_{}".format(field_type, component))
    spectra = spectra ** 0.25
    labels = load_testing_data("X_info", "{}_{}".format(field_type, component))
    #predictions = generate_predictions(shapes,model_path)
    predictions = spectra

    return spectra, labels, shapes, predictions

def GetFieldSubplot(Field, FigShape, Location, Label, Axes = [False, False]):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.set_title(Label)
    ax.get_xaxis().set_visible(Axes[0])
    ax.get_yaxis().set_visible(Axes[1])
    ax.set_ylabel("θ")
    ax.set_xlabel("ɸ")
    ax.imshow(Field,extent=[0,2*np.pi,0,np.pi])
    return ax


def PlotComponent(Component, Wavelengths, Label, Color, WavelengthIDX, MarkerLabel):
    wl = round(Wavelengths[WavelengthIDX] * 1E9)
    ax = plt.plot(Wavelengths * 1E9, Component, label=Label, color=Color, linewidth = 2)
    if MarkerLabel == True:
        ax = plt.plot(Wavelengths[WavelengthIDX] * 1E9, Component[WavelengthIDX], '.', color = "black", markersize = 10, label = "λ: {} nm".format(wl))
        
    else:
        ax = plt.plot(Wavelengths[WavelengthIDX] * 1E9, Component[WavelengthIDX], '.', color = "black", markersize = 10)
    return ax

def GetComponentSubplot(Component, Wavelengths, WavelengthIDX, FigShape, Location):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=2)
    ax.plot()
    ax.set_xlabel("λ [nm]")
    ax = PlotComponent(Component[0], Wavelengths, '$aE_{1}^{1}$ SIM', "darkred", WavelengthIDX, False)
    ax = PlotComponent(Component[3], Wavelengths, '$aE_{1}^{1}$ CNN', "red", WavelengthIDX, False)
    ax = PlotComponent(Component[1], Wavelengths, '$aE_{2}^{1}$ SIM', "navy", WavelengthIDX, False)
    ax = PlotComponent(Component[4], Wavelengths, '$aE_{2}^{1}$ CNN', "cornflowerblue", WavelengthIDX, False)
    ax = PlotComponent(Component[2], Wavelengths, '$aE_{2}^{2}$ SIM', "darkgreen", WavelengthIDX, False)
    ax = PlotComponent(Component[5], Wavelengths, '$aE_{2}^{2}$ CNN', "springgreen", WavelengthIDX, True)
    
    plt.legend()    
    return ax

def GetShapeSubplot(Shape, FigShape, Location):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.imshow(Shape)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax





electric_l1_m1_component, wavelengths, profiles, predicted_electric_l1_m1_component = get_multipole_term("electric", "l1_m1", "/media/work/evan/deep_learning_data/trained_models/electric_dipole_1000epoch")
electric_l2_m1_component, _, _, predicted_electric_l2_m1_component   = get_multipole_term("electric", "l2_m1", "/media/work/evan/deep_learning_data/trained_models/electric_quadl2m1_1000epoch")
electric_l2_m2_component, _, _, predicted_electric_l2_m2_component   = get_multipole_term("electric", "l2_m2", "/media/work/evan/deep_learning_data/trained_models/electric_quadl2m2_1000epoch")
magnetic_l1_m1_component, _, _, predicted_magnetic_l1_m1_component   = get_multipole_term("magnetic", "l1_m1", "/media/work/evan/deep_learning_data/trained_models/magnetic_dipole_1000epoch")
magnetic_l2_m1_component, _, _, predicted_magnetic_l2_m1_component  = get_multipole_term("magnetic", "l2_m1", "/media/work/evan/deep_learning_data/trained_models/magnetic_quadl2m1_1000epoch")
magnetic_l2_m2_component, _, _, predicted_magnetic_l2_m2_component  = get_multipole_term("magnetic", "l2_m2", "/media/work/evan/deep_learning_data/trained_models/magnetic_quadl2m2_1000epoch")

wavelengths = wavelengths[0]
ShapeIDX = 2
WavelengthIDX = 50

aE = [electric_l1_m1_component[ShapeIDX, WavelengthIDX],
        electric_l2_m1_component[ShapeIDX, WavelengthIDX],
        electric_l2_m2_component[ShapeIDX, WavelengthIDX]]


aH = [magnetic_l1_m1_component[ShapeIDX, WavelengthIDX],
        magnetic_l2_m1_component[ShapeIDX, WavelengthIDX],
        magnetic_l2_m2_component[ShapeIDX, WavelengthIDX]]
        
dTheta = 0.01
dPhi = 0.01 

E, H = GetField(aE,aH, wavelengths[WavelengthIDX], dTheta, dPhi)
SimEx, SimEy, SimEz = E
SimHx, SimHy, SimHz = H


aE = [predicted_electric_l1_m1_component[ShapeIDX, WavelengthIDX],
        predicted_electric_l2_m1_component[ShapeIDX, WavelengthIDX],
        predicted_electric_l2_m2_component[ShapeIDX, WavelengthIDX]]


aH = [predicted_magnetic_l1_m1_component[ShapeIDX, WavelengthIDX],
        predicted_magnetic_l2_m1_component[ShapeIDX, WavelengthIDX],
        predicted_magnetic_l2_m2_component[ShapeIDX, WavelengthIDX]]
        
dTheta = 0.01
dPhi = 0.01 

E, H = GetField(aE,aH, wavelengths[WavelengthIDX], dTheta, dPhi)
CnnEx, CnnEy, CnnEz = E
CnnHx, CnnHy, CnnHz = H


aEfig = [electric_l1_m1_component[ShapeIDX, :],
        electric_l2_m1_component[ShapeIDX, :],
        electric_l2_m2_component[ShapeIDX, :],
        predicted_electric_l1_m1_component[ShapeIDX, :],
        predicted_electric_l2_m1_component[ShapeIDX, :],
        predicted_electric_l2_m2_component[ShapeIDX, :]]

aHfig = [magnetic_l1_m1_component[ShapeIDX, :],
        magnetic_l2_m1_component[ShapeIDX, :],
        magnetic_l2_m2_component[ShapeIDX, :],
        predicted_magnetic_l1_m1_component[ShapeIDX, :],
        predicted_magnetic_l2_m1_component[ShapeIDX, :],
        predicted_magnetic_l2_m2_component[ShapeIDX, :]]


import matplotlib
font = {'family' : 'sans serif',
        'weight' : 'bold',
        'size'   : 20}

Representation = "Phase"
FieldType = "Electric"
FigSize = [15, 30]

def PlotField(Field, Components, FieldType, Representation, Shape, ShapeIDX, Wavelengths, WavelengthIDX):
    FigShape = (3,3)
    if Representation == "Magnitude":
        plt.set_cmap(plt.get_cmap('inferno'))
        enclosure = ["|", "|"]
        for i in range(6):
            Field[i] = magnitude(Field[i])

    elif Representation == "Phase":
        plt.set_cmap(plt.get_cmap('hsv'))
        enclosure = ["arg(", ")"]
        for i in range(6):
            Field[i] = phase(Field[i])

    SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz = Field

    fig = plt.figure()
    fig.set_figheight(FigSize[0])
    fig.set_figwidth(FigSize[1])
    fig.suptitle(f"Comparison of Predicted and Simulated Multipole Components in {FieldType} Far-Field Generation", fontsize=24)
    ax1 = GetFieldSubplot(SimEx, FigShape, (0,0), f"{enclosure[0]}$a{FieldType[0]}_{1}^{1}${enclosure[1]} SIM", Axes = [False, True])
    ax2 = GetFieldSubplot(SimEy, FigShape, (0,1), f"{enclosure[0]}$a{FieldType[0]}_{2}^{1}${enclosure[1]} SIM")
    ax3 = GetFieldSubplot(SimEz, FigShape, (0,2), f"{enclosure[0]}$a{FieldType[0]}_{2}^{2}${enclosure[1]} SIM")
    ax4 = GetFieldSubplot(CnnEx, FigShape, (1,0), f"{enclosure[0]}$a{FieldType[0]}_{1}^{1}${enclosure[1]} CNN", Axes = [True, True])
    ax5 = GetFieldSubplot(CnnEy, FigShape, (1,1), f"{enclosure[0]}$a{FieldType[0]}_{2}^{1}${enclosure[1]} CNN", Axes = [True, False])
    ax6 = GetFieldSubplot(CnnEz, FigShape, (1,2), f"{enclosure[0]}$a{FieldType[0]}_{2}^{2}${enclosure[1]} CNN", Axes = [True, False])
    ax7 = GetComponentSubplot(Components, Wavelengths, WavelengthIDX, FigShape, (2,0))
    plt.set_cmap(plt.get_cmap('inferno'))
    ax8 = GetShapeSubplot(profiles[ShapeIDX], FigShape, (2,2))
    plt.savefig(f"Electric{Representation}-Shape{ShapeIDX}-Wavelength{round(Wavelengths[WavelengthIDX])*1E9}.png")
    plt.close()



PlotField(Field=[SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz],
    Components = aEfig,
    FieldType = "Electric",
    Representation = "Phase",
    Shape = profiles,
    ShapeIDX = ShapeIDX,
    Wavelengths = wavelengths,
    WavelengthIDX = WavelengthIDX
    )

PlotField(Field=[SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz],
    Components = aEfig,
    FieldType = "Electric",
    Representation = "Magnitude",
    Shape = profiles,
    ShapeIDX = ShapeIDX,
    Wavelengths = wavelengths,
    WavelengthIDX = WavelengthIDX
    )

PlotField(Field=[SimHx, SimHy, SimHz, CnnHx, CnnHy, CnnHz],
    Components = aEfig,
    FieldType = "Magnetic",
    Representation = "Phase",
    Shape = profiles,
    ShapeIDX = ShapeIDX,
    Wavelengths = wavelengths,
    WavelengthIDX = WavelengthIDX
    )

PlotField(Field=[SimHx, SimHy, SimHz, CnnHx, CnnHy, CnnHz],
    Components = aEfig,
    FieldType = "Magnetic",
    Representation = "Magnitude",
    Shape = profiles,
    ShapeIDX = ShapeIDX,
    Wavelengths = wavelengths,
    WavelengthIDX = WavelengthIDX
    )