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

import imageio.v2 as imagio

import matplotlib
font = {'family' : 'sans serif',
        'size'   : 18}
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rc('font', **font)
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
def PercentDifference(A,B):
        return 100*abs(A-B)/((A+B)/2)

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
    predictions = generate_predictions(shapes,model_path)
    #for testing formatting without the need to run predictions every time
    #predictions = spectra
    return spectra, labels, shapes, predictions

def GetFieldSubplot(Field, FigShape, Location, Label, Axes = [False, False]):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.set_title(Label)
    ax.get_xaxis().set_visible(Axes[0])
    ax.get_yaxis().set_visible(Axes[1])
    ax.set_ylabel("θ")
    ax.set_xlabel("ɸ")
    p = ax.imshow(Field,extent=[0,2*np.pi,0,np.pi])
    return ax
def GetErrorSubplot(Field, FigShape, Location, Label, Axes = [False, False]):
    #cmap = LinearSegmentedColormap.from_list("", ["blue","red"], N = 2)
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.set_title(Label)
    ax.get_xaxis().set_visible(Axes[0])
    ax.get_yaxis().set_visible(Axes[1])
    ax.set_ylabel("θ")
    ax.set_xlabel("ɸ")
    p = ax.imshow(Field,extent=[0,2*np.pi,0,np.pi], cmap = "binary")
    p.set_clim(0,100)
    return ax


def PlotComponent(Component, Wavelengths, Label, Color, WavelengthIDX, MarkerLabel):
    wl = round(Wavelengths[WavelengthIDX] * 1E9)
    ax = plt.plot(Wavelengths * 1E9, Component, label=Label, color=Color, linewidth = 3)
    if MarkerLabel == True:
        ax = plt.plot(Wavelengths[WavelengthIDX] * 1E9, Component[WavelengthIDX], '.', color = "black", markersize = 20, label = "λ: {} nm".format(wl))
        
    else:
        ax = plt.plot(Wavelengths[WavelengthIDX] * 1E9, Component[WavelengthIDX], '.', color = "black", markersize = 20)
    return ax

def GetComponentSubplot(Component, Wavelengths, WavelengthIDX, FigShape, Location, FieldType):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=2)
    ax.plot()
    ax.set_xlabel("λ [nm]")
    ax = PlotComponent(Component[0], Wavelengths, f"$a_{{1,1}}^{FieldType[0]}$ SIM", "darkred", WavelengthIDX, False)
    ax = PlotComponent(Component[3], Wavelengths, f"$a_{{1,1}}^{FieldType[0]}$ CNN", "red", WavelengthIDX, False)
    ax = PlotComponent(Component[1], Wavelengths, f"$a_{{2,1}}^{FieldType[0]}$ SIM", "navy", WavelengthIDX, False)
    ax = PlotComponent(Component[4], Wavelengths, f"$a_{{2,1}}^{FieldType[0]}$ CNN", "cornflowerblue", WavelengthIDX, False)
    ax = PlotComponent(Component[2], Wavelengths, f"$a_{{2,2}}^{FieldType[0]}$ SIM", "darkgreen", WavelengthIDX, False)
    ax = PlotComponent(Component[5], Wavelengths, f"$a_{{2,2}}^{FieldType[0]}$ CNN", "springgreen", WavelengthIDX, True)
    
    plt.legend()    
    return ax


def GetShapeSubplot(Shape, FigShape, Location):
    ax = plt.subplot2grid(shape = FigShape, loc=Location, colspan=1)
    ax.imshow(Shape)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def PlotField(Field, Components, FieldType, Representation, Shape, ShapeIDX, Wavelengths, WavelengthIDX, SavePath, FigSize):
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
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    ax1 = GetFieldSubplot(SimEx, FigShape, (0,0), f"{enclosure[0]}${FieldType[0]}_{{x}}${enclosure[1]} SIM", Axes = [False, True])
    ax2 = GetFieldSubplot(SimEy, FigShape, (0,1), f"{enclosure[0]}${FieldType[0]}_{{y}}${enclosure[1]} SIM")
    ax3 = GetFieldSubplot(SimEz, FigShape, (0,2), f"{enclosure[0]}${FieldType[0]}_{{z}}${enclosure[1]} SIM")
    ax4 = GetFieldSubplot(CnnEx, FigShape, (1,0), f"{enclosure[0]}${FieldType[0]}_{{x}}${enclosure[1]} CNN", Axes = [True, True])
    ax5 = GetFieldSubplot(CnnEy, FigShape, (1,1), f"{enclosure[0]}${FieldType[0]}_{{y}}${enclosure[1]} CNN", Axes = [True, False])
    ax6 = GetFieldSubplot(CnnEz, FigShape, (1,2), f"{enclosure[0]}${FieldType[0]}_{{z}}${enclosure[1]} CNN", Axes = [True, False])
    ax7 = GetComponentSubplot(Components, Wavelengths, WavelengthIDX, FigShape, (2,0), FieldType)
    plt.set_cmap(plt.get_cmap('inferno'))
    ax8 = GetShapeSubplot(profiles[ShapeIDX], FigShape, (2,2))
    if FieldType == "Hagnetic":
        FieldType = "Magnetic"
    plt.savefig(f"{SavePath}{FieldType}{Representation}-{round(Wavelengths[WavelengthIDX]*1E9)}nm.png")
    plt.close()

def makemovie(FieldType, PlotType, ShapeIDX, Wavelengths, WavelengthRange=[0, 101, 1]):
    with imageio.get_writer(f"/media/work/evan/MultipoleFieldImageData/movie/{FieldType}{PlotType}-Shape{ShapeIDX}.gif", mode='I') as writer:
        for i in np.arange(WavelengthRange[0],WavelengthRange[1],WavelengthRange[2]):
            wl = round(Wavelengths[i]*1E9)
            image = imageio.imread(f"/media/work/evan/MultipoleFieldImageData/temp/{FieldType}{PlotType}-{wl}nm.png")
            writer.append_data(image)

def PlotError(Field, Components, FieldType, Representation, Shape, ShapeIDX, Wavelengths, WavelengthIDX, SavePath, FigSize):
    FigShape = (3,3)

    SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz = Field

    

    fig = plt.figure()
    fig.set_figheight(FigSize[0])
    fig.set_figwidth(FigSize[1])
    fig.suptitle(f"Pixel Difference Between FDTD and CNN Prediction for {FieldType} Far-Field", fontsize=24)
    if FieldType == "Magnetic":
        FieldType = "Hagnetic"
    ax1 = GetErrorSubplot(PercentDifference(magnitude(SimEx),magnitude(CnnEx)), FigShape, (0,0), f"Magnitude ${FieldType[0]}_{{x}}$ Difference")
    ax2 = GetErrorSubplot(PercentDifference(magnitude(SimEy),magnitude(CnnEy)), FigShape, (0,1), f"Magnitude ${FieldType[0]}_{{y}}$ Difference")
    ax3 = GetErrorSubplot(PercentDifference(magnitude(SimEz),magnitude(CnnEz)), FigShape, (0,2), f"Magnitude ${FieldType[0]}_{{z}}$ Difference")
    ax4 = GetErrorSubplot(PercentDifference(phase(SimEx),phase(CnnEx)), FigShape, (1,0), f"Phase ${FieldType[0]}_{{x}}$ Difference")
    ax5 = GetErrorSubplot(PercentDifference(phase(SimEy),phase(CnnEy)), FigShape, (1,1), f"Phase ${FieldType[0]}_{{y}}$ Difference")
    ax6 = GetErrorSubplot(PercentDifference(phase(SimEz),phase(CnnEz)), FigShape, (1,2), f"Phase ${FieldType[0]}_{{z}}$ Difference")
    plt.set_cmap(plt.get_cmap('inferno'))
    ax7 = GetComponentSubplot(Components, Wavelengths, WavelengthIDX, FigShape, (2,0), FieldType)
    ax8 = GetShapeSubplot(profiles[ShapeIDX], FigShape, (2,2))
    if FieldType == "Hagnetic":
        FieldType = "Magnetic"
    plt.savefig(f"{SavePath}{FieldType}Error-{round(Wavelengths[WavelengthIDX]*1E9)}nm.png")
    #plt.show()
    plt.close()


electric_l1_m1_component, wavelengths, profiles, predicted_electric_l1_m1_component = get_multipole_term("electric", "l1_m1", "/media/work/evan/deep_learning_data/trained_models/electric_dipole_1000epoch")
electric_l2_m1_component, _, _, predicted_electric_l2_m1_component   = get_multipole_term("electric", "l2_m1", "/media/work/evan/deep_learning_data/trained_models/electric_quadl2m1_1000epoch")
electric_l2_m2_component, _, _, predicted_electric_l2_m2_component   = get_multipole_term("electric", "l2_m2", "/media/work/evan/deep_learning_data/trained_models/electric_quadl2m2_1000epoch")
magnetic_l1_m1_component, _, _, predicted_magnetic_l1_m1_component   = get_multipole_term("magnetic", "l1_m1", "/media/work/evan/deep_learning_data/trained_models/magnetic_dipole_1000epoch")
magnetic_l2_m1_component, _, _, predicted_magnetic_l2_m1_component  = get_multipole_term("magnetic", "l2_m1", "/media/work/evan/deep_learning_data/trained_models/magnetic_quadl2m1_1000epoch")
magnetic_l2_m2_component, _, _, predicted_magnetic_l2_m2_component  = get_multipole_term("magnetic", "l2_m2", "/media/work/evan/deep_learning_data/trained_models/magnetic_quadl2m2_1000epoch")

wavelengths = wavelengths[0]
WavelengthRange=[0, 101, 5]
for ShapeIDX in range(len(profiles)):
    print(f"\nCurrent Shape: {ShapeIDX}")
    for WavelengthIDX in np.arange(WavelengthRange[0],WavelengthRange[1],WavelengthRange[2]):
        print(f"{round(wavelengths[WavelengthIDX]*1E9)} nm")

        aE = [electric_l1_m1_component[ShapeIDX, WavelengthIDX],
                electric_l2_m1_component[ShapeIDX, WavelengthIDX],
                electric_l2_m2_component[ShapeIDX, WavelengthIDX]]


        aH = [magnetic_l1_m1_component[ShapeIDX, WavelengthIDX],
                magnetic_l2_m1_component[ShapeIDX, WavelengthIDX],
                magnetic_l2_m2_component[ShapeIDX, WavelengthIDX]]
                
        dTheta = 0.05
        dPhi = 0.05 
        E, H = GetField(aE,aH, wavelengths[WavelengthIDX], dTheta, dPhi)
        SimEx, SimEy, SimEz = E
        SimHx, SimHy, SimHz = H


        aE = [predicted_electric_l1_m1_component[ShapeIDX, WavelengthIDX],
                predicted_electric_l2_m1_component[ShapeIDX, WavelengthIDX],
                predicted_electric_l2_m2_component[ShapeIDX, WavelengthIDX]]


        aH = [predicted_magnetic_l1_m1_component[ShapeIDX, WavelengthIDX],
                predicted_magnetic_l2_m1_component[ShapeIDX, WavelengthIDX],
                predicted_magnetic_l2_m2_component[ShapeIDX, WavelengthIDX]]
                

        E, H = GetField(aE,aH, wavelengths[WavelengthIDX], dTheta, dPhi)
        CnnEx, CnnEy, CnnEz = E
        CnnHx, CnnHy, CnnHz = H


        ElectricComponents = [electric_l1_m1_component[ShapeIDX, :],
                electric_l2_m1_component[ShapeIDX, :],
                electric_l2_m2_component[ShapeIDX, :],
                predicted_electric_l1_m1_component[ShapeIDX, :],
                predicted_electric_l2_m1_component[ShapeIDX, :],
                predicted_electric_l2_m2_component[ShapeIDX, :]]

        MagneticComponents = [magnetic_l1_m1_component[ShapeIDX, :],
                magnetic_l2_m1_component[ShapeIDX, :],
                magnetic_l2_m2_component[ShapeIDX, :],
                predicted_magnetic_l1_m1_component[ShapeIDX, :],
                predicted_magnetic_l2_m1_component[ShapeIDX, :],
                predicted_magnetic_l2_m2_component[ShapeIDX, :]]




        PlotField(Field=[SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz],
            Components = ElectricComponents,
            FieldType = "Electric",
            Representation = "Phase",
            Shape = profiles,
            ShapeIDX = ShapeIDX,
            Wavelengths = wavelengths,
            WavelengthIDX = WavelengthIDX,
            SavePath = "/media/work/evan/MultipoleFieldImageData/temp/",
            FigSize = [15, 30]
            )

        PlotField(Field=[SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz],
            Components = ElectricComponents,
            FieldType = "Electric",
            Representation = "Magnitude",
            Shape = profiles,
            ShapeIDX = ShapeIDX,
            Wavelengths = wavelengths,
            WavelengthIDX = WavelengthIDX,
            SavePath = "/media/work/evan/MultipoleFieldImageData/temp/",
            FigSize = [15, 30]
            )

        PlotField(Field=[SimHx, SimHy, SimHz, CnnHx, CnnHy, CnnHz],
            Components = MagneticComponents,
            FieldType = "Magnetic",
            Representation = "Phase",
            Shape = profiles,
            ShapeIDX = ShapeIDX,
            Wavelengths = wavelengths,
            WavelengthIDX = WavelengthIDX,
            SavePath = "/media/work/evan/MultipoleFieldImageData/temp/",
            FigSize = [15, 30]
            )

        PlotField(Field=[SimHx, SimHy, SimHz, CnnHx, CnnHy, CnnHz],
            Components = MagneticComponents,
            FieldType = "Magnetic",
            Representation = "Magnitude",
            Shape = profiles,
            ShapeIDX = ShapeIDX,
            Wavelengths = wavelengths,
            WavelengthIDX = WavelengthIDX,
            SavePath = "/media/work/evan/MultipoleFieldImageData/temp/",
            FigSize = [15, 30]
            )
            
        PlotError(Field=[SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz],
            Components = ElectricComponents,
            FieldType = "Electric",
            Representation = "Phase",
            Shape = profiles,
            ShapeIDX = ShapeIDX,
            Wavelengths = wavelengths,
            WavelengthIDX = WavelengthIDX,
            SavePath = "/media/work/evan/MultipoleFieldImageData/temp/",
            FigSize = [15, 30]
            )
        PlotError(Field=[SimEx, SimEy, SimEz, CnnEx, CnnEy, CnnEz],
            Components = MagneticComponents,
            FieldType = "Magnetic",
            Representation = "Phase",
            Shape = profiles,
            ShapeIDX = ShapeIDX,
            Wavelengths = wavelengths,
            WavelengthIDX = WavelengthIDX,
            SavePath = "/media/work/evan/MultipoleFieldImageData/temp/",
            FigSize = [15, 30]
            )
    print("Electric Error Movie")
    makemovie("Electric", "Error", ShapeIDX, wavelengths, WavelengthRange)
    print("Electric Phase Movie")
    makemovie("Electric", "Phase", ShapeIDX, wavelengths, WavelengthRange)
    print("Electric Magnitude Movie")
    makemovie("Electric", "Magnitude", ShapeIDX, wavelengths, WavelengthRange)
    print("Magnetic Error Movie")
    makemovie("Magnetic", "Error", ShapeIDX, wavelengths, WavelengthRange)
    print("Magnetic Phase Movie")
    makemovie("Magnetic", "Phase", ShapeIDX, wavelengths, WavelengthRange)
    print("Magnetic Magnitude Movie")
    makemovie("Magnetic", "Magnitude", ShapeIDX, wavelengths, WavelengthRange)















