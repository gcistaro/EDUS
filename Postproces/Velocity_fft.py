#from scipy.fft import fft, fftfreq, fftshift
from scipy import constants
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import sys

from custom_functions.read import read_observables
from custom_functions.fouriertransform import FourierTransform


import argparse, sys

#define arguments
parser=argparse.ArgumentParser()

parser.add_argument("--folder", default="./Output/", help="Folder where the .txt files are contained")
parser.add_argument("--version", default="new", help="Version of the EDUS code that you used")
args=parser.parse_args()

#print in output some infos
print("folder       :    ", args.folder)
print("version      :    ", args.version)



t_au, _, _, _, Vt_au = read_observables(args.folder, args.version)

freq_eV, Vw = FourierTransform(t_au, Vt_au, 0.0)

#plt.plot(freq_eV, np.real(Vw[0]), label="$Re(V_x(\omega))$")
#plt.plot(freq_eV, np.imag(Vw[0]), label="$Im(V_x(\omega))$")
#plt.plot(freq_eV, np.real(Vw[1]), label="$Re(V_y(\omega))$")
#plt.plot(freq_eV, np.real(Vw[1]), label="$Re(V_y(\omega))$")
#plt.plot(freq_eV, np.imag(Vw[2]), label="$Im(V_z(\omega))$")
#plt.plot(freq_eV, np.imag(Vw[2]), label="$Im(V_z(\omega))$")
plt.plot(freq_eV, np.abs(Vw[0]), label="$abs(V_x(\omega))$")
plt.plot(freq_eV, np.abs(Vw[1]), label="$abs(V_y(\omega))$")
plt.plot(freq_eV, np.abs(Vw[2]), label="$abs(V_z(\omega))$")
plt.xlim(-0.5,0.5)
#plt.axvline(x=7.25)
plt.legend()
plt.show()
plt.close()

