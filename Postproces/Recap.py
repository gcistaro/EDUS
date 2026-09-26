import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from custom_functions.read import read_observables
from custom_functions.window import CutWindow
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
t_au, Pt, Et_au, At_au, Vt_au = read_observables(args.folder, args.version)
t_fs = t_au*constants.physical_constants["atomic unit of time"][0]*1.e15

fig = plt.figure()
gs = fig.add_gridspec(4, 2, hspace=0)
ax = gs.subplots(sharex="col")

fig.suptitle('Plots of output variables')

for ip,p in enumerate(Pt):
    ax[0][0].plot(t_fs, p, label=str(ip))
ax[0][0].legend()
ax[0][0].set_ylabel("Population")

ax[1][0].plot(t_fs, Et_au[0], label="$E_x$")
ax[1][0].plot(t_fs, Et_au[1], label="$E_y$")
ax[1][0].plot(t_fs, Et_au[2], label="$E_z$")
ax[1][0].legend()
ax[1][0].set_ylabel("Electric field")

ax[2][0].plot(t_fs, At_au[0], label="$A_x$")
ax[2][0].plot(t_fs, At_au[1], label="$A_y$")
ax[2][0].plot(t_fs, At_au[2], label="$A_z$")
ax[2][0].legend()
ax[2][0].set_ylabel("Vector Potential")

ax[3][0].plot(t_fs, np.real(Vt_au[0]), label="$V_x$")
ax[3][0].plot(t_fs, np.real(Vt_au[1]), label="$V_y$")
ax[3][0].plot(t_fs, np.real(Vt_au[2]), label="$V_z$")
ax[3][0].set_xlabel("Time (fs)")
ax[3][0].legend()
ax[3][0].set_ylabel("Velocity")


#Fourier transform everything
#w_eV, Pw = FourierTransform(t_au, Pt, 0.2)
w_eV, Vw = FourierTransform(t_au, Vt_au, 0.)
w_eV, Ew = FourierTransform(t_au, Et_au, 0.)
index = CutWindow(w_eV, np.linalg.norm(Ew, axis=0), None)
w_eV = w_eV[index]
Vw = Vw[:,index]
Ew=Ew[:,index]
#Pw=Pw[:,index]



ax[1][1].plot(w_eV, np.abs(Ew[0]), label="$E_x$")
ax[1][1].plot(w_eV, np.abs(Ew[1]), label="$E_y$")
ax[1][1].plot(w_eV, np.abs(Ew[2]), label="$E_z$")
ax[1][1].set_xlabel("Time (fs)")
ax[1][1].legend()

ax[2][1].plot(w_eV, np.abs(Vw[0]), label="$V_x$")
ax[2][1].plot(w_eV, np.abs(Vw[1]), label="$V_y$")
ax[2][1].plot(w_eV, np.abs(Vw[2]), label="$V_z$")
ax[2][1].set_xlabel("Frequency (eV)")
ax[2][1].legend()




plt.show()
plt.savefig("Population.png", dpi=200)
