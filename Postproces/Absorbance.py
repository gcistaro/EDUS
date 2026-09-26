#from scipy.fft import fft, fftfreq, fftshift
from scipy import constants
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import plotly.io as pio

from custom_functions.read import read_observables
from custom_functions.absorbance import get_absorbance


import argparse, sys

#define arguments
parser=argparse.ArgumentParser()

parser.add_argument("--smearing", default=0.2, help="Smearing used in the current. The current will decay exponentially as e^{-t/smearing}", type=float)
parser.add_argument("--window", help="Window of energy where we want to plot the absorbance, numbers given in a sequence", nargs=2, type=float)
parser.add_argument("--folder", default="./Output/", help="Folder where the .txt files are contained")
parser.add_argument("--version", default="new", help="Version of the EDUS code that you used")
args=parser.parse_args()

#print in output some infos
print("Starting calculation of Absorbance. Here are the input parameters with their values:")
print("smearing (eV):    ", args.smearing)
print("window (eV)  :    ", args.window)
print("folder       :    ", args.folder)
print("version      :    ", args.version)


#read data
print("Reading t, E(t), V(t)...", end=" ", flush=True)
t_au, _, Et_au, _, Vt_au = read_observables(args.folder, args.version)
print("done.")
freq_eV, Absorbance = get_absorbance(t_au, Vt_au, Et_au, args.window, args.smearing)


print("Plotting absorbance...", end=" ", flush=True)
fig, ax = plt.subplots()
ax.plot(freq_eV, 100*Absorbance,linestyle='dashed', color="black")
ax.fill_between(freq_eV, 0*freq_eV, 100*Absorbance, facecolor = "green", alpha=0.6)
ax.set_xlabel("$\\omega$ (eV)")
ax.set_ylabel("Absorbed light (%)")
np.savetxt("absorbance.txt", np.array(np.transpose([freq_eV, Absorbance])))
plt.savefig("Absorbance.png", dpi=200)
plt.show()
plt.close()
print("done.")
absorbance_fig = go.Figure()
absorbance_fig.add_trace(go.Scatter(x=freq_eV, y=Absorbance, mode='lines'))
pio.write_html(absorbance_fig, 'absorbance.html')

