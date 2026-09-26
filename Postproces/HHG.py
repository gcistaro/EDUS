#from scipy.fft import fft, fftfreq, fftshift
from scipy import constants
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import plotly.io as pio

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

plt.plot(t_au, Vt_au[0])
plt.show()
freq_eV, Vw = FourierTransform(t_au, Vt_au, 0.01)

aw = 1j*freq_eV*Vw
aw = np.linalg.norm(aw, axis=0)
logaw = np.log10(np.abs(aw*aw))
print("Vw shape:" , aw.shape)


plt.plot(freq_eV/8.2656e-01, logaw, label="$log_{10}|a|^2$")
#plt.axvline(x=7.25)
plt.legend()
plt.show()
np.savetxt("HHG.txt", np.array(np.transpose([freq_eV, logaw])))
plt.close()

#absorbance_fig = go.Figure()
#absorbance_fig.add_trace(go.Scatter(x=freq_eV, y=Absorbance, mode='lines'))
#pio.write_html(absorbance_fig, 'absorbance.html')

