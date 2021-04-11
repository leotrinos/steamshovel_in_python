#!/usr/bin/env python3

import os
import sys
from optparse import OptionParser

import numpy as np
import scipy as ci 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import sklearn 
import math
from matplotlib.widgets import Slider, Button, RadioButtons
import pandas as pd 

import tables 
import glob
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import shutil

# NOTE: this file reads h5 files with pulse info and visualize the event.
# To visualize one event, open the terminal, then: ./steamshovel.py -i filename.h5 

""" define Arrow object """

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])    # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])                # z coordinate of points in inside surface
    return x, y, z

""" IT ineff parametrization """ 

ineff_params = np.loadtxt('/Users/yang/Desktop/IceCube/Codes/my_proj/analysis_1_downgoing_neutrino/1_distribution_plots/IT_model_robust_2.txt')

def ineff_model_new(dist,E): # E: log10 of energy!!! 
    # low stat region
    # if (E > 5.47+2*np.sqrt(1 - ((dist-20)/2210)**2)) or dist > 2210 :
    #     return -1

    a = ineff_params[0] * (E - ineff_params[1])**2 + ineff_params[2]
    b = ineff_params[3] * (E - ineff_params[4])**2 + ineff_params[5]

    val = a * np.exp(-b/(dist/2000)**(4.2)) # 2 pulse robust

    if val < 1e-5: 
        val = 1e-5

    return val


""" read args """

usage = "%prog [options] <inputfiles>"
parser = OptionParser(usage=usage)

parser.add_option(
                  "-i", "--inputfile",
                  type      = "string",
                  help      = "Name of the input h5 file",
                  dest      = "inputfile"
                  )

parser.add_option(
                  "-t", "--icetop",
                #   type      = ,
                  help      = "display icetop or not",
                  dest      = "icetop",
                  default   = "True",
                  )

(options, args) = parser.parse_args()
input_file = options.inputfile

filelist = glob.glob('/'.join(input_file.rsplit('/')[:-1])+'/*.h5')
filelist.sort()

for i in range(len(filelist)): 
    if input_file == filelist[i]:
        frame = i

fig = plt.figure(figsize=(8,8))

f = tables.open_file(filelist[frame],'r')
    
""" Read pulses and physics parameters"""

# string = f.root.InIceDSTPulses.cols.string[:]
# om = f.root.InIceDSTPulses.cols.om[:]
# charge = f.root.InIceDSTPulses.cols.charge[:]
# time = f.root.InIceDSTPulses.cols.time[:]

string = f.root.InIcePulsesHLC_NoDC_TW6000.cols.string[:]
om = f.root.InIcePulsesHLC_NoDC_TW6000.cols.om[:]
charge = f.root.InIcePulsesHLC_NoDC_TW6000.cols.charge[:]
time = f.root.InIcePulsesHLC_NoDC_TW6000.cols.time[:]

try:
    string_IT = np.concatenate((f.root.OfflineIceTopHLCVEMPulses.cols.string[:], f.root.OfflineIceTopSLCVEMPulses.cols.string[:]))
    om_IT = np.concatenate((f.root.OfflineIceTopHLCVEMPulses.cols.om[:], f.root.OfflineIceTopSLCVEMPulses.cols.om[:]))
    charge_IT = np.concatenate((f.root.OfflineIceTopHLCVEMPulses.cols.charge[:], f.root.OfflineIceTopSLCVEMPulses.cols.charge[:]))
    time_IT = np.concatenate((f.root.OfflineIceTopHLCVEMPulses.cols.time[:], f.root.OfflineIceTopSLCVEMPulses.cols.time[:]))
except:
    string_IT = f.root.OfflineIceTopSLCVEMPulses.cols.string[:]
    om_IT = f.root.OfflineIceTopSLCVEMPulses.cols.om[:]
    charge_IT = f.root.OfflineIceTopSLCVEMPulses.cols.charge[:]
    time_IT = f.root.OfflineIceTopSLCVEMPulses.cols.time[:]

params = {
    'zenith' : np.round(math.degrees(f.root.SplineMPE.cols.zenith[:][0]), 1),
    'log10(truncated)' : np.round(np.log10(f.root.SplineMPETruncatedEnergy_SPICEMie_BINS_Muon.cols.energy[:][0]), 2),
    'log10(Qtot)' : np.round(np.log10(f.root.QTot.cols.value[:][0]), 2),
    'log10(chi2_red_new)' : np.round(np.log10(f.root.Collection.cols.chi2_red_new[:])[0], 2),
    'd_to_IT_center' : np.round(f.root.Dist_to_IT_center.cols.value[:][0], 1),
    'len(dEdxVector)' : int(f.root.Collection.cols.len_dEdxVector[:][0]),
    'N_correlated_IT_HLC': int(f.root.IT_veto_double.cols.num_correlated_HLC_hits_in_window[:][0]),
    'N_correlated_IT_SLC': int(f.root.IT_veto_double.cols.num_correlated_SLC_hits_in_window[:][0]),
    '2_pulse_veto_passed?': int(f.root.IT_veto_double.cols.flag_window_2[:][0]),
    'IT_ineff' : np.round(ineff_model_new(f.root.Dist_to_IT_center.cols.value[:][0], np.log10(f.root.SplineMPETruncatedEnergy_SPICEMie_BINS_Muon.cols.energy[:][0])), 5),

}

""" construct coordinate matrix """
# order of txt: string, om, x, y, z --> (1,1), (1,2),...(1,60),(2,1),...(86,60), including IT tanks
S,O,X,Y,Z = np.loadtxt('/Users/yang/Desktop/IceCube/dom_positions.txt',dtype=float,delimiter=',',unpack=True)

IT_om = [61, 63] # 2 tanks for each string, after merging high-gain and low-gain doms
IT_zcoord = 1500

Z[O==IT_om[0]] = IT_zcoord
Z[O==IT_om[1]] = IT_zcoord

coord = np.transpose(np.array([S,X,Y,Z]))

# remove deepcore from IC visualization
DeepCoreStrings = [79,80,81,82,83,84,85,86]

# remove DC grids
for i in DeepCoreStrings:
    coord = coord[coord[:,0] != i]
coord = coord[:,1:]
coord[:,2][coord[:,2] > 600] = IT_zcoord # shift IT position

""" Construct pulse DataFrame """

# make mapping of DOM key to XYZ
dom_to_coord = {}
for i in range(0,len(S)):
    # scale IT dom position
    dom_to_coord[(S[i],O[i])] = (X[i],Y[i],Z[i])

data_inice = []
s_ = np.linspace(1,86,86,dtype=int)
o_ = np.linspace(1,60,60,dtype=int) # including IT

for s in s_:
    for o in o_:
        if s not in DeepCoreStrings: # remove deepcore from data

            sel = (string==s)&(om==o) # select corresponding values for this DOM
            total_q = np.sum(charge[sel])

            if total_q != 0:
                
                charges = charge[sel]
                times = time[sel]
                weighted_time = np.sum(charges*times) / np.sum(charges)
                earliest_time = np.min(times)
            
                # data_inice.append([dom_to_coord[(s,o)][0],dom_to_coord[(s,o)][1],dom_to_coord[(s,o)][2],total_q,weighted_time]) # data: X,Y,Z,Total Charge in one DOM,Time (weighted)
                data_inice.append([dom_to_coord[(s,o)][0],dom_to_coord[(s,o)][1],dom_to_coord[(s,o)][2],total_q,earliest_time]) # data: X,Y,Z,Total Charge in one DOM,Time (earliest)

df_inice = pd.DataFrame(data_inice, columns=['x','y','z','q','t'])

data_IT = []
s_ = np.linspace(1,86,86,dtype=int)
o_ = np.linspace(61,64,4,dtype=int) # IT
for s in s_:
    for o in o_:
        if s not in DeepCoreStrings: # remove deepcore from data

            sel = (string_IT==s)&(om_IT==o) # select corresponding values for this DOM
            total_q = np.sum(charge_IT[sel])

            if total_q != 0:
                
                charges = charge_IT[sel]
                times = time_IT[sel]
                weighted_time = np.sum(charges*times) / np.sum(charges)
                earliest_time = np.min(times)
            
                # data_IT.append([dom_to_coord[(s,o)][0],dom_to_coord[(s,o)][1],IT_zcoord,total_q,weighted_time]) # data: X,Y,Z,Total Charge in one DOM,Time (weighted)
                data_IT.append([dom_to_coord[(s,o)][0],dom_to_coord[(s,o)][1],IT_zcoord,total_q,earliest_time]) # data: X,Y,Z,Total Charge in one DOM,Time (earliest)

df_IT = pd.DataFrame(data_IT, columns=['x','y','z','q','t'])
print(df_IT)
df_combined = pd.concat([df_inice,df_IT],ignore_index=True)

# ----- find main peak for in-ice pulse distributions ----- # 
# Need to manually change 'lower' and 'upper' if charge cluster identification is wrong. 
center = np.sum(df_inice['q'] * df_inice['t']) / np.sum(df_inice['q'])

std = 2500
# lower = center - 4.8*std
# upper = center + 4.2*std
lower = np.min(df_combined['t']-1)
upper = np.max(df_combined['t']+1)

""" Steamshovel: Main """

gs = gridspec.GridSpec(3, 2, height_ratios=[8.,0.7,0.2],width_ratios=[0.0,1]) 
ax = plt.subplot(gs[1], projection='3d')

def show_pulses(df, lower_, upper_):

    ax.scatter(coord[:,0], coord[:,1], coord[:,2], s=0.3, c='black', marker='.',alpha=0.3) # DOMs
    ax.scatter(df['x'], df['y'], df['z'], s=100*df['q']**(1/3.), c=df['t'], vmin=lower_, vmax=upper_, marker='.',alpha=1,cmap='rainbow_r') # Charges
    
    # --- reco event info box --- # 
    y_pos = 1
    for key,val in params.items():
        ax.text2D(-0.1,y_pos, s=key+': ' + str(val), transform=ax.transAxes,fontsize=9)
        y_pos -= 0.03

    # reco track
    theta = f.root.SplineMPE.cols.zenith[:][0]
    phi = f.root.SplineMPE.cols.azimuth[:][0]
    x0 = f.root.SplineMPE.cols.x[:][0]
    y0 = f.root.SplineMPE.cols.y[:][0]
    z0 = f.root.SplineMPE.cols.z[:][0]

    tt = np.linspace(-1000,1000,10)
    xx = x0 - np.sin(theta)*np.cos(phi)*tt
    yy = y0 - np.sin(theta)*np.sin(phi)*tt
    zz = z0 - np.cos(theta)*tt
    ax.plot(xx[0:1], yy[0:1], zz[0:1], '-',color='steelblue',label='SplineMPE')
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0)
    a = Arrow3D([xx[0],xx[-1]], [yy[0],yy[-1]], [zz[0],zz[-1]], **arrow_prop_dict, color='steelblue')
    ax.add_artist(a)

    # arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([0, 150], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 150], [0, 0], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 150], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.tick_params(axis='x', pad=1)
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='z', pad=1)

    name = str(filelist[frame].rsplit('/')[-1].rsplit('.')[0])
    fig.suptitle(name,fontsize=9)
    ax.set_xlim(-500,500)
    ax.set_ylim(-500,500)
    ax.set_zlim(-500,IT_zcoord*1.05)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return name 

name = show_pulses(df_combined, lower, upper)
print('frame '+str(frame)+', '+name+'.png')


""" Steamshovel: Pulse distribution """

ax2 = plt.subplot(gs[3])
ax2.hist(df_inice['t'],bins=100,histtype='step',weights=df_inice['q'],log=True) # use OLD data array!
ax2.hist(df_IT['t'],bins=100,histtype='step',weights=df_IT['q'],log=True,color='red') # use OLD data array!
ax2.set_xlim(np.min(df_IT['t'])-1000,np.max(df_IT['t'])+1000)
l1 = ax2.axvline(lower,color='darkorange')
l2 = ax2.axvline(upper,color='m')
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(8) 
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(8) 
# ax2.set_yticks([], [])
plt.xlabel('pulse time [ns]')
plt.ylabel('log10 charge [PE]')


""" Steamshovel: Update() """

L = plt.axes([0.13, 0.24, 0.1, 0.03], facecolor='darkorange')
R = plt.axes([0.13, 0.20, 0.1, 0.03], facecolor='m')

# leftbar = Slider(L, 'left', np.min(df_inice['t']), center, valinit=lower, valstep=200)
# rightbar = Slider(R, 'right', center, np.max(df_inice['t']), valinit=upper,valstep=200)
leftbar = Slider(L, 'left', lower, center, valinit=lower, valstep=200)
rightbar = Slider(R, 'right', center, upper, valinit=upper,valstep=200)

def update(val):
    lower = leftbar.val
    upper = rightbar.val

    ax.clear()
    data_new = df_combined.loc[(df_combined['t']>lower)&(df_combined['t']<upper)].copy(deep=True)
    show_pulses(data_new, lower, upper)

    l1.set_xdata(lower)
    l2.set_xdata(upper)

    fig.canvas.draw_idle()

rightbar.on_changed(update)
leftbar.on_changed(update)
plt.tight_layout()

plt.show()

