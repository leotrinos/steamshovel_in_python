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
# from sklearn.cluster import KMeans
from matplotlib.widgets import Slider, Button, RadioButtons

import tables 
import glob
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import shutil

# NOTE: this file reads h5 files with pulse info and visualize the event.
# To visualize one event, open the terminal, then: ./steamshovel.py -i filename.h5 

usage = "%prog [options] <inputfiles>"
parser = OptionParser(usage=usage)

parser.add_option(
                  "-i", "--inputfile",
                  type      = "string",
                  help      = "Name of the input h5 file",
                  dest      = "inputfile"
                  )

(options, args) = parser.parse_args()
input_file = options.inputfile

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

fig = plt.figure(figsize=(8,8))

f = tables.open_file(input_file,'r')

events = f.root.I3EventHeader.cols.Event[:]

cut = f.root.InIceDSTPulses.cols.Event[:]==events[0]

mc = 0
# if 'corsika' in filelist[frame] or 'numu' in filelist[frame]:
#     mc = 0 # FIXME: change this 
# else:
#     mc = 0
    
# --- #FIXME: which pulse to use? --- # 

string = f.root.InIceDSTPulses.cols.string[:][cut]
om = f.root.InIceDSTPulses.cols.om[:][cut]
charge = f.root.InIceDSTPulses.cols.charge[:][cut]
time = f.root.InIceDSTPulses.cols.time[:][cut]
print(string)
# string = f.root.InIcePulsesHLC_NoDC_TW6000.cols.string[:]
# om = f.root.InIcePulsesHLC_NoDC_TW6000.cols.om[:]
# charge = f.root.InIcePulsesHLC_NoDC_TW6000.cols.charge[:]
# time = f.root.InIcePulsesHLC_NoDC_TW6000.cols.time[:]

# string = f.root.InIcePulsesHLC_NoDC.cols.string[:]
# om = f.root.InIcePulsesHLC_NoDC.cols.om[:]
# charge = f.root.InIcePulsesHLC_NoDC.cols.charge[:]
# time = f.root.InIcePulsesHLC_NoDC.cols.time[:]

# string = f.root.L4_TTPulsesTWIC.cols.string[:]
# om = f.root.L4_TTPulsesTWIC.cols.om[:]
# charge = f.root.L4_TTPulsesTWIC.cols.charge[:]
# time = f.root.L4_TTPulsesTWIC.cols.time[:]

# ----- construct coord matrix ----- #
# order of txt: string, om, x, y, z --> (1,1), (1,2),...(1,60),(2,1),...(86,60)
s,o,x,y,z = np.loadtxt('/Users/yang/Jupyter/IceCube/dom_positions.txt',dtype=float,delimiter=',',unpack=True)

# make mapping of DOM key to XYZ
dom_to_coord = {}
for i in range(0,len(s)):
    dom_to_coord[(s[i],o[i])] = (x[i],y[i],z[i])

coord = np.transpose(np.array([s,x,y,z]))
# remove deepcore from IC visualization
DeepCoreStrings = [79,80,81,82,83,84,85,86]
for i in DeepCoreStrings:
    coord = coord[coord[:,0] != i]
coord = coord[:,1:]

# ----- make data matrix ----- # 
data = []
s_ = np.linspace(1,86,86,dtype=int)
o_ = np.linspace(1,60,60,dtype=int)

for s in s_:
    for o in o_:
        if s not in DeepCoreStrings: # remove deepcore from data

            sel = (string==s)&(om==o) # select corresponding values for this DOM
            total_q = np.sum(charge[sel])

            if total_q != 0:
                
                charges = charge[sel]
                times = time[sel]
                weighted_time = np.sum(charges*times) / np.sum(charges)
                earliest_time = np.min(time[sel])

                # use time weighted by pulses for each dom or first-arrival time: 
                # data.append([dom_to_coord[(s,o)][0],dom_to_coord[(s,o)][1],dom_to_coord[(s,o)][2],total_q,weighted_time]) # data: X,Y,Z,Total Charge in one DOM,Time (earliest)
                data.append([dom_to_coord[(s,o)][0],dom_to_coord[(s,o)][1],dom_to_coord[(s,o)][2],total_q,earliest_time]) # data: X,Y,Z,Total Charge in one DOM,Time (earliest)

data = np.array(data)

# remove doms with 0 charges     
# data = data[data[:,3] != 0.]

# ------ if input is I3DSTPulse (or anything that does not have time-window cleaning): need to select region where charge is the most! ------ # 
# ------ calculating mean, upper and lower bounds  ------ # 

# ----- weighted peak ----- # FIXME: Need to manually change 'lower' and 'upper' if charge cluster identification is wrong. 
Q = data[:,3] # weighted by charge
T = data[:,4] # time 
center = np.sum(Q*T) / np.sum(Q)

std = 2500
lower = center - 0.8*std
upper = center + 1.2*std
data_new = data[(data[:,4]>lower)&(data[:,4]<upper)]

# ------else (after time-window cleaning): ------ # 
# data_new = data    





" ----- plot 2: SteamShovel -----" 

gs = gridspec.GridSpec(3, 2, height_ratios=[8.,0.7,0.2],width_ratios=[0.0,1]) 
ax = plt.subplot(gs[1], projection='3d')

def show_pulses(data_new):

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.tick_params(axis='x', pad=1)
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='z', pad=1)

    ax.scatter(coord[:,0], coord[:,1], coord[:,2], s=0.3, c='black', marker='.',alpha=0.3) # DOMs
    ax.scatter(data_new[:,0], data_new[:,1], data_new[:,2], s=100*data_new[:,3]**(1/3.), c=data_new[:,4], marker='.',alpha=1,cmap='rainbow_r') # Charges

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

    # --- reco event info box --- # 

    # ax.text2D(-0.1,1, s='QTot: %.1f'%f.root.QTot.cols.value[:][0],transform=ax.transAxes,fontsize=9)
    # ax.text2D(-0.1,0.97, s='log truncated: %.2f'%np.log10(f.root.SplineMPETruncatedEnergy_SPICEMie_BINS_Muon.cols.energy[:][0]),transform=ax.transAxes,fontsize=9)
    # ax.text2D(-0.1,0.94, s='chi2_red: %.3f'%f.root.Collection.cols.chi2_red[:][0],transform=ax.transAxes,fontsize=9)
    # ax.text2D(-0.1,0.91, s='len(dEdx): %i'%f.root.Collection.cols.len_dEdxVector[:][0],transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.88,s="SplineMPE:",transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.85,s=r"$\theta$="+str(np.round(np.degrees(theta),1))+r", $\phi$="+str(np.round(np.degrees(phi),1)),transform=ax.transAxes,fontsize=9)

    # MCPrimary
    if mc == True:
        # find closest approach position to the origin 
        def closest_approach_position(f,pos):
            theta = f.root.MCPrimary_new.cols.zenith[:][0]
            phi = f.root.MCPrimary_new.cols.azimuth[:][0]
            x = f.root.MCPrimary_new.cols.x[:][0]
            y = f.root.MCPrimary_new.cols.y[:][0]
            z = f.root.MCPrimary_new.cols.z[:][0]

            a = -1 * np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T # track direction is radially inward! 
            b = np.array([pos[0] - x, pos[1] - y, pos[2] - z]).T

            return np.multiply(a.T, (np.multiply(a,b).sum())/np.linalg.norm(a) ).T + np.array([x,y,z]).T  # x = (a . b)/a * r + Q
        
        closest = closest_approach_position(f,[0,0,0])

        theta = f.root.MCPrimary_new.cols.zenith[:][0]
        phi = f.root.MCPrimary_new.cols.azimuth[:][0]
        x0 = closest[0]
        y0 = closest[1]
        z0 = closest[2]

        ax.text2D(-0.1,0.82,s="MCPrimary:",transform=ax.transAxes,fontsize=9)
        ax.text2D(-0.1,0.79,s=r"$\theta$="+str(np.round(np.degrees(theta),1))+r", $\phi$="+str(np.round(np.degrees(phi),1)),transform=ax.transAxes,fontsize=9)
        ax.text2D(-0.1,0.76, s='Coincidence: %i'%f.root.MCPrimary_coincident.cols.value[:][0],transform=ax.transAxes,fontsize=9)
        ax.text2D(-0.1,0.73, s='Singleness: %0.2f'%f.root.Bundle.cols.Singleness[:][0],transform=ax.transAxes,fontsize=9)

        tt = np.linspace(-1000,1000,10)
        xx = x0 - np.sin(theta)*np.cos(phi)*tt
        yy = y0 - np.sin(theta)*np.sin(phi)*tt
        zz = z0 - np.cos(theta)*tt
        ax.plot(xx[0:1], yy[0:1], zz[0:1], '-',color='red',label='MCPrimary')
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0)
        a = Arrow3D([xx[0],xx[-1]], [yy[0],yy[-1]], [zz[0],zz[-1]], **arrow_prop_dict, color='red')
        ax.add_artist(a)

    # arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([0, 150], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 150], [0, 0], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 150], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    
    # name = str(filelist[frame].rsplit('/')[-1].rsplit('.')[0])
    # fig.suptitle(name,fontsize=9)
    ax.set_xlim(-500,500)
    ax.set_ylim(-500,500)
    ax.set_zlim(-500,500)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # return name 
# name = show_pulses(data_new)
# print('frame '+str(frame)+', '+name+'.png')
show_pulses(data_new)
" ----- plot 3: pulse distribution -----"


ax2 = plt.subplot(gs[3])
ax2.hist(data[:,4],bins=100,histtype='step',weights=data[:,3],log=True) # use OLD data array!
l1 = ax2.axvline(lower,color='darkorange')
l2 = ax2.axvline(upper,color='m')
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(8) 
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(8) 
# ax2.set_yticks([], [])
plt.xlabel('pulse time [ns]')
plt.ylabel('log10 charge [PE]')


" ----- update plot ----- "

L = plt.axes([0.13, 0.24, 0.1, 0.03], facecolor='darkorange')
R = plt.axes([0.13, 0.20, 0.1, 0.03], facecolor='m')

leftbar = Slider(L, 'left', np.min(data[:,4]), center, valinit=lower, valstep=200)
rightbar = Slider(R, 'right', center, np.max(data[:,4]), valinit=upper,valstep=200)

def update(val):
    lower = leftbar.val
    upper = rightbar.val

    ax.clear()
    data_new = data[(data[:,4]>lower)&(data[:,4]<upper)]
    show_pulses(data_new)

    l1.set_xdata(lower)
    l2.set_xdata(upper)

    fig.canvas.draw_idle()

rightbar.on_changed(update)
leftbar.on_changed(update)
plt.tight_layout()

plt.show()

