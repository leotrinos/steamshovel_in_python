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

import tables 
import glob
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import shutil

# NOTE: this file reads h5 files with pulse info and visualize the event.
# To visualize one event, open the terminal, then: ./steamshovel.py -i filename.h5 
# If all the events are inside one folder, could navigate between events by typing direction keys on the keyboard. 

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


# input_file: /Users/yang/Desktop/h5_files/data_sig_129327_53979238.h5
filelist = glob.glob('/'.join(input_file.rsplit('/')[:-1])+'/*.h5')
filelist.sort()
# print(filelist)

filelist2 = os.listdir('/'.join(input_file.rsplit('/')[:-1]))
# print(filelist2)
for i in range(len(filelist)): 
    if input_file == filelist[i]:
        frame = i



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


def plotting():
    global filelist, frame

    f = tables.open_file(filelist[frame],'r')

    if 'corsika' in filelist[frame] or 'numu' in filelist[frame]:
        mc = 0 # FIXME: change this 
    else:
        mc = 0
        
    # --- #FIXME: which pulse to use? --- # 

    string = f.root.InIceDSTPulses.cols.string[:]
    om = f.root.InIceDSTPulses.cols.om[:]
    charge = f.root.InIceDSTPulses.cols.charge[:]
    time = f.root.InIceDSTPulses.cols.time[:]

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

    ax.text2D(-0.1,1, s='QTot: %.1f'%f.root.QTot.cols.value[:][0],transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.97, s='log truncated: %.2f'%np.log10(f.root.SplineMPETruncatedEnergy_SPICEMie_BINS_Muon.cols.energy[:][0]),transform=ax.transAxes,fontsize=9)
    # ax.text2D(-0.1,0.94, s='chi2_red: %.3f'%f.root.Collection.cols.chi2_red[:][0],transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.94, s='log10 chi2_red_new: %.3f'%np.log10(f.root.Collection.cols.chi2_red_new[:][0]),transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.91, s='log10 prim E: %.3f'%np.log10(f.root.MCPrimary_new.cols.energy[:][0]),transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.88, s='len(dEdx): %i'%f.root.Collection.cols.len_dEdxVector[:][0],transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.85,s="SplineMPE:",transform=ax.transAxes,fontsize=9)
    ax.text2D(-0.1,0.82,s=r"$\theta$="+str(np.round(np.degrees(theta),1))+r", $\phi$="+str(np.round(np.degrees(phi),1)),transform=ax.transAxes,fontsize=9)
    # ax.text2D(-0.1,0.78, s='multiplicity: %i'%f.root.PolyplopiaInfo.cols.Multiplicity[:][0],transform=ax.transAxes,fontsize=9)

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
    
    name = str(filelist[frame].rsplit('/')[-1].rsplit('.')[0])
    # ax.set_title(name)
    fig.suptitle(name,fontsize=9)
    ax.set_xlim(-500,500)
    ax.set_ylim(-500,500)
    ax.set_zlim(-500,500)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.legend(loc='upper right')
    plt.tight_layout()

    print('frame '+str(frame)+', '+name+'.png')


    " ----- plot 3: pulse distribution -----"


    ax2 = plt.subplot(gs[3])
    ax2.hist(data[:,4],bins=100,histtype='step',weights=data[:,3],log=True) # use OLD data array!
    ax2.axvline(lower,color='darkorange')
    ax2.axvline(upper,color='m')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    # ax2.set_yticks([], [])
    plt.xlabel('pulse time [ns]')
    plt.ylabel('log10 charge [PE]')



def press(event):
    global frame,filelist
    if (event.key == 'down' or event.key == 'right') and frame +1 <= len(filelist)-1:
        frame += 1
    elif (event.key == 'up' or event.key == 'left') and frame -1 >= 0:
        frame -= 1

    plt.clf()
    plotting()
    plt.draw()

    # if (event.key == 'm'): # mark! 
    #     azim, elev = item[1].azim, item[1].elev
    #     print(azim,elev)
    #     # item[1].view_init(elev=elev, azim=azim)
    #     plt.savefig('/Users/yang/Desktop/steamshovel_plots/'+item[0]+'.png',dpi=300)

fig = plt.figure(figsize=(8,8))
plotting()
fig.canvas.mpl_connect('key_press_event', press)


plt.show()






# -------- backup ----------- # 


# mean_t = x_[h==np.max(h)]
# print(mean_t)
# print([h==np.max(h)])
# plt.figure()
# plt.hist(data[:,4],bins=100,label='pulses',weights=data[:,3])
# plt.show()

# fitting method 
# def gauss(x,A,mu,sigma):
#     return A*np.exp(-(x-mu)**2/(2*sigma**2))

# x = x[h>0]
# y = np.log10(h[h>0])
# popt,pcov = curve_fit(gauss,x,y,p0=[np.max(y),mean_t,1000])
# mu = popt[1]
# sig = np.abs(popt[2])

# width = 3*sig
# lower = mean_t - 0.2*width
# upper = mean_t + 1.2*width








" ----- plot 1: energy loss ----- "

# plt.figure(figsize=(7,7))
# loss = []
# loss = np.array([f.root.Collection.cols.EnergyLoss_0[:][0], 
#     f.root.Collection.cols.EnergyLoss_1[:][0], 
#     f.root.Collection.cols.EnergyLoss_2[:][0], 
#     f.root.Collection.cols.EnergyLoss_3[:][0], 
#     f.root.Collection.cols.EnergyLoss_4[:][0], 
#     f.root.Collection.cols.EnergyLoss_5[:][0], 
#     f.root.Collection.cols.EnergyLoss_6[:][0], 
#     f.root.Collection.cols.EnergyLoss_7[:][0], 
#     f.root.Collection.cols.EnergyLoss_8[:][0], 
#     f.root.Collection.cols.EnergyLoss_9[:][0], 
#     f.root.Collection.cols.EnergyLoss_10[:][0], 
#     f.root.Collection.cols.EnergyLoss_11[:][0], 
#     f.root.Collection.cols.EnergyLoss_12[:][0], 
#     f.root.Collection.cols.EnergyLoss_13[:][0], 
#     f.root.Collection.cols.EnergyLoss_14[:][0]])

# loss = loss[loss>0]
# bins = np.arange(0,len(loss))+0.5

# def line(x,k,b):
#     return k*x + b
# popt, pcov = curve_fit(line, bins, loss, sigma=loss)

# calculated_chi2_red = np.sum((line(bins, *popt) - loss)**2/loss**2)/(len(loss)-2)

# plt.figure(figsize=(7,7))
# plt.errorbar(bins,loss,fmt='o',yerr=loss)
# t_ = np.arange(0,15,0.1)
# plt.plot(t_,line(t_,*popt))
# plt.title(r'calulated $\chi^2_{red}$ = '+str(np.round(calculated_chi2_red,2)))
# plt.xlabel('bin')
# plt.ylabel('muon energy loss [GeV]')
# plt.show()



# finding clustering charge peaks 


# h,b = np.histogram(data[:,4],bins=100,weights=data[:,3]) # pulse distribution weighted by charge deposit at each time
# b = (b[1:]+b[:-1])/2

# tmp = np.zeros_like(h)
# for i in range(0,len(h)-3): # end of tmp is 0. easier to deal with. 
#     if h[i]>1 and h[i+1]>1 and h[i+2]>1:
#         tmp[i],tmp[i+1],tmp[i+2]=1,1,1
# # print(tmp)
# seg = []
# if tmp[0]==1:
#     seg.append(0)
# for i in range(0,len(tmp)-1):
#     # edge case
#     if (tmp[i]==0. and tmp[i+1]==1.):
#         seg.append(i)
#     if (tmp[i]==1. and tmp[i+1]==0.):
#         seg.append(i)

# for i in range(0,len(seg),2):
#     peak = np.where(h==np.max(h))[0][0]
#     if peak>=seg[i] and peak<=seg[i+1]:
#         try:
#             lower = b[seg[i]-2]
#             upper = b[seg[i+1]+2]
#         except:
#             lower = b[seg[i]]
#             upper = b[seg[i+1]]
# #  cut on time window 
# data_new = data[(data[:,4]>lower)&(data[:,4]<upper)]




# ----- K-means clustering ----- # 
# Q = data[:,3] # weighted by charge
# T = data[:,4] # time 
# # plt.figure()
# # plt.scatter(Q,T)
# # plt.show()
# kmeans = KMeans(n_clusters=2).fit(np.c_[Q,T])#,sample_weight=weight)
# print(kmeans.labels_)
# label = kmeans.labels_
# lower = np.min(T[label==0])
# upper = np.max(T[label==0])
# print(lower,upper)
# data_new = data[(data[:,4]>lower)&(data[:,4]<upper)]