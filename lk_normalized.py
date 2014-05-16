#!/usr/bin/env python

'''
Clustering people algorithm which use Lucas-Kanade as a Tracker
====================

usage: python lk_cluster video_file

'''

import numpy as np
import cv2
import video
import pylab as pl
import sys
from common import anorm2, draw_str
from time import clock
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

lk_params = dict(winSize = (30, 30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 500,
                       qualityLevel = 0.08,
                       minDistance = 1,
                       blockSize = 7)

mean0 = -5.62189454539
mean1 = -6.01162818854
mean2 =  388.122677881 
mean3 =  245.87251023

var0 = 1082.38781566
var1 = 10030.1169921
var2 = 7752.48100664
var3 = 18325.0892826


def create_colors():
    rainbow_colors = []
    rainbow_colors.append((255, 0, 0))
    rainbow_colors.append((255, 127, 0))
    rainbow_colors.append((255, 255, 0))
    rainbow_colors.append((0, 255, 0))
    rainbow_colors.append((0, 0, 255))
    rainbow_colors.append((75, 0, 130))
    rainbow_colors.append((145, 0, 255))
    return rainbow_colors


def cluster_features(X, clustered_pixels, vis2):

#medias: -5.62189454539 -6.01162818854 388.122677881 245.87251023
#varianzas: 1082.38781566 10030.1169921 7752.48100664 18325.0892826

    clustered_xy = np.array(clustered_pixels) 
    #get colors
    rainbow = create_colors()

    
    X = np.delete(X, 1, 0)
    clustered_xy = np.delete(clustered_xy, 1, 0)

    #normalize all variables
    #col0 = np.true_divide(X[:,0],max(X[:,0]))
    #col1 = np.true_divide(X[:,1],max(X[:,1]))
    #col2 = np.true_divide(clustered_xy[:,0],max(clustered_xy[:,0]))
    #col3 = np.true_divide(clustered_xy[:,1],max(clustered_xy[:,1]))

    #col0 = X[:,0] / np.linalg.norm(X[:,0])
    #col1 = X[:,1] / np.linalg.norm(X[:,1])
    #col2 = clustered_xy[:,0] / np.linalg.norm(clustered_xy[:,0])
    #col3 = clustered_xy[:,1] / np.linalg.norm(clustered_xy[:,1])

    col0 = np.divide((np.subtract(X[:,0],mean0)),var0)
    col1 = np.divide((np.subtract(X[:,1],mean1)),var1)
    col2 = np.divide((np.subtract(clustered_xy[:,0],mean2)),var2)
    col3 = np.divide((np.subtract(clustered_xy[:,1],mean3)),var3)
    #join all variables in one matrix
    data_set = np.array([col0,col1,col2,col3])
    data_set = data_set.T

    #perform clustering in input_set
    db = DBSCAN(eps = 0.06).fit(data_set)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    #plot clustering
    unique_labels = set(labels)
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    color_ind = 0
    for k, col in zip(unique_labels, colors):
        color_ind += 1
        if color_ind > 6:
            color_ind = 0
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6

        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]

        for index in class_members:
            x = clustered_pixels[index]
            if k != -1:
                cv2.circle(vis2,clustered_pixels[index],2,rainbow[color_ind],2)
            if index in core_samples and k != -1:
                markersize = 14
            else:
                markersize = 6
            #pl.plot(x[0], x[1], 'o', markerfacecolor=col,
            #markeredgecolor='k', markersize=markersize)

    cv2.imshow("person detected", vis2)
    cv2.waitKey(0)
    #pl.xlabel("flujo en x")
    #pl.ylabel("flujo en y")
    #pl.title('Estimated number of clusters: %d' % n_clusters_)
    #pl.show()


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

    def run(self):
        cont = 0

        while True:
            cont += 1
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            vis2 = frame.copy()
            #fy = np.array([0])
            #fx = np.array([0])
            input_set = np.array([1,1])
            clustered_pixels = []
            clustered_pixels.append((1,1))
            #draw_str(vis, (20, 20), 'track count: %d' % cont)
            
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))

                    #optical flow from x axis and y axis
                    #fy = np.concatenate((fy,[y - fy[-1]]), axis=0)
                    #fx = np.concatenate((fx,[x - fx[-1]]), axis=0)

                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks

                #acumulate flow and store it in input_set
                for temporal_tracker in self.tracks:
                    flowx = 0
                    flowy = 0
                    for j in range(1,len(temporal_tracker)):
                        flowx += (temporal_tracker[j][0]-temporal_tracker[j-1][0])
                        flowy += (temporal_tracker[j][1]-temporal_tracker[j-1][1])
                    if ((-8 > flowx) or (flowx > 8)) and ((-18 > flowy) or (flowy > 18)):
                        input_set = np.vstack((input_set, [flowx, flowy]))
                        #last coordinates pixels which forms optical flow
                        clustered_pixels.append(temporal_tracker[-1])
                        
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                cv2.imshow('lk_track', vis)
                if (len(input_set)) > 2:
                    cluster_features(input_set, clustered_pixels, vis2)
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            ch = 0xFF & cv2.waitKey(0)
            if ch == 27:
                break
            if ch & 0xFF == ord("p"):
                clustered_xy = np.array(clustered_pixels) 
                #print "numero total: " + str(len(input_set)) + " " + str(len(clustered_pixels))
                #mean1 = np.mean(input_set[:,0])
                #mean2 = np.mean(input_set[:,1])
                #mean3 = np.mean(clustered_xy[:,0])
                #mean4 = np.mean(clustered_xy[:,1])
                #print "medias:" + " " + str(mean1) + " " + str(mean2) + " " + str(mean3) + " " + str(mean4) 
                #var1 = np.var(input_set[:,0])
                #var2 = np.var(input_set[:,1])
                #var3 = np.var(clustered_xy[:,0])
                #var4 = np.var(clustered_xy[:,1])
                #print "varianzas:" + " " + str(var1) + " " + str(var2) + " " + str(var3) + " " + str(var4)


def main():
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
