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
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 500,
                       qualityLevel = 0.08,
                       minDistance = 1,
                       blockSize = 7)


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

#flags for counting
flag_in = False
flag_out = False
count_in =  0
count_out = 0

def cluster_features(X, clustered_pixels, vis2):
    global count_in,count_out
    del clustered_pixels[0]
    clustered_xy = np.array(clustered_pixels) 
    result = []
    z = np.array([0,0,0])
    #get colors
    rainbow = create_colors()

    #perform clustering in input_set
    X = np.delete(X, 0, 0)
    #normalize all variables
    #col0 = np.true_divide(X[:,0],max(X[:,0]))
    #col1 = np.true_divide(X[:,1],max(X[:,1]))
    #col2 = np.true_divide(clustered_xy[:,0],max(clustered_xy[:,0]))
    #col3 = np.true_divide(clustered_xy[:,1],max(clustered_xy[:,1]))

    #col0 = X[:,0] / np.linalg.norm(X[:,0])
    #col1 = X[:,1] / np.linalg.norm(X[:,1])
    #col2 = clustered_xy[:,0] / np.linalg.norm(clustered_xy[:,0])
    #col3 = clustered_xy[:,1] / np.linalg.norm(clustered_xy[:,1])

    #join all variables in one matrix
    #data_set = np.array([col0,col1,col2,col3])
    #data_set = data_set.T
    #1) opcion: 70 pixeles en y, 120 pixelesi
    #2) opcion: 431 en y    <----

    db = DBSCAN(eps = 120).fit(clustered_xy)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    #plot clustering
    unique_labels = set(labels)
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    color_ind = 0
    #temp_col = colors
    for k, col in zip(unique_labels, colors):
        #print "1: " + str(col)
        #print "2: " + str(temp_col)
        #if temp_col != col:
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
            #print "color es: " + str(color_ind)
            x = clustered_xy[index]
            #cv2.circle(vis2,clustered_pixels[index],2,rainbow[color_ind],2)
            if k != -1:
                cv2.circle(vis2,clustered_pixels[index],2,rainbow[color_ind],2)
                x1 = X[index][1]
                x2 = clustered_pixels[index][0]
                x3 = clustered_pixels[index][1]
                c = np.array([x1,x2,x3])
                z = np.vstack((z,c))
            if index in core_samples and k != -1:
                markersize = 14
                #cv2.circle(vis2,clustered_pixels[index],2,rainbow[color_ind],2)
            else:
                markersize = 6
            #pl.plot(x[0], x[1], 'o', markerfacecolor=col,
            #markeredgecolor='k', markersize=markersize)
        if k != -1:
            z = np.delete(z, 0, 0)
            result.append(z)
    for person in result:
        row = np.mean(person,axis=0)
        print "xxx: " + str(row)
        if (row[0]>0) and (403<row[2]<441):
            count_in += 1
    print "end"
    draw_str(vis2, (20, 20), 'count in: %d' % count_in)
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
                    if ((-6 > flowx) or (flowx > 6)) and ((-14 > flowy) or (flowy > 14)):
                        input_set = np.vstack((input_set, [flowx, flowy]))

                        #last coordinates pixels which forms optical flow
                        clustered_pixels.append(temporal_tracker[-1])
                        

                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
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
            #if ch & 0xFF == ord("p"):
                #cluster_features(input_set, clustered_pixels, vis2)


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
