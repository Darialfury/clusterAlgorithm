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
flag_in = []
flag_out = []

temporal_in = []
temporal_out = []
count_in = 0
count_out = 0


def cluster_features(X, clustered_pixels, vis2, angles):
    global count_in, count_out, temporal_in, temporal_out
    del clustered_pixels[0]
    vis3 = vis2.copy()
    clustered_xy = np.array(clustered_pixels)
    result = []
    people_in = []
    people_out = []
    track_exist = False
    z = np.array([0, 0, 0])
    #get colors
    rainbow = create_colors()

    X = np.delete(X, 0, 0)
    angles = np.delete(angles, 0, 0)
    col1 = angles[:, 0]
    col2 = clustered_xy[:, 0]
    col3 = clustered_xy[:, 1]

    #join all variables in one matrix
    data_set = np.array([col1,col2,col3])
    data_set = data_set.T

    db = DBSCAN(eps = 120).fit(data_set)
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
            x = clustered_xy[index]
            if k != -1:
                track_exist = True
                cv2.circle(vis3, clustered_pixels[index], 2, rainbow[color_ind],2)
                cv2.line(vis2, (1,250),(639,250),(0,0,255),2)
                cv2.line(vis2, (1,300),(639,300),(0,0,255),2)
                cv2.line(vis2, (1,230),(639,230),(255,0,255),2)
                cv2.line(vis2, (1,180),(639,180),(255,0,255),2)
                x1 = X[index][1]
                x2 = clustered_pixels[index][0]
                x3 = clustered_pixels[index][1]
                c = np.array([x1,x2,x3])
                z = np.vstack((z,c))
            if index in core_samples and k != -1:
                markersize = 14
            else:
                markersize = 6
        if k != -1:
            z = np.delete(z, 0, 0)
            result.append(z)
            z = np.array([0,0,0])
    #search for niggas in area
    if track_exist:
        color_ind = 0
        for person in result:
            color_ind += 1
            row = np.mean(person,axis=0)
            cv2.circle(vis2, (int(row[1]),int(row[2])), 2, rainbow[color_ind],2)
            if (row[0] > 0) and (250 < row[2] < 300):
                people_in.append((row[1],row[2]))
            if (row[0] < 0) and (180 < row[2] < 230):
                people_out.append((row[1],row[2]))
        #check how many entered or got out
        if len(temporal_in) > len(people_in):
            count_in += len(temporal_in) - len(people_in)
        temporal_in = people_in
        people_in = []
        if len(temporal_out) > len(people_out):
            count_out += len(temporal_out) - len(people_out)
        temporal_out = people_out
        people_out = []
    #Visualize counting on frame
    string = "Counter goes: In = " + str(count_in) + " Out = " +  str(count_out)
    cv2. putText(vis2, string , (20, vis2.shape[0] - 20),cv2.FONT_HERSHEY_PLAIN,2.3, (255, 255, 255))
    cv2.imshow("person detected", vis2)
    cv2.imshow("group people", vis3)
    cv2.waitKey(5)


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
            angles = np.array([0])
            clustered_pixels = []
            clustered_pixels.append((1,1))
            
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
                        angles = ((np.vstack((angles, [ ((np.arctan2(flowy,flowx))/np.pi)*180 ]))))
                        #last coordinates pixels which forms optical flow
                        clustered_pixels.append(temporal_tracker[-1])
                        

                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                cv2.imshow('lk_track', vis)
                if (len(input_set)) > 2:
                    cluster_features(input_set, clustered_pixels, vis2, angles)
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

            ch = 0xFF & cv2.waitKey(10)
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
