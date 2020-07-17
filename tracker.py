# -*- coding: utf-8 -*-
'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
			 in the novel Hex-Maze experiment. Serves as a 
			 replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute	
			  Radboud University, Nijmegen

Author(s): Atharva Kand

Version: v1.02
Last Updated: 17th July, 2020

Last change: Added nearest node to X,Y position to logger

'''

from tools import mask, vid_writer, kalman_filter
from itertools import groupby
from datetime import date 
from pathlib import Path 
from collections import deque

import cv2
import math
import time
import logging
import argparse
import os
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
BG_SUB = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 150, detectShadows = False)

FONT = cv2.FONT_HERSHEY_TRIPLEX
RT_FPS = 25

MIN_RAT_SIZE = 5


#find the shortest distance between two points in space
def points_dist(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist


class Tracker:
    def __init__(self, vp, nl, file_id):
        nsp = str(date.today()) + '_' + file_id
        self.save = os.path.join(gui.save_path, nsp)
        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))

        self.paused = False
        self.frame = None
        self.frame_rate = 0
        self.disp_frame = None
        self.total_detections = -1
        self.trialnum = 1

        self.pos_centroid = None
        self.kf_coords = None
        self.centroid_list = deque(maxlen = 500)
        self.node_pos = []
        self.node_id = []
        self.saved_nodes = []
        self.KF_age = 0
        
        self.record_detections = False
 
        self.hex_mask = mask.create_mask(self.node_list)
        self.KF = kalman_filter.KF()

        self.run_vid()

    #process video to calculate rat position and display video 
    def run_vid(self):
        '''Frame by Frame looping of video'''
        save_flag = 0
        logtime = 0
        print('loading tracker...\n')
        time.sleep(2.0)

        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    self.paused = True

            start = time.time()
            
            #process and display frame
            if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (1176, 712)) 
                self.preprocessing(self.disp_frame)
                self.annotate_frame(self.disp_frame)
                cv2.imshow('Tracker', self.disp_frame)

            end = time.time()
            diff = end - start
            fps = 1 / diff
            rfps = round(fps / RT_FPS, 2)
            self.frame_rate = rfps
            
            if self.record_detections:
                logtime += rfps
                if logtime >= 1:
                    if self.pos_centroid is not None:
                        logger.info('The rat position is: {} @ {}'. format(self.pos_centroid, self.saved_nodes[-1]))
                    else:
                        logger.info('The rat position is: {}'.format(self.pos_centroid))
                    logtime = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
            	print('#Program ended by user')
            	break

            elif key == ord('s'):
                self.record_detections = not self.record_detections
                save_flag += 1

                #condition to save/log data to file upon second press of 's' key
                if save_flag / 2 == 1:
                	fname = self.save + '{}'.format('.txt')
                	self.save_to_file(fname)
                	self.saved_nodes = []
                	self.node_pos = []
                	self.centroid_list = []
                	self.trialnum += 1
                	save_flag = 0
                else:
                	logger.info('Recording Trial {}'.format(self.trialnum))
            elif key == ord(' '):
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()

    #pre-process frame - apply mask, bg subtractor and morphology operations
    def preprocessing(self, frame):
        
        frame  = np.array(frame)
        #apply mask on frame from mask.py 								
        for i in range(0,3):
            frame[:, :, i] = frame[:, :, i] * self.hex_mask		

        #background subtraction and morphology
        backsub = BG_SUB.apply(frame)
        morph = cv2.morphologyEx(backsub, cv2.MORPH_HITMISS, KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, KERNEL)

        self.find_contours(morph)


    def find_contours(self, frame):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)          #_, cont, _ in new opencv version
        detections_in_frame = 0
        cx_mean= []
        cy_mean = []

        for contour in contours:
            area = cv2.contourArea(contour)

            
            if area > MIN_RAT_SIZE:
                contour_moments = cv2.moments(contour)
                cx = int(contour_moments['m10'] / contour_moments['m00'])
                cy = int(contour_moments['m01'] / contour_moments['m00'])
                cx_mean.append(cx)
                cy_mean.append(cy)

                detections_in_frame += 1
                self.total_detections += 1
            else:
            	continue
            	

        if self.total_detections:
                if detections_in_frame != 0:
                    self.pos_centroid = (int(sum(cx_mean) / len(cx_mean)), 
                						int(sum(cy_mean) / len(cy_mean)))
                    if self.record_detections:
                    	self.centroid_list.append(self.pos_centroid)
                    #cv2.circle(frame, self.pos_centroid, 20, color = (0, 69, 255), thickness = 1)                    
                else:
                	if self.record_detections and self.centroid_list:
                		self.pos_centroid = self.centroid_list[-1]
                    #cv2.circle(frame, self.pos_centroid, 20, color = (0, 69, 255), thickness = 1)

                if len(self.centroid_list) > 2:
                	if points_dist(self.pos_centroid, self.centroid_list[-2]) > 1.5:
                		self.kf_coords = self.KF.estimate()
                		self.pos_centroid = self.centroid_list[-2]
    @staticmethod
    def annotate_node(frame, point, node):
        cv2.circle(frame, point, 20, color = (0, 69, 255), thickness = 1)
        cv2.putText(frame, str(node), (point[0] + 2, point[1] + 2), 
        			fontScale=0.5, fontFace=FONT, color = (0, 69, 255), thickness=1,
                	lineType=cv2.LINE_AA)


    def annotate_frame(self, frame):
        nodes_dict = mask.create_node_dict(self.node_list)				#dictionary of node names and corresponding coordinates
        record = self.record_detections and not self.paused

        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 20:
                    if record: 
                        self.saved_nodes.append(node_name)						
                        self.node_pos.append(nodes_dict[node_name])
        
        #annotate all nodes the rat has traversed
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i])

        # for node in nodes_dict:
        #     cv2.circle(frame, nodes_dict[node], 20, color = (0,255,255), thickness = 2)
        #     cv2.putText(frame, str(node), (nodes_dict[node][0] + 2, nodes_dict[node][1] + 2), 
        #             fontScale=0.5, fontFace=FONT, color = (0, 0, 255), thickness=1,
        #             lineType=cv2.LINE_AA)


        #frame annotations during recording
        if record:
            #savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Currently writing to file...', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame, str(self.frame_rate), (1110,690), 
                        fontFace = FONT, fontScale = 0.75, color = (0,0,255), thickness = 1)	

            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1], 
                             color = (255, 0, 255), thickness = 2)

            if self.pos_centroid is not None:
            	cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]), (self.pos_centroid[0] + 5, self.pos_centroid[1]), 
            	color = (0, 255, 0), thickness = 2)
            	cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5), (self.pos_centroid[0], self.pos_centroid[1] + 5), 
            	color = (0, 255, 0), thickness = 2)
                    
        elif self.paused:
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Paused', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
                        
    
    #save recorded nodes to file
    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)         
            file.writelines('%s,' % items for items in savelist)
            file.write('\n')



if __name__ == "__main__":
    today  = date.today()
    parser = argparse.ArgumentParser(description = 'Enter required paths')
    parser.add_argument('-i', '--id',  type = str, help = 'enter unique file id')
    args = parser.parse_args()
    
    file_id = '' if not args.id else args.id

    print('#\nHex-Maze Tracker: v1.02\nLast updated: 11th June 2020\n#\n')
    import gui

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    logfile_name = 'log_{}_{}.log'.format(str(today), file_id)

    fh = logging.FileHandler(str(logfile_name))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 

    node_list = Path('node_list_new.csv').resolve()
    vid_path = gui.vpath
    logger.info('Video Imported: {}'.format(vid_path))
    print('creating log files...')

    
    Tracker(vp = vid_path, nl = node_list, file_id = file_id)


    


    
            
        