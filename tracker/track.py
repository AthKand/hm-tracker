# -*- coding: utf-8 -*-
# @Author: Atharva
# @Date:   2020-09-21 15:40:40
# @Last Modified by:   Mahe

# @Last Modified time: 2020-11-05 17:40:52



'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
			 in the novel Hex-Maze experiment. Serves as a 
			 replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute	
			  Radboud University, Nijmegen

Author(s): Atharva Kand
'''

from itertools import groupby
from datetime import date 
from pathlib import Path 
from collections import deque
from utils.frame_grabber import FrameGrab
from utils.frame_display import FrameDisplay

import cv2
import math
import time
import logging
import argparse
import os
import numpy as np
import csv


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
    def __init__(self, vp, sp, nl, file_id):
        '''Tracker class initialisations'''
        nsp = str(date.today()) + '_' + file_id
        self.save_file = os.path.join(sp, nsp) + '{}'.format('.txt')
        self.node_list = str(nl)
        #self.cap = cv2.VideoCapture(str(vp))

        #rat meta-data
        self.rat = input("Enter Rat Number: ")
        self.date = input("Enter date of trial: ")

        self.grabber = FrameGrab(str(vp)).start()
        self.shower = FrameDisplay(self, self.grabber.frame).start()

        self.paused = False
        self.frame = None
        self.frame_rate = 0
        self.disp_frame = None
        self.total_detections = -1
        self.trialnum = 1

        self.pos_centroid = None
        self.kf_coords = None
        self.centroid_list = deque(maxlen = 500)          #change maxlen value to chnage how long the pink line is
        self.node_pos = []
        self.node_id = []
        self.saved_nodes = []
        self.KF_age = 0
        
        self.record_detections = False

        mask_loc = str(Path('resources/hex_mask.npy').resolve())
        self.hex_mask = np.load(mask_loc)

    #process and display video 
    def run_vid(self):
        '''
        Frame by Frame looping of video
        
        '''
        save_flag = 0
        logtime = 0
        print('loading tracker...\n')
        time.sleep(2.0)

        with open(self.save_file, 'a+') as file:
        	file.write(f'Rat Number: {self.rat}, Date: {self.date} \n')

        while True:

            if self.shower.paused:
                self.grabber.paused = True
            else:
                self.grabber.paused = False

            if self.grabber.stopped or self.shower.stopped: 
                self.shower.stop()
                self.grabber.stop()
                print('Program ended by user')
                break

            start = time.time()

            self.frame = self.grabber.frame
            
            #process and display frame
            if self.frame is not None:
                #self.disp_frame = self.frame.copy()
                self.frame = cv2.resize(self.frame, (1176, 712)) 
                self.preprocessing(self.frame)
                self.annotate_frame(self.frame)
                self.shower.frame = self.frame

            end = time.time()
            diff = end - start
            fps = 1 / diff
            rfps = round(fps / RT_FPS, 2)
            self.frame_rate = rfps


            #log present centroid position if program is in 'save mode'
            if self.record_detections:
                logtime += rfps
                if logtime >= 1:
                    if self.pos_centroid is not None:
                    	if self.saved_nodes:
                        	logging.info(f'The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')
                    	else:
                    		logging.info(f'The rat position is: {self.pos_centroid}')
                    else:
                        logging.info('Rat not detected')
                    logtime = 0

        self.grabber.cap.release()
        cv2.destroyAllWindows()

    def preprocessing(self, frame):
        '''
        pre-process frame - apply mask, bg subtractor and morphology operations

        Input: Frame (i.e image to be preprocessed)
        '''
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
        '''
        Function to find contours

        '''
        _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)          #_, cont, _ in new opencv version
        detections_in_frame = 0

        #create lists of centroid x and y means 
        cx_mean= []             
        cy_mean = []

        #find contours greater than the minimum area and 
        #caluclate means of the of all such contours 
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
            	
        #find centroid position by calculating means of x and y of contour centroids
        #if the program is on 'save mode', add centroid poisitions to list. if no 
        #detections are in frame, assume rat is stationary, i.e centroid = previous 
        #centroid. if distance between prev centroid and centroid > 2 pixels , 
        #centroid = previous centroid  
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
                		#self.kf_coords = self.KF.estimate()
                		self.pos_centroid = self.centroid_list[-2]


    @staticmethod
    def annotate_node(frame, point, node):
        '''Annotate traversed nodes on to the frame

        Input: Frame (to be annotated), Point: x, y coords of node, Node: Node name
        '''
        cv2.circle(frame, point, 20, color = (0, 69, 255), thickness = 1)
        cv2.putText(frame, str(node), (point[0] + 2, point[1] + 2), 
        			fontScale=0.5, fontFace=FONT, color = (0, 69, 255), thickness=1,
                	lineType=cv2.LINE_AA)


    
    def annotate_frame(self, frame):
        '''
        Annotates frame with frame information, path and nodes resgistered

        '''
        nodes_dict = self.create_node_dict(self.node_list)				#dictionary of node names and corresponding coordinates
        record = self.record_detections and not self.paused             #condition to go into 'save mode'


        #if the centroid position of rat is within 20 pixels of any node
        #register that node to a list. 
        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 20:
                    if record: 
                        self.saved_nodes.append(node_name)						
                        self.node_pos.append(nodes_dict[node_name])
        
        #annotate all nodes the rat has traversed
        for i in range(0, len(self.saved_nodes)):
        	self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i])

        if record and self.saved_nodes:
            cv2.circle(frame, self.node_pos[-1], 20, color = (255, 69, 0), thickness = 1)
            cv2.putText(frame, str(self.saved_nodes[-1]), (self.node_pos[-1][0] +2, self.node_pos[-1][1] +2),
                fontScale=0.5, fontFace = FONT, color = (255, 69, 0), thickness=1, 
                lineType=cv2.LINE_AA)


        #frame annotations during recording
        if record:
            #savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Currently writing to file...', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame, str(self.frame_rate), (1110,690), 
                        fontFace = FONT, fontScale = 0.75, color = (0,0,255), thickness = 1)	

            #draw the path that the rat has traversed
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

    @staticmethod
    def create_node_dict(node_list):
    	nodes_dict = {}
    	with open(node_list, 'r') as nl:
    		read = csv.reader(nl)
    		for nl_values in read:
    			point = (int(nl_values[1]), int(nl_values[2]))
    			nodes_dict.update({nl_values[0] : point})
    	return nodes_dict


    


    
            
        