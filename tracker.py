# -*- coding: utf-8 -*-
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
import numpy as np


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
BG_SUB = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 150, detectShadows = False)

FONT = cv2.FONT_HERSHEY_TRIPLEX
RT_FPS = 30

MIN_RAT_SIZE = 5


#find the shortest distance between two points in space
def points_dist(p1, p2):
    
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist


class Tracker:
    def __init__(self, vp, nl, vsp, nsp, file_id):
        self.vid_save_path = vsp + '_' + file_id
        self.node_save_path = nsp + '_' + file_id
        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))

        self.paused = False
        self.frame = None
        self.disp_frame = None
        self.total_detections = -1
        self.trialnum = 1

        self.pos_centroid = None
        self.kf_coords = None
        self.centroid_list = deque(maxlen = 100)
        self.node_pos = []
        self.node_id = []
        self.saved_nodes = []
        self.KF_age = 0
        
        self.record_detections = False

        self.vid_writer = vid_writer.Writer() 
        self.hex_mask = mask.create_mask(self.node_list)
        self.KF = kalman_filter.KF()

        self.run_vid()

    
    #process video to calculate rat position and display video 
    def run_vid(self):
        '''Frame by Frame looping of video'''
        save_flag = 0
        logtime = 0
        
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
            fps = 1 / (end - start)
            rfps = fps / RT_FPS
            
            if self.record_detections:
                logtime += rfps
                if int(logtime) >= 10:
                    logger.info('The rat position is: {}'.format(self.pos_centroid))
                    logtime = 0

            #condition to save/log data to file upon second press of 's' key
            if save_flag / 2 == 1:
                fname = self.node_save_path + '{}'.format('.txt')
                self.save_to_file(fname)
                self.saved_nodes = []
                self.node_pos = []
                self.trialnum += 1
                save_flag = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('s'):
                self.record_detections = not self.record_detections
                save_flag += 1

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
                    self.centroid_list.append(self.pos_centroid)
                    #cv2.circle(frame, self.pos_centroid, 20, color = (0, 69, 255), thickness = 1)                    
                else:
                    self.pos_centroid = self.centroid_list[-1]
                    #cv2.circle(frame, self.pos_centroid, 20, color = (0, 69, 255), thickness = 1)

                if len(self.centroid_list) > 2:
                	if points_dist(self.pos_centroid, self.centroid_list[-2]) > 1.5:
                		self.kf_coords = self.KF.estimate()
                		self.pos_centroid = self.centroid_list[-2]

                #self.kf_coords = self.KF.estimate()
                '''
                self.KF_age += 1
                if self.KF_age > 3:
                    self.KF.correct(self.pos_centroid[0], self.pos_centroid[1])
                    self.KF_age = 0
                '''
                    
                #self.kf_coords = self.KF.estimate(self.pos_centroid[0], self.pos_centroid[1])
                #cv2.circle(frame, (kf_coords[0], kf_coords[1]), 20, color = (0, 255, 0), thickness = 1)
                    
    @staticmethod
    def annotate_node(frame, point, node):
        cv2.circle(frame, point, 20, color = (0, 69, 255), thickness = 1)
        cv2.putText(frame, str(node), (point[0] + 2, point[1] + 2), 
        			fontScale=0.5, fontFace=FONT, color = (0, 69, 255), thickness=1,
                	lineType=cv2.LINE_AA)


    def annotate_frame(self, frame):
        nodes_dict = mask.create_node_dict(self.node_list)				#dictionary of node names and corresponding coordinates
        record = self.record_detections and not self.paused				#recording condition


        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 20:
                    if record: 
                        self.saved_nodes.append(node_name)						
                        self.node_pos.append(nodes_dict[node_name])

            #if self.kf_coords is not None:            
            	#cv2.circle(frame, (self.kf_coords[0], self.kf_coords[1]), 10, color = (0, 255, 0), thickness = 1)
            cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]), (self.pos_centroid[0] + 5, self.pos_centroid[1]), 
            	color = (0, 255, 0), thickness = 2)
            cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5), (self.pos_centroid[0], self.pos_centroid[1] + 5), 
            	color = (0, 255, 0), thickness = 2)
            #print(self.centroid_li
        
        #annotate all nodes the rat has traversed 
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i])

        #frame annotations during recording
        if record:
            savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            #self.vid_writer.write(frame = frame, saved_to = savepath)
            cv2.putText(frame,'Currently writing to file: ' + savepath, (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            #cv2.putText(frame, str(self.frame_rate), (60,100), 
                        #fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)	
            if len(self.centroid_list) > 10:	
            	for i in range(1, len(self.centroid_list)):
                	thickness = int(np.sqrt(100 / (i + 1)) * .75)
                	cv2.line(frame, self.centroid_list[i - 1], self.centroid_list[i], 
                         color = (0, 255, 0), thickness = thickness)
            
        elif self.paused:
            cv2.putText(frame,'Trial:' + str(self.trialnum), (60,80), 
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

    #parser.add_argument('-v', '--video' , type = str, help = 'path to video file')
    #parser.add_argument('-n', '--node', type = str , help = 'path to node file')
    parser.add_argument('-i', '--id',  type = str, help = 'enter unique file id')
    args = parser.parse_args()
    
    file_id = '' if not args.id else args.id 
    
    #vid_path = Path(args.video).resolve()

    '''
    if args.node is not None:
        node_list = Path(args.node).expanduser().resolve()
        if not node_list.exists():
            raise FileNotFoundError("The node list file does not exist!")
    else:
        node_list = Path('C:/Users/students/src/hm-tracker/node_list_new.csv')
    '''
    import gui

    vid_path = gui.tvpath
    node_list = gui.tnpath

    nsp = "C:/Users/students/Desktop/tracker-saved/%s" % (str(today))
    vsp = "C:/Users/students/Desktop/tracker-saved/%s" % (str(today))

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    logfile_name = 'log_{}_{}.log'.format(str(today), file_id)

    fh = logging.FileHandler(str(logfile_name))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    Tracker(vp = vid_path, nl = node_list , vsp = vsp, nsp = nsp, file_id = file_id)

    


    
            
        