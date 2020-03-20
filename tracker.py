# -*- coding: utf-8 -*-
from tools import mask, vid_writer
from itertools import groupby
from datetime import date 
from pathlib import Path 

import cv2
import math
import argparse
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
BG_SUB = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 150, detectShadows = False)

FONT = cv2.FONT_HERSHEY_TRIPLEX


def points_dist(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist


class Tracker:
    def __init__(self, vp, nl, vsp, nsp):
        self.vid_save_path = vsp
        self.node_save_path = nsp
        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))

        self.paused = False
        self.frame = None
        self.disp_frame = None
        self.total_detections = -1
        self.num = 1

        self.pos_centroid = None
        self.cent_list = []
        self.node_pos = []
        self.node_id = []
        self.saved_nodes = []
        
        self.backsub = None
        self.morph = None
        self.record_detections = False

        self.vid_writer = vid_writer.Writer() 
        self.hex_mask = mask.create_mask(self.node_list)

        self.run_vid()

    
    def run_vid(self):
        save_flag = 0

        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    self.paused = True

            if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (588, 356)) 
                self.preprocessing(self.disp_frame)
                self.annotate_frame(self.disp_frame)
                cv2.imshow('Tracker', self.disp_frame) 

            if save_flag / 2 == 1:
                
                fname = self.node_save_path + '{}'.format('.txt')
                self.save_to_file(fname)
                self.saved_nodes = []
                self.node_pos = []
                self.num += 1
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


    def preprocessing(self, frame):
        #treat returned frame as np array
        frame = np.array(frame)
        for i in range(0,3):
            frame[:, :, i] = frame[:, :, i] * self.hex_mask

        self.backsub = BG_SUB.apply(frame)
        self.morph = cv2.morphologyEx(self.backsub, cv2.MORPH_HITMISS, KERNEL)
        self.morph = cv2.morphologyEx(self.morph, cv2.MORPH_CLOSE, KERNEL)

        self.find_contours(self.morph)


    def find_contours(self, frame):
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections_in_frame = 0
        cx_mean= []
        cy_mean = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:
                contour_moments = cv2.moments(contour)
                detections_in_frame += 1
                self.total_detections += 1
                cx = int(contour_moments['m10'] / contour_moments['m00'])
                cy = int(contour_moments['m01'] / contour_moments['m00'])
                cx_mean.append(cx)
                cy_mean.append(cy)
            else:
                continue

        if self.total_detections:
                if detections_in_frame != 0:
                    self.pos_centroid = (int(sum(cx_mean) / len(cx_mean)), 
                						int(sum(cy_mean) / len(cy_mean)))
                    self.cent_list.append(self.pos_centroid)                    
                else:
                    self.pos_centroid = self.cent_list[len(self.cent_list) - 1]

    @staticmethod
    def annotate_node(frame, point, node):
        cv2.circle(frame, point, 2, color = (0, 69, 255), thickness = -1)
        cv2.putText(frame, str(node), (point[0] + 2, point[1] + 2), 
        			fontScale=0.35, fontFace=FONT, color = (0, 69, 255), thickness=1,
                	lineType=cv2.LINE_AA)


    def annotate_frame(self, frame):
        nodes_dict = mask.create_node_dict(self.node_list)
        record = self.record_detections and not self.paused

        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 10:
                    if record: 
                        self.saved_nodes.append(node_name) 
                        self.node_pos.append(nodes_dict[node_name])
        
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i])

        if record:
            savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.num), (60,80), 
                        fontFace = FONT, fontScale = 0.50, color = (255,255,255), thickness = 1)
            self.vid_writer.write(frame = frame, saved_to = savepath)
            cv2.putText(frame,'Currently writing to file: ' + savepath, (60,60), 
                        fontFace = FONT, fontScale = 0.50, color = (255,255,255), thickness = 1)
    
    #save_to_file
    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)         
            file.writelines('%s, ' % items for items in savelist)
            file.write('\n')


if __name__ == "__main__":
    today  = date.today()
    parser = argparse.ArgumentParser(description = 'Enter required paths')

    parser.add_argument('-v', '--video' , type = str, help = 'path to video file')
    parser.add_argument('-n', '--node', type = str , help = 'path to node file')
    args = parser.parse_args()

    vid_path = Path(args.video).resolve()
    if args.node is not None:
        node_list = Path(args.node).expanduser().resolve()
        if not node_list.exists():
            raise FileNotFoundError("The node list file does not exist!")
    else:
        node_list = Path('C:/Users/students/src/hm-tracker/node_list_new.csv')

    nsp = "C:/Users/students/Desktop/tracker-saved/%s" % (str(today))
    vsp = "C:/Users/students/Desktop/tracker-saved/%s" % (str(today))

    Tracker(vp = vid_path, nl = node_list , vsp = vsp, nsp = nsp)

    


    
            
        