# from - https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
from queue import Queue
from threading import Thread 

import cv2

class VidStream:
	def __init__(self, vid):
		self.stream = cv2.VideoCapture(vid)
		self.stopped = False
		
		self.Q = Queue(maxsize = 1000000)

	def start(self):
		t = Thread(target = self.update)
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:

			if self.stopped:
				return

			if not self.Q.full():
				ret , frame = self.stream.read()
				if not ret:
					self.stop()
					return 

				self.Q.put(frame)


	def more(self):
		return self.Q.qsize() > 0

	def read(self):
		return self.Q.get()

	def stop(self):
		self.stopped = True




