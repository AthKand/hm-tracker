from threading import Thread
import cv2

class FrameGrab:

	def __init__(self, source = 0):
		self.cap = cv2.VideoCapture(source)
		self.ret, self.frame = self.cap.read()
		self.stopped = False
		self.paused = False

	def start(self):
		Thread(target = self.get_frame, args = ()).start()
		return self

	def get_frame(self):
		while not self.stopped:
			if not self.paused:
				if not self.ret:
					self.stop()

				else:
					self.ret, self.frame = self.cap.read()

			else:
				continue				

	def stop(self):
		self.stopped = True
		


