from threading import Thread
import logging
import cv2

class FrameDisplay:

	def __init__(self, tracker, frame = None):
		self.frame = frame
		self.stopped = False
		self.paused = False
		self.tk = tracker
		self._SAVE_FLAG = 0

	def start(self):
		Thread(target = self._display_frame, args = ()).start()
		return self

	def _display_frame(self):
		while not self.stopped:
			cv2.imshow('Tracker', self.frame)
			self._check_button_press()

	def _check_button_press(self):
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			self.stop()

		elif key == ord('s'):
			self.tk.record_detections = not self.tk.record_detections
			self._SAVE_FLAG += 1 

			if self._SAVE_FLAG / 2 == 1:
				self.tk.save_to_file(self.tk.save_file)
				self.tk.saved_nodes = []
				self.tk.node_pos = []
				self.tk.centroid_list = []
				self.tk.trialnum += 1
				self._SAVE_FLAG = 0
			else:
				logging.info('Recording Trial {}'.format(self.tk.trialnum))

		elif key == ord(' '):
			self.paused = not self.paused

	def stop(self):
		self.stopped = True
