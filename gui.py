from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog

from tracker import Tracker


def fopen(message, ftype):
	if ftype == 'node':
		root.filename = filedialog.askopenfilename(title = message, filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")])
	elif ftype == 'video':
		root.filename = filedialog.askopenfilename(title = message, filetypes = [("Video Files", [".mp4", ".avi", ".mkv"]), ("All Files", "*.*")])

	return root.filename

if __name__ == '__main__':
	print('Open gui from tracker.py')

else:
	root = Tk()
	root.withdraw()
	#root.title('TrackerGUI')
	
	vfile = fopen('Select Video File', ftype = 'video')
	tvpath = Path(vfile).resolve()

	print('vid file imported...')

	print('selecting node file...')

	nfile = fopen('Select Node List', ftype = 'node')
	tnpath = Path(nfile).resolve()

	#root.mainloop()

	



