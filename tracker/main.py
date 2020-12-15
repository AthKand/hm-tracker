# -*- coding: utf-8 -*-
# @Author: Atharva
# @Date:   2020-09-22 15:22:01
# @Last Modified by:   Atharva
# @Last Modified time: 2020-11-05 15:57:01

from track import Tracker
from datetime import date
from pathlib import Path

import logging
import os

if __name__ == "__main__":
    today  = date.today()
    # parser = argparse.ArgumentParser(description = 'Enter required paths')
    # parser.add_argument('-i', '--id',  type = str, help = 'enter unique file id')
    # args = parser.parse_args()
    
    enter = input('Enter unique file name: ')
    file_id = '' if not enter else enter

    print('#\nHex-Maze Tracker: v1.4\n#\n')
    import gui

    #logger intitialisations
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    parent_path = Path(os.getcwd()).parent

    logfile_name = '{}/logs/log_{}_{}.log'.format(str(parent_path), str(today), file_id)


    fh = logging.FileHandler(str(logfile_name))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 


    node_list = Path('resources/node_list_new.csv').resolve()
    vid_path = gui.vpath
    logger.info('Video Imported: {}'.format(vid_path))
    print('creating log files...')
    
    Tracker(vp = vid_path, sp = gui.save_path, nl = node_list, file_id = file_id).run_vid()
