from __future__ import division
from ctypes import windll, create_string_buffer
import time
import sys
import struct
import subprocess

def progressbar_win_console(cur_iter, tot_iter, deci_dig):
    """
    Presents the percentage and draws a progress bar.
    Import at the begining of a file. Call at the end of a loop.

    cur_iter: current iteration number. Counted from 1.
    tot_iter: total iteration number.
    deci_dig: decimal digits for percentage number.
    
    Works for windows type console.
    """

    csbi = create_string_buffer(22)
    h = windll.kernel32.GetStdHandle(-11)
    res = windll.kernel32.GetConsoleScreenBufferInfo(h,csbi)
    (_,_,_,_,_,left,_,right,_,_,_) = struct.unpack('11h',csbi.raw)
    # Grab console window width. 
    # Modified from http://stackoverflow.com/questions/17993814/why-the-irrelevant-code-made-a-difference
    console_width = right-left+1
    bar_width = int(console_width * 0.8)
    tot_dig = deci_dig + 4 #  to make sure 100.(4 digits) + deci_dig
    percentage = '{:{m}.{n}f}%'.format(cur_iter*100/tot_iter, m = tot_dig, n = deci_dig)
    numbar = bar_width*cur_iter/tot_iter
    numbar = int(numbar)
    sys.stdout.write(percentage)
    sys.stdout.write("[" + unichr(0x2588)*numbar + " "*(bar_width-numbar) + "]")
    sys.stdout.flush()
    sys.stdout.write('\r')
    if cur_iter == tot_iter:
        sys.stdout.write('\n')

def progressbar_tty(cur_iter, tot_iter, deci_dig):
    """
    Presents the percentage and draws a progress bar.
    Import at the begining of a file. Call at the end of a loop.

    cur_iter: current iteration number. Counted from 1.
    tot_iter: total iteration number.
    deci_dig: decimal digits for percentage number.
    
    Works for linux type terminal emulator.
    """

    #rows, columns = subprocess.check_output(['stty', 'size']).split()
    # Grab width of the current terminal. 
    # Modified from http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    # won't work inside vim using "\r"

    columns = subprocess.check_output(['tput','cols'])
    rows = subprocess.check_output(['tput','lines'])
    columns = int(columns)
    bar_width = int(columns* 0.8)
    tot_dig = deci_dig + 4 #  to make sure 100.(4 digits) + deci_dig
    percentage = '{:{m}.{n}f}%'.format(cur_iter*100/tot_iter, m = tot_dig, n = deci_dig)
    numbar = bar_width*cur_iter/tot_iter
    numbar = int(numbar)
    sys.stdout.write(percentage)
    sys.stdout.write("[" + u'\u2588'.encode('utf-8')*numbar + " "*(bar_width-numbar) + "]")
    sys.stdout.flush()
    sys.stdout.write('\r')
    if cur_iter == tot_iter:
        sys.stdout.write('\n')
