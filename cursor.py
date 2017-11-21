import ctypes
import time

start_time = time.time()
# see http://msdn.microsoft.com/en-us/library/ms646260(VS.85).aspx for details
ctypes.windll.user32.SetCursorPos(100, 40)
ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up
ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up
time_1 = time.time()
print '1st file opened'
ctypes.windll.user32.SetCursorPos(200, 40)
ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up
ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up
print '2nd file opened'
time_2 = time.time()
print start_time
print '%.5f' % time_1
print '%.5f' % time_2
