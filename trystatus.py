import time
import progressbar
for i in range(400):
    # work
    time.sleep(0.01)
    progressbar.progressbar_tty(i,399,3)
