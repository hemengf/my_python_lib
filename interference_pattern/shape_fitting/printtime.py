#import os
#if os.getenv("TZ"):
#    os.unsetenv("TZ")
from time import strftime, localtime,gmtime,timezone
print strftime("%H_%M_%S",localtime())
print timezone/3600.
