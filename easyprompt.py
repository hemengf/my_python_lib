import sys
from colorama import init, Fore, Style

class easyprompt:
    def __init__(self):
	init()
	self.count = 0
    def __str__(self):
	self.count += 1
	print(Fore.GREEN + '(%d)>>>>>>>>>>>>>>>' % self.count)
	print(Style.RESET_ALL)

sys.ps1 = easyprompt()
