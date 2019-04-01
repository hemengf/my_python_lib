import subprocess

filename = "/cygdrive/c/Lib/site-packages/matplotlib"
cmd = ['cygpath','-w',filename]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
output = proc.stdout.read()
#output = output.replace('\\','/')[0:-1] #strip \n and replace \\

print output
