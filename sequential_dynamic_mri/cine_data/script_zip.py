import os
import time

dirs = ['cine_freebreathing','cine_breathhold']

iterations = 20
curr_iter = 0

#while curr_iter < iterations:
#    print("Iteration:", curr_iter)
tot_files = 0
for folder in dirs:
    files = os.listdir(folder)
    files = [element for element in files if 'mat' in element]
    print(len(files)/2)
    tot_files += len(files)/2

print(tot_files)
'''
        curr_dir = os.getcwd()
        os.chdir(folder)
        os.system(' '.join(["zip", folder+".zip"] + files))
        os.sytem(' '.join(["rm"]+files))
        os.chdir(curr_dir)

    time.sleep(60*5)
'''

