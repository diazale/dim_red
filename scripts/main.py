import os
import time
import sys
os.system('pipreqs .')

old_stdout = sys.stdout
log_file = open("dimred_messages.log", "a")
sys.stdout = log_file

start_time = time.time()
os.system('python pca_umap_dimred.py')
print("Total time in sec: ", time.time() - start_time)
