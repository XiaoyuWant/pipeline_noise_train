import os
import glob
'''
Delete Images <1kb which can not be trained
'''
cnt=0

folder_path="/root/dataset/refresh1000/"
folders=glob.glob(folder_path+"*")
print(len(folders))
for folder in folders:
    files = glob.glob(folder+'/*.jpg')
    for item in files:
        fsize = os.path.getsize(item)
        if(fsize<1000):
            os.remove(item)
            print("Remove\t{}".format(item))
            cnt+=1

print("delete number:{}".format(cnt))
