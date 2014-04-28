import os
import glob
import cv2

def getImagesFromGDrive():
    os.chdir(os.path.dirname(__file__)+"/../photos")
    infos = []
    for jpg in glob.glob("*.jpg"):
        infos.append({'title':jpg, 'downloadUrl':'', 'webContentLink':'', 'full_path':os.path.dirname(__file__)+"/../photos/"+jpg})
    return infos, None
    
def createImageFromGDriveObject (img_info, http=None):
    return cv2.imread(img_info['full_path'])