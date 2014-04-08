#!/usr/bin/env python
# encoding: utf-8
'''
 -- Gaz counter values recognizing

@author:     malefic
@contact:    andrey.shutkin@gmail.com
'''

import sys
import os

from models import getImage, sess, mylogger, getLastRecognizedImage
from gdrive import getImagesFromGDrive, createImageFromGDriveObject

if __name__ == '__main__':

    images, http = getImagesFromGDrive()
    
    # Process each photo
    for img_info in images:

        img = createImageFromGDriveObject (img_info, http)
        file_name = img_info['title']
        
        mylogger.info("Process %s" % file_name)

        # create image object     
        try:    
            dbimage = getImage(os.path.basename(file_name))
            dbimage.img = img
            dbimage.download_url = img_info["downloadUrl"]
            dbimage.img_link = img_info['webContentLink'].replace('&export=download','')
        except ValueError as e:
            print e
            continue
             
        # try to recognize image
        if dbimage.identifyDigits():
            mylogger.info("Result is %s" % dbimage.result)
        
        sess.commit()
    
    

    
    
    
    
    