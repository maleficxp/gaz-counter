#!/usr/bin/env python
# encoding: utf-8
'''
 -- Gaz counter values recognizing

@author:     malefic
@contact:    andrey.shutkin@gmail.com
'''

from gdrive import downloadImageFromGDrive

from models import Image, sess, mylogger
        
if __name__ == '__main__':

    # fetch unrecognized images
    unrecognized_images = sess.query(Image).filter_by(result='').all()  
    for image in unrecognized_images:
        mylogger.info("Process %s" % image.file_name)
        # try to recognize digits using new training data
        image.img = downloadImageFromGDrive(image.download_url)
        image.identifyDigits()
    sess.commit()
    

    
    
    
    
    