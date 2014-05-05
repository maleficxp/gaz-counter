#!/usr/bin/env python
# encoding: utf-8
'''
 -- Gaz counter values recognizing

@author:     malefic
@contact:    andrey.shutkin@gmail.com
'''

import re

from gdrive import downloadImageFromGDrive

from models import Image, sess, mylogger
        
if __name__ == '__main__':

    # fetch unrecognized images
    unrecognized_images = sess.query(Image).filter_by(result='').all()  
    for image in unrecognized_images:
        mylogger.info("Process %s" % image.file_name)
        # try to recognize digits using new training data
        m = re.search('id=(.*)', image.img_link)
        image.img = downloadImageFromGDrive(image.download_url, file_id=m.group(1))        
        image.identifyDigits()
    sess.commit()
    

    
    
    
    
    