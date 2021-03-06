import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, PickleType
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import base64
import cv2
import numpy as np
import os
import logging
import sys
from cgitb import grey

dbengine = create_engine('sqlite:///' + os.path.dirname(__file__) + '/../db/images.db', echo=False)

Session = sessionmaker(bind=dbengine)
sess = Session()

Base = declarative_base()

digit_base_h = 24
digit_base_w = 16

mylogger = logging.getLogger('gaz')
mylogger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.dirname(__file__)+'/gaz.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
mylogger.addHandler(fh)
mylogger.addHandler(ch)

# image class
class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    file_name = Column(String)
    img_link = Column(Text)
    download_url = Column(Text)
    check_time = Column(DateTime)
    result = Column(String(8))
    digits = relationship("Digit", backref="image")
    img = None # source image
    digits_img = None # cropped source image
    
    def __init__(self, file_name):
        self.file_name = file_name   
        self.check_time = datetime.datetime.strptime(file_name, "gaz.%Y-%m-%d.%H.%M.%S.jpg")
        self.result = ""
    
    def __repr__(self):
        return "<Image ('%s','%s','%s')>" % (self.id, self.file_name, self.result)
    
    def dbDigit(self, i, digit_img):
        digit = sess.query(Digit).filter_by(image_id=self.id).filter_by(i=i).first()
        if not digit:
            digit = Digit(self, i, digit_img)
            sess.add(digit)
        else:
            digit.body = digit_img
        return digit
    
    def findFeature(self, queryImage, trainImage, trainMask=None, silent=False):
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp_q, des_q = sift.detectAndCompute(queryImage,None)
        kp_t, des_t = sift.detectAndCompute(trainImage, trainMask)
        
        if not kp_q or not kp_t:
            if not silent:
                mylogger.warn("No keypoints found")
            return False

#         img=cv2.drawKeypoints(queryImage, kp_q, outImage=np.zeros((1,1)), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
#             
#         img=cv2.drawKeypoints(trainImage, kp_t, outImage=np.zeros((1,1)), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()                

        # create BFMatcher object
        bf = cv2.BFMatcher()
        
        # Match descriptors.
        matches = bf.knnMatch(des_q, des_t, k=2)
        
        # Apply ratio test
        good = []
        for best_matches in matches:
            if len(best_matches)>1:
                m,n = best_matches
            else:
                m=n=best_matches[0]
            if m.distance < 0.75*n.distance:
                good.append([m])
                
        if len(good)<2:
            if not silent:
                mylogger.warn("Feature not found")
            return False 
        
#         img = cv2.drawMatchesKnn(queryImage, kp_q, trainImage, kp_t, good, outImg=np.zeros((1,1)), flags=2)
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
        
        img_pts = np.float32([ kp_t[m[0].trainIdx].pt for m in good ])
        mean = cv2.reduce(img_pts, 0, cv2.REDUCE_AVG)[0]

        # check dispersion
        disp = 0
        for pt in img_pts:
            delta = 0
            delta += (pt[0]-mean[0]) * (pt[0]-mean[0])           
            delta += (pt[1]-mean[1]) * (pt[1]-mean[1])
            if delta>1000:
                # remove extremums
                np.delete(img_pts, pt)
            else:            
                disp += delta
        
        if len(img_pts)<2:
            if not silent:
                mylogger.warn("G4 not found")
            return False 
        
        # recalc mean
        mean = cv2.reduce(img_pts, 0, cv2.REDUCE_AVG)[0]

#        mylogger.info("Dispersion: %s" % disp)

        if disp>5000:
            if not silent:
                mylogger.warn("Dispersion too big")
            return False 
        
#         cv2.circle(trainImage, (int(mean[0]), int(mean[1])), 5, (0,0,255))
#         cv2.imshow('img',trainImage)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
            
        return (int(mean[0]), int(mean[1]))
        
    
    def extractDigitsFromImage (self):

        # init sample template
        sample_right = cv2.imread(os.path.dirname(__file__)+"/sample_right.jpg", cv2.IMREAD_GRAYSCALE)
        
        img = self.img          
                
        # rotate 90 
        h, w, k = img.shape
        if w>h:            
            M = cv2.getRotationMatrix2D((w/2,h/2),270,1)
            img = cv2.warpAffine(img,M,(w,h))
            # crop black sides
            img = img[0:h, (w-h)/2:h+(w-h)/2]
            h, w, k = img.shape
        
        # resize
        img = cv2.resize(img, dsize=(480, 640), interpolation = cv2.INTER_AREA)
        h, w, k = img.shape
        
        sample_G4 = cv2.imread(os.path.dirname(__file__)+"/sample_G4.jpg")        
        mask = np.zeros(img.shape[:2],np.uint8)
        mask[0:int(h/2),0:w] = 255
        mean = self.findFeature(sample_G4, img, mask)
        if not mean:
            return False
        
        # caclulate "center" point coordinates
        x_center = int(mean[0])
        y_center = int(mean[1]) + 25
        
        # make grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # blur
        kernel = np.ones((3,3),np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

#         cv2.imshow('img',gray)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
        
        # search for edges using Canny
        edges = cv2.Canny(gray, 100, 200)
        
#         cv2.imshow('img',edges)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
        
        # detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=110)
        
        if None == lines:
            mylogger.warn("No lines found")
            return False 
        
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()    
        
        # detect nearest for center horisontal lines
        rho_below = rho_above = np.sqrt(h*h+w*w)
        line_above = None
        line_below = None
        for line in lines:
            rho,theta = line[0]
            sin = np.sin(theta)
            cos = np.cos(theta)
            
            # discard not horisontal
            if (sin<0.75):
                continue
    
            # calculate rho for line parallel to current line and passes through the "center" point             
            rho_center = x_center*cos + y_center*sin
            
            # compare line with nearest line above
            if rho_center>rho and rho_center-rho<rho_above:
                rho_above = rho_center-rho
                line_above = {"rho":rho, "theta":theta, "sin":sin, "cos":cos}
                
#                 x0 = cos*rho
#                 y0 = sin*rho
#                 x1 = int(x0 + 1000*(-sin))
#                 y1 = int(y0 + 1000*(cos))
#                 x2 = int(x0 - 1000*(-sin))
#                 y2 = int(y0 - 1000*(cos))    
#                 cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            
            # compare line with nearest line below
            if rho_center<rho and rho-rho_center<rho_below:
                rho_below = rho-rho_center
                line_below = {"rho":rho, "theta":theta, "sin":sin, "cos":cos}
                
#                 x0 = cos*rho
#                 y0 = sin*rho
#                 x1 = int(x0 + 1000*(-sin))
#                 y1 = int(y0 + 1000*(cos))
#                 x2 = int(x0 - 1000*(-sin))
#                 y2 = int(y0 - 1000*(cos))    
#                 cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
# 
#         x0 = line_above["cos"]*line_above["rho"]
#         y0 = line_above["sin"]*line_above["rho"]
#         x1 = int(x0 + 1000*(-line_above["sin"]))
#         y1 = int(y0 + 1000*(line_above["cos"]))
#         x2 = int(x0 - 1000*(-line_above["sin"]))
#         y2 = int(y0 - 1000*(line_above["cos"]))    
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#             
#         x0 = line_below["cos"]*line_below["rho"]
#         y0 = line_below["sin"]*line_below["rho"]
#         x1 = int(x0 + 1000*(-line_below["sin"]))
#         y1 = int(y0 + 1000*(line_below["cos"]))
#         x2 = int(x0 - 1000*(-line_below["sin"]))
#         y2 = int(y0 - 1000*(line_below["cos"]))    
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#                     
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()     
         
        # check result 
        if line_below==None or line_above==None:
            mylogger.warn("No lines found")
#             cv2.imshow('img',img)
#             key = cv2.waitKey(0)
#             if key==1048603:
#                 sys.exit()            
            return False 
         
        # make lines horizontal
        M = cv2.getRotationMatrix2D((w/2,(line_below["rho"]-line_above["rho"])/2+line_above["rho"]),
                                    (line_above["theta"]+line_below["theta"])/2/np.pi*180-90,1)
        img = cv2.warpAffine(img,M,(w,h))

        # crop image
        img = img[line_above["rho"]-int(w/2*line_above['cos']):line_below["rho"]-int(w/2*line_below['cos']), 0:w]
        h, w, k = img.shape
        
        sample_star = cv2.imread(os.path.dirname(__file__)+"/sample_star.jpg")        
        mask = np.zeros(img.shape[:2],np.uint8)
        mask[0:h,0:int(w/4)] = 255
        mean = self.findFeature(sample_star, img, mask, silent=True)
        
        # crop image
        if mean:
            img = img[0:h, mean[0]:w]
            h, w, k = img.shape
        
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()    

        # binarize using adaptive threshold
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2) 

        # try find arrow using findFeature
        sample_arr = cv2.imread(os.path.dirname(__file__)+"/sample_arr.jpg")        
        mask = np.zeros(img.shape[:2],np.uint8)
        mask[0:h,int(w/2):w] = 255
        mean = self.findFeature(sample_arr, img, mask, silent=True)

        if mean:
            x_right = mean[0]-12
        else:
            # match sample_right template
            res = cv2.matchTemplate(thres,sample_right,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            x_right = max_loc[0]-4

#         cv2.circle(thres, (x_right, 5), 5, (0,0,255))
#         cv2.imshow('img',thres)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()        
        
        kernel = np.ones((5,5),np.uint8)
        thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
        
        # search for left edge position
        x_left=0
        while x_left<w :
            if thres[h/2,x_left]==0:
                break
            x_left+=1
        
        # crop image
        img = img[:, x_left:x_right]
        h, w, k = img.shape
        
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()        
        
        # check ratio
        if float(w)/float(h)<5.5 or float(w)/float(h)>10:
            mylogger.warn("Image has bad ratio: %f" % (float(w)/float(h)))
#             cv2.imshow('img',img)
#             key = cv2.waitKey(0)
#             if key==1048603:
#                 sys.exit()
            return False    
        
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()     
        
        self.digits_img = img
        return True
    
    def splitDigits (self):
    
        if None == self.digits_img:
            if not self.extractDigitsFromImage():
                return False
    
        img = self.digits_img
        h, w, k = img.shape
        
        # split to digits
        for i in range(1,8):
            digit = img[0:h, (i-1)*w/8:i*w/8]
            dh, dw, dk = digit.shape
            # binarize each digit
            digit_gray = cv2.cvtColor(digit,cv2.COLOR_BGR2GRAY)
            
            digit_bin = cv2.adaptiveThreshold(digit_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,-5)

            # remove some noise
            # kernel = np.ones((2,2),np.uint8)
            # digit_bin = cv2.morphologyEx(digit_bin, cv2.MORPH_OPEN, kernel)
            
#             cv2.imshow("digit",digit_bin)
#             k = cv2.waitKey(0)
#             if k==1048603:
#                 sys.exit()            

            # find contours
            other, contours, hierarhy = cv2.findContours(digit_bin.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            #mylogger.debug("Crop digit %d" % i)
            
            # analyze contours
            biggest_contour = None
            biggest_contour_area = 0
            for cnt in contours:
                M = cv2.moments(cnt)

                # centroid
                cx = M['m10']/M['m00'] if M['m00'] else 0
                cy = M['m01']/M['m00'] if M['m00'] else 0
                
                # mylogger.debug("Center: %f,%f" % (cx/dw,cy/dh))
                
                # digit must be in the center
                if cx/dw<0.15 or cx/dw>0.85 or cy/dh<0.25 or cy/dh>0.75:
                    continue
                
                # mylogger.debug("Area: %d, perimeter: %d" % (cv2.contourArea(cnt),cv2.arcLength(cnt,True)))

                # skip very small area contours
                if cv2.contourArea(cnt)<17:
                    continue
                # skip very small perimeter contours
                if cv2.arcLength(cnt,True)<30:
                    continue

                # identify biggest contour
                if cv2.contourArea(cnt)>biggest_contour_area:
                    biggest_contour = cnt
                    biggest_contour_area = cv2.contourArea(cnt)
                    biggest_contour_cx = cx
                    biggest_contour_cy = cy
            
            # if biggest contour not found, mark digit unknown
            if biggest_contour==None:
                digit = self.dbDigit(i, digit_bin)
                digit.markDigitForManualRecognize (use_for_training=False)
                mylogger.warn("Digit %d: no biggest contour found" % i)
                continue    
            
            # filter digit image using biggest contour    
            mask = np.zeros(digit_bin.shape,np.uint8)
            cv2.drawContours(mask,[biggest_contour],0,255,-1)
            digit_bin = cv2.bitwise_and(digit_bin,digit_bin,mask = mask)
            
#             cv2.imshow("digit",digit_bin)
#             k = cv2.waitKey(0)
#             if k==1048603:
#                 sys.exit()        
            
            # caclulate bounding rectangle
            rw = dw/2.0
            rh = dh/1.4
            if biggest_contour_cy-rh/2 < 0:
                biggest_contour_cy = rh/2
            if biggest_contour_cx-rw/2 < 0:
                biggest_contour_cx = rw/2
            
            # crop digit
            digit_bin = digit_bin[int(biggest_contour_cy-rh/2):int(biggest_contour_cy+rh/2), int(biggest_contour_cx-rw/2):int(biggest_contour_cx+rw/2)]
            
            # adjust size to standart
            digit_bin = cv2.resize(digit_bin,(digit_base_w, digit_base_h))
            digit_bin = cv2.threshold(digit_bin, 128, 255, cv2.THRESH_BINARY)[1]
            
            # save to base
            digit = self.dbDigit(i, digit_bin)
            
        return True
        
    def identifyDigits(self):

        # if image already recognized, do nothing
        if self.result!='':
            return True
            
        # if no digits extracted yet, try to do it
        if len(self.digits)==0:
            # We cannot split digits if image not defined
            if self.img == None:            
                return False
            if not self.splitDigits():
                return False
            # store digits objects to base
            sess.commit()
    
        # try to recognize each digit
        for digit in self.digits:
            digit.identifyDigit()
        
        # convert digits to chars
        str_digits = map(str,self.digits)
        
        # if we has undefined digits we can't get result
        if '?' in str_digits:
            return False
        
        # concat digits to result
        self.result = ''.join(str_digits)
        return True
        
    
# digit class
class Digit(Base):
    __tablename__ = 'digits'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    i = Column(Integer)
    body = Column(PickleType)
    result = Column(String(1))
    use_for_training = Column(Boolean)
    
    def __init__(self, image, i, digit_img):
        self.image_id = image.id
        self.i = i
        self.body = digit_img
        self.markDigitForManualRecognize()
        
    def __repr__(self):
        return "%s" % self.result
        
    def markDigitForManualRecognize (self, use_for_training=False):  
        self.result = '?'
        self.use_for_training = use_for_training        
        
    def getEncodedBody (self):
        enc = cv2.imencode('.png',self.body)[1]
        b64 = base64.b64encode(enc)
        return b64        
        
    def identifyDigit (self):
    
        if self.result!='?':
            return True
        
        if not KNN.recognize(self):
            self.markDigitForManualRecognize()
            # if it the last digit, mark it "0"
            if self.i==7:
                self.result = 0
                return True
            return False
        else:
            self.use_for_training = True
        
        return True

#         cv2.imshow("digit",self.body)
#         print self.result
#               
#         k = cv2.waitKey(0)
#         if k==1048603:
#             sys.exit()
    
Base.metadata.create_all(bind=dbengine)
    
# function to get Image object by file_name and img
def getImage(file_name):
    image = sess.query(Image).filter_by(file_name=file_name).first()
    if not image:
        image = Image(file_name)
        sess.add(image)
        # store image object to base
        sess.commit()
    image.digits_img = None
    return image

def getLastRecognizedImage():
    return sess.query(Image).filter(Image.result!='').order_by(Image.check_time.desc()).first()

def dgDigitById(digit_id):
    digit = sess.query(Digit).get(digit_id)
    return digit

class KNN (object):
    _knn = None
    _trained = False
    @staticmethod
    def getKNN():
        if KNN._knn==None:
            KNN._knn = cv2.ml.KNearest_create()
        return KNN._knn
    @staticmethod
    def train():    
        mylogger.info("Start training")
        knn = KNN.getKNN()
        # fetch digits for train from base, use only last 20000
        train_digits = sess.query(Digit).filter(Digit.result!='?').filter_by(use_for_training=True).order_by("id DESC").limit(20000).all()
        train_data = []
        responses = []
        for dbdigit in train_digits:    
            h,w = dbdigit.body.shape
            # skip digits with bad shape
            if h*w != digit_base_h*digit_base_w:
                continue
            # cast digit to 1 dimension array
            sample = dbdigit.body.reshape(digit_base_h*digit_base_w).astype(np.float32)
            train_data.append(sample)
            responses.append(int(dbdigit.result))
        # store training data
        knn.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(responses))
        KNN._trained = True
        mylogger.info("Training complete")
    @staticmethod
    def recognize(dbdigit):
        if not KNN._trained:
            KNN.train()
            
        # check digit resolution    
        h,w = dbdigit.body.shape
        if h!=digit_base_h or w!=digit_base_w:
            dbdigit.markDigitForManualRecognize(use_for_training=False)
            mylogger.warn("Digit %d has bad resolution: %d x %d" % (dbdigit.i,h,w))
            return False
            
        # cast digit to 1 dimension array
        sample = dbdigit.body.reshape(digit_base_h*digit_base_w).astype(np.float32)
        test_data = np.array([sample])
        
        knn = KNN.getKNN()
        ret,result,neighbours,dist = knn.findNearest(test_data,k=5)

        # filter bad results
        if result[0,0]!=neighbours[0,0]:
            mylogger.debug("Digit %d. Result %d doesn't match first neighbour %d" % (dbdigit.i, result[0,0], neighbours[0,0]))
            dbdigit.markDigitForManualRecognize()
            return False
        if neighbours[0,1]!=neighbours[0,0] or neighbours[0,2]!=neighbours[0,0]:
            mylogger.debug("Digit %d. Result %d. Three first neighbours are not the same: %d, %d, %d" % (dbdigit.i, result[0,0], neighbours[0,0], neighbours[0,1], neighbours[0,2]))
            dbdigit.markDigitForManualRecognize()
            return False
        if dist[0,0]>3000000 or dist[0,1]>3500000 or dist[0,2]>4000000:
            mylogger.debug("Digit %d. Result %d. Three first neighbours are not so close: %d, %d, %d" % (dbdigit.i, result[0,0], dist[0,0], dist[0,1], dist[0,2]))
            dbdigit.markDigitForManualRecognize()
            return False
    
        dbdigit.result = str(int(ret))
        return True
    