import httplib2
import os

from datetime import tzinfo, timedelta, date
from dateutil.relativedelta import relativedelta

import cv2
import numpy as np
import ConfigParser

# Gdrive
from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow
from oauth2client.file import Storage

from models import getLastRecognizedImage, mylogger

class TZ(tzinfo):
    def utcoffset(self, dt): return timedelta(hours=+4)

def getAuthorizedHttp():
    
    config = ConfigParser.ConfigParser()
    config.read([os.path.dirname(__file__)+'/config.ini'])
    
    # Copy your app credentials from the console
    CLIENT_ID = config.get('gdrive','CLIENT_ID')
    CLIENT_SECRET = config.get('gdrive','CLIENT_SECRET')

    # OAuth 2.0 scope that will be authorized.
    # Check https://developers.google.com/drive/scopes for all available scopes.
    OAUTH_SCOPE = 'https://www.googleapis.com/auth/drive'
    
    # Redirect URI for installed apps
    REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'
    
    # Init auth storage
    storage = Storage(os.path.dirname(__file__) + '/client_secrets.json')
    credentials = storage.get()
    
    # Check credentials existance
    if not credentials:
        # Perform OAuth2.0 authorization flow.
        flow = OAuth2WebServerFlow(CLIENT_ID, CLIENT_SECRET, OAUTH_SCOPE, REDIRECT_URI)
        authorize_url = flow.step1_get_authorize_url()
        print 'Go to the following link in your browser: ' + authorize_url
        code = raw_input('Enter verification code: ').strip()
        credentials = flow.step2_exchange(code)
        # Store allowed credentials
        storage.put(credentials)
    
    # Create an authorized Drive API client.
    http = httplib2.Http()
    credentials.authorize(http)
    return http
    

def getImagesFromGDrive():
    
    # Photo folder id
    FOLDER_ID = '0B5mI3ROgk0mJcHJKTm95Ri1mbVU'
    
    http = getAuthorizedHttp()
    
    drive_service = build('drive', 'v2', http=http)
    
    # move old images to trash
    mylogger.debug("Delete old files on google drive")
    month_ago = date.today() + relativedelta( months = -1 )
    files = drive_service.files().list(q = "'%s' in parents and mimeType = 'image/jpeg' and trashed = false and modifiedDate<'%s'" % (FOLDER_ID, month_ago.isoformat()), 
                                       maxResults=1000).execute()
    for image in files.get('items'): 
        mylogger.debug("Delete %s" % image['title'])
        drive_service.files().trash(fileId=image['id']).execute()
    mylogger.debug("Deleting complete")
    
    # define last recognized image time
    last_image = getLastRecognizedImage()    
     
    # Fetch photos from folder
    page_size = 1000
    result = []
    pt = None
    while True:
        q = "'%s' in parents and trashed = false and mimeType = 'image/jpeg' and modifiedDate>'%s'" % (FOLDER_ID, last_image.check_time.replace(tzinfo=TZ()).isoformat('T'))
        files = drive_service.files().list(q = q, maxResults=page_size, pageToken=pt).execute()
        result.extend(files.get('items'))
        pt = files.get('nextPageToken')
        if not pt:
            break
    
    result.reverse()
        
    return result, http

def downloadImageFromGDrive (downloadUrl, http=None):
    if http==None:
        http = getAuthorizedHttp()
    # Download photo
    resp, content = http.request(downloadUrl)
    # Create cv image
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def createImageFromGDriveObject (img_info, http=None):
    return downloadImageFromGDrive(img_info['downloadUrl'], http)