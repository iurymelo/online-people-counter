#Flask related imports
from flask import Flask, request, render_template, redirect, flash, url_for, Response
from werkzeug.utils import secure_filename


import cv2 as cv
import numpy as np

import sys
import os.path
import os

import utils.centroid_tracker as centroid
import utils.get_frame as get_frame


app = Flask(__name__)
app.secret_key = b'_51123*1823*&!*y2L"F4Q8z\n\xec]/'

ENV = 'dev'

if ENV == 'dev':
    app.debug = True

else:
    app.debug = False

# Feed Link
feed = 'http://176.57.73.231/mjpg/video.mjpg'
timeToWait = 3000

# File with the prediction boxes
outputFile = 'static/boxes.jpg'

# Initialize the Centroid Tracker
ct = centroid.CentroidTracker()

# Detector Params
confThreshold = 0.5  # Confidence Threshold
nmsThreshold = 0.4   # Non max suppression threshold
inpWidth = 416       # Image Width to feed the network
inpHeight = 416      # Image Height to feed the network

# Load Classes
classesFile = "dnn_config_files/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load model and configuration file
modelConfiguration = "dnn_config_files/yolov3.cfg"
modelWeights = "dnn_config_files/yolov3.weights"   
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Configurations for image upload
DEV_PATH = '/app/static/images/uploads'
app.config["IMAGE_UPLOADS"] = 'static/images/uploads'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPG", "JPEG"]
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SEND_FILE_MAX_FILESIZE'] = 2.5 * 1024 * 1024



def getOutputsNames(net):
    """Get the output names from the output layer

    Arguments:
        net {network} -- Yolo network

    Returns:
        list -- List of names
    """
    layersNames = net.getLayerNames()

    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame, classId, conf, left, top, right, bottom):
    """Draw a rectangle on around a found person.

    Arguments:
        classId {list} -- List of class id's.
        conf {list} -- List of configurations.
        left {integer} -- Left point of the box.
        top {integer} -- top point of the box.
        right {integer} -- rigth point of the box.
        bottom {integer} -- bottom point of the box.
    """
    if classId == 0:
        # Draw box border
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
            

        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
    
    
def postprocess(frame, outs):
    """Remove predictions with low confidence using non-maximum suppression

    Arguments:
        frame {frame} -- Video Frame
        outs {output} -- Output from a network

    Returns:
        outupt -- Output with boxes removed 
    """

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-maximum suppression to eliminate redudances and overlapping in
    # boxes with low confidence. 
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    cont = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        cont+=1
    return boxes


def count_boxes(boxes):
    """Count the number of boxes (predictions)

    Arguments:
        boxes {list} -- List of boxes

    Returns:
        integer -- Number of boxes
    """
    
    count = 0
    for i in boxes:
        count +=1
    return count


def confidence(outs):
    """Calculate the result's confidence using the median of all boxes' confidences.

    Arguments:
        outs {output} -- Result of a network.

    Returns:
        float -- Confidence between 0 and 1. 
    """
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                confidences.append(float(confidence))
    if len(confidences) > 0:
        res = np.median(confidences)
        
        return round(res, 2)
    else:
        return 1


def do_predictions(live=True):
    """Predict the number of people in a video feed.
    Arguments: 
        live {boolean}: True = Capturing from a live feed. False: Capture from uploaded file.
        imgPath {string}: Path to the image

    Returns:
        list -- Success: Boolean, Number of detected objects: int, confidence: str, and error message: str 
    """
    try:
        
        if live:
            # Get the last frame saved
            get_frame.getFrames(feed)
            cap = cv.VideoCapture("static/images/feed/4.jpg")
        else:
            cap = cv.VideoCapture("static/images/uploads/input")

        hasFrame, frame = cap.read()

        if not hasFrame:
            return [False, 0, 0, 'No Frame Found']
        
        # Cria um blob 4D do frame
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        # Seta a entrada da rede neural
        net.setInput(blob)
        
        # Processa a rede
        outs = net.forward(getOutputsNames(net))

        # Retira detecções de baixa confiabilidade
        caixas = postprocess(frame, outs)
        
        # Conta
        numObj = count_boxes(caixas)
        
        #Atualiza centroids
        objetos = ct.update(caixas)
        
        # Adiciona informação de eficiencia. A funcao getPerfProfile retorna o 
        #tempo geral por inferencia(t) e os tempos para cada camada.
        t, _ = net.getPerfProfile()
        
        
        # Conta a quantidade de objetos encontrados
        # numObj = count_boxes(outs)
        
        # Retorna a média da confiança de todas as predições
        conf = confidence(outs)

        cv.imwrite(outputFile, frame.astype(np.uint8));

        cap.release()

        return_obj = [True, numObj, conf, None]

        return return_obj
    except:
        try:
            if cap:
                cap.release()
        except: 
            return [False, 0, 0, 'Detection Failed and exited.']
        return [False, 0, 0, 'Detection Failed and exited.']


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to remove page cache
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/livefeed')
def live_feed():
    suc, numObj, conf, errorMsg = do_predictions()
    return render_template('livefeed.html', numObj=numObj, conf=conf)    

def allowed_image(filename):
    """Verify if the file type is allowed

    Arguments:
        filename {string} -- File Name

    Returns:
        {boolean} -- True if file extesions is allowed. False otherwise. 
    """
    if not "." in filename:
        return False
    
    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def allowed_size(filesize):
    """Verify if the file size is valid.

    Arguments:
        filesize {int} -- Size of the file

    Returns:
        {boolean} -- True if the file size is valid. False otherwise. 
    """
    if int(filesize) <= app.config['SEND_FILE_MAX_FILESIZE']:
        return True
    else:
        return False

@app.route('/upload', methods=["GET", "POST"])
def upload_image():
    try: 
        if request.method == 'POST':
            if request.files:
                if not allowed_size(request.cookies.get("filesize")):
                    flash("File exceeded maximum size.")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    flash("Image must have a name!")
                    return redirect(request.url)
                
                if not allowed_image(image.filename):
                    flash("Only JPG, JPEG allowed!")
                    return redirect(request.url)

                else:
                    filename = secure_filename(image.filename)
                    ext = filename.rsplit(".", 1)[1]
                    fileName = 'input'
                    if os.path.exists(fileName):
                        os.remove(fileName)
                    image.save(os.path.join(app.config["IMAGE_UPLOADS"], fileName))
                    suc, numObj, conf, errorMsg = do_predictions(live=False)

                    return render_template('upload.html', uploaded=True, numObj=numObj, conf=round(conf*100,1), suc=suc, errorMsg=errorMsg)
        return render_template('upload.html')
    except Exception as e:
        return render_template('upload.html', suc=suc, errorMsg=e)
    

if __name__ == '__main__':
    app.run()