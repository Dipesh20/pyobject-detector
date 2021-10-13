import numpy as np
import time
import cv2
import math
import pandas as pd


def getYoloOutput(imagePath, extension):
    labelsPath = './yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = 'yolo-coco/yolov3.weights'
    configPath = 'yolo-coco/yolov3.cfg'
    conf_threshold = 0.5
    nms_threshold = 0.4
    # initialize a list of colors to represent each possible class label
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    image = cv2.imread(imagePath)
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf_threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                            nms_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # file = open("./static/outputFile.txt","a")
        test = {}
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            test[LABELS[classIDs[i]]] = test.get(LABELS[classIDs[i]], 0) + 1
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        lst = list(test.keys())
        lst2 = list(test.values())

        df = pd.DataFrame(list(zip(lst, lst2)), columns=['Category', 'Count'])
        outputFile = "output" + str(math.trunc(time.time()))+".csv"
        outputFilePath = './static/'+outputFile
        df.to_csv(outputFilePath, index=False)
    else:
        outputFile = "output" + str(math.trunc(time.time()))+".txt"
        outputFilePath = './static/'+outputFile
        file2 = open(outputFilePath, "w+")
        file2.write("No Category Detected\n")
        file2.close()

    filename = "savedImage" + str(math.trunc(time.time()))+"." + extension
    filePath = "./static/"+filename
    cv2.imwrite(filePath, image)
    return filePath, outputFilePath
