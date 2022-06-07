import cv2
import numpy as np
# import matplotlib.pyplot as plt

yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", 'r') as f:
    classes = f.read().splitlines()


class VideoDetection:
    def run(self):
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('result.avi', fourcc, 6.0, size)

        while True:
            _, img = cap.read()
            H, W, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1/255, (480,480),(0,0,0), swapRB=True, crop=False)

            i = blob[0].reshape(480,480,3)

            yolo.setInput(blob)
            outputLayer_names = yolo.getUnconnectedOutLayersNames()
            layerOutput = yolo.forward(outputLayer_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutput:
                for detection in output:
                    #In 85 values we just need first 5 values (x,y,center,width,height).
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        center_x, center_y, width, height = box.astype("int")
                        # center_x = int(detection[0]*W)
                        # center_y = int(detection[0]*H)
                        
                        # w = int(detection[0]*W)
                        # h = int(detection[0]*H)

                        x = int(center_x - (width / 2))
                        y = int(center_y - (height / 2))
                        # x = int(center_x - w/2)
                        # y = int(center_y - h/2)

                        boxes.append([x,y,int(width),int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))

            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confi = str(round(confidences[i], 2))
                color = colors[i]

                cv2.rectangle(img, (x,y), (x+w, y+h), color, 4)
                cv2.putText(img, label +" "+confi, (x,y+20), font, 1, (255,255,0), 3)


            cv2.imshow('Image', img)

            
            key = cv2.waitKey(33)
            if key == ord('w'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


class ImageDetection:
    def getFile(self, image, save=False):
    
        inputImage = cv2.imread(image)
        H = inputImage.shape[0]
        W = inputImage.shape[1]
        blob = cv2.dnn.blobFromImage(inputImage, 1/255, (480,480),(0,0,0), swapRB=True, crop=False)

        i = blob[0].reshape(480,480,3)

        yolo.setInput(blob)
        outputLayer_names = yolo.getUnconnectedOutLayersNames()
        layerOutput = yolo.forward(outputLayer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutput:
            for detection in output:
                #In 85 values we just need first 5 values (x,y,center,width,height).
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    center_x, center_y, width, height = box.astype("int")
                    # center_x = int(detection[0]*W)
                    # center_y = int(detection[0]*H)
                    
                    # w = int(detection[0]*W)
                    # h = int(detection[0]*H)

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    # x = int(center_x - w/2)
                    # y = int(center_y - h/2)

                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(inputImage, (x,y), (x+w, y+h), color, 4)
            cv2.putText(inputImage, label +" "+confi, (x,y+20), font, 2, (255,255,0), 5)

        if( save == True):
            fileName = input("Enter the file name with extension to save : ")
            filePath = f"Final_Images/{fileName}"
            cv2.imshow('Image', inputImage)
            cv2.imwrite(filePath, inputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow('Image', inputImage)
            cv2.waitKe(0)
            cv2.destroyAllWindows()

    
print("1.Image Detection \n2.Video Detection")
choice = int(input("Enter your choice: "))
print(choice)
if(choice == 2):
    vidDetect = VideoDetection()
    vidDetect.run()
    
else:
    imgDetect = ImageDetection()
    fileNameToOpen = input("Enter the file name with extension to open: ")
    imgDetect.getFile(f'Images/{fileNameToOpen}', True)