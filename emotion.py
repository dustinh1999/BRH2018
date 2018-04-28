import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os, random
import tkinter as tk
from PIL import ImageGrab
from PIL import Image, ImageTk
from skimage.measure import structural_similarity as ssim
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

def close_event():
    plt.close()

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err = float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
 
def compare_images(imageA, imageB, hum_emo, meme_emo, title, points):
    # compute the mean squared error and structural similarity
    # index for the images
    picA = facechop(imageA)
    picB =facechop(imageB)
    #m = mse(picA, picB)
    #s = ssim(picA, picB)
    
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("meme emotion: " + str(meme_emo) + "        human emotion:" + str(hum_emo) + "        Points:" + str(points)) 
    # show first image
    #timer = fig.canvas.new_timer(interval = 5000)
    #timer.add_callback(close_event)


    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
 
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
 
    # show the images
    #timer.start()
    plt.show()   

def facechop(image):  
    facedata = "./models/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    minisize = (image.shape[1],image.shape[0])
    miniframe = cv2.resize(image, minisize)

    faces = cascade.detectMultiScale(miniframe)
    face_file_name = ""
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,255))

        sub_face = image[y:y+h, x:x+w]
        face_file_name = "face_" + str(y) + ".jpg"
        cv2.imwrite(face_file_name, sub_face)
    return cv2.resize(cv2.imread(face_file_name, 0), (180,180))

def game(points):
    USE_WEBCAM = True # If false, loads video file source

    # parameters for loading data and images
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    # starting video streaming

    cv2.namedWindow('Meme_Matcher')
    video_capture = cv2.VideoCapture(0)

    # Select video or webcam feed
    cap = None
    if (USE_WEBCAM == True):
        cap = cv2.VideoCapture(0) # Webcam source
    else:
        cap = cv2.VideoCapture('./demo/b08.jpg') # Video file source

    folder=r"C:\Users\Matthew\Downloads\Emotion-master\Pictures"
    pick=random.choice(os.listdir(folder))
    #print(a)
    #meme = cv2.imread(folder+'\\'+a)
    #print(pick)
    meme = cv2.imread(folder+'\\'+str(pick))
    meme =  cv2.cvtColor(meme, cv2.COLOR_BGR2GRAY)

    memes = face_cascade.detectMultiScale(meme, scaleFactor=1.1, minNeighbors=7,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    meme_emotion = ""
    meme_prob = 0
    #image = Image.open(folder+'\\'+str(pick))
    #image.show()

    window = tk.Tk()
    window.configure(background = 'white')
    img = ImageTk.PhotoImage(Image.open(folder+'\\'+str(pick)))
    panel = tk.Label(window, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    window.after(3000, lambda: window.destroy()) # Destroy the widget after 30 seconds
    window.mainloop()
    for face_coordinates in memes:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = meme[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        #print("meme prob" + str(emotion_probability))
        meme_prob = emotion_probability
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        meme_emotion = emotion_text
        #print(emotion_text)

    start_time = time.time()

    while cap.isOpened(): # True:
        ret, bgr_image = cap.read()

        #bgr_image = video_capture.read()[1]

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=7,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            #print("human prob" + str(emotion_probability))
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            #if emotion_text == 'angry':
            #    color = emotion_probability * np.asarray((255, 0, 0))
            #if emotion_text == 'sad':
            #    color = emotion_probability * np.asarray((0, 0, 255))
            #elif emotion_text == 'happy':
            #    color = emotion_probability * np.asarray((255, 255, 0))
            #elif emotion_text == 'surprise':
            #    color = emotion_probability * np.asarray((0, 255, 255))
            #else:
            #    color = emotion_probability * np.asarray((0, 255, 0))

            if emotion_text == meme_emotion and meme_emotion == 'angry' and abs(emotion_probability - meme_prob) < 0.08 and time.time()-start_time < 10:
                cv2.imwrite("Haha.jpg", rgb_image)
                img = cv2.imread(folder+'\\'+ str(pick))
                img2 = cv2.imread("Haha.jpg")
                points = int(points + (10-(time.time()-start_time)))
                compare_images(img, img2, emotion_probability, meme_prob, "Blah blah", points)
                cap.release()
                cv2.destroyAllWindows()
            elif emotion_text == meme_emotion and meme_emotion == 'sad' and abs(emotion_probability - meme_prob) < 0.08 and time.time()-start_time < 10:
                cv2.imwrite("Haha.jpg", rgb_image)
                img = cv2.imread(folder+'\\'+ str(pick))
                img2 = cv2.imread("Haha.jpg")
                points = int(points + (10-(time.time()-start_time)))
                compare_images(img, img2, emotion_probability, meme_prob, "Blah blah", points)
                cap.release()
                cv2.destroyAllWindows()
            elif emotion_text == meme_emotion and abs(emotion_probability - meme_prob) < 0.02 and time.time()-start_time < 10:
                cv2.imwrite("Haha.jpg", rgb_image)
                img = cv2.imread(folder+'\\'+ str(pick))
                img2 = cv2.imread("Haha.jpg")
                points = int(points + (10-(time.time()-start_time)))
                compare_images(img, img2, emotion_probability, meme_prob, "Blah blah", points)
                cap.release()
                cv2.destroyAllWindows()
            elif time.time()-start_time > 10:
                cap.release()
                cv2.destroyAllWindows()


            #color = color.astype(int)
            #color = color.tolist()

            #draw_bounding_box(face_coordinates, rgb_image, color)
            #draw_text(face_coordinates, rgb_image, emotion_mode,
            #          color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

points = 0

game(points)

game(points)

game(points)

game(points)