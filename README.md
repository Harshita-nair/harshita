# harshita
project repository
from tkinter import *
from tkinter.filedialog import askopenfilename
import shutil
import random
import os
import sys
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')

window =Tk()

window.title("BUS EMPTY SEAT DETECTION")

window.geometry("1550x800")
window.configure(background ="cyan")
img=Image.open("bag.jpg")
img=img.resize((1550,800))
bg=ImageTk.PhotoImage(img)

lbl=Label(window,image=bg)
lbl.place(x=0,y=0)

title = Label(text="Click below to choose picture for \n Check Bus status ....", background = "Brown", fg="white", font=("elephant", 28, "bold" ))
title.place(x=450,y=20)
def exitwin():
    window.destroy()
def analysis():
    button2.destroy()
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'bus-new-{}-{}.model'.format(LR, '2conv-basic')
##    button2.destroy()
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')


    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 5, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        pred= random.randint(90,98)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))


        #if model_out > 0.5

        if np.argmax(model_out) == 4:
            print("The predicted image of the bus status is full with a accuracy of {} %".format(model_out[4]))
            str_label = 'FULL'
            col="#f25565"
            res='STATUS OF THE BUS : FULL  \n \n WAIT FOR THE NEXT BUS,'
        elif np.argmax(model_out) == 0:
            str_label = 'EMPTY'
            col="green"
            print("The predicted image of the bus status is empty with a accuracy of {} %".format(model_out[0]))
            res=' BUS STATUS IS: EMPTY, \n \n YOU CAN BOARD INTO THE BUS'
        elif np.argmax(model_out) == 1:
            str_label = '5_seats'
            col="green"
            print("The predicted image of the bus status is empty with a accuracy of {} %".format(model_out[1]))
            res=' BUS STATUS IS: Around 5 seats are Empty, \n \n YOU CAN BOARD INTO THE BUS'
        elif np.argmax(model_out) == 2:
            str_label = '10_seats'
            col="green"
            print("The predicted image of the bus status is empty with a accuracy of {} %".format(model_out[2]))
            res=' BUS STATUS IS: Around 10 seats are Empty, \n \n YOU CAN BOARD INTO THE BUS'
        elif np.argmax(model_out) == 3:
            str_label = '15_seats'
            col="green"
            print("The predicted image of the bus status is empty with a accuracy of {} %".format(model_out[3]))
            res=' BUS STATUS IS: Around 15 seats are Empty, \n \n YOU CAN BOARD INTO THE BUS'
        res_lbl=Label(window,text=res,bg=col,fg="white",font=("times",20,"bold"),width=40).place(x=450,y=520)

        button3 = Button(text="Recheck", command=openphoto,width=10,bg="brown",fg="white",font=("times",20,"bold"))
        button3.place(x=530,y=650)
        
        button = Button(text="Exit", command=exitwin,width=10,bg="brown",fg="white",font=("times",20,"bold"))
        button.place(x=800,y=650)

def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='BUSSSSS\\test\\', title='Select image for analysis ',
                           filetypes=[('image files', '.jpeg')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    load=load.resize((400,400))
    render = ImageTk.PhotoImage(load)
    img =Label(image=render, height="400", width="400")
    img.image = render
    img.place(x=550, y=100)
##    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    global button2
    button2 = Button(text="CHECK STATUS", command=analysis,width=30,bg="brown",fg="white",font=("times",20,"bold"))
    button2.place(x=500,y=650)
button1 = Button(text="Get Photo", command = openphoto,width=10,bg="brown",fg="white",font=("times",20,"bold"))
button1.place(x=650,y=650)




window.mainloop()



