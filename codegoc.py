import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model
import tensorflow as tf

#Load du lieu tu file 'Model.h5'
model = load_model('keras_model.h5')
classes = { 
    1:'con mèo',
    0:'con chó',
}

#Tao giao dien GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Phân loại')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

#Phan loai
def classify(file_path):
    global label_packed
    image = Image.open(file_path)  
    image = image.resize((224,224))
    image = numpy.expand_dims(image, axis=0)  
    image = numpy.array(image)
    image = image/255
    pred = numpy.argmax(model.predict(image),axis=1)[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign) 

#Goi va hien thi nut phan loai
def show_classify_button(file_path):
    classify_b=Button(top,text="Phân loại",
   command=lambda: classify(file_path),
   padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

#Upload anh
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)  
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))  
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Tải ảnh lên",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='black',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Phân loại chó mèo",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='black')
heading.pack()
top.mainloop()
