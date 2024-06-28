import cv2
from cv2 import dnn_superres
import easyocr
import streamlit as st
import numpy as np
from ultralytics import YOLO

def application():
    st.header('Automatic Number Plate Detection')
    st.subheader('AIML Capstone Project - Group 3')
    st.write('Instructions: Please upload images of cars in the given link and switch through the tabs to check if the number plate gets detected')

    uploaded_file = st.file_uploader(label="Choose an image file",type=['png', 'jpg', 'jpeg'])
    
    tab1, tab2, tab3 = st.tabs(["Original", "Detected", "Number Plate"])
    
    if uploaded_file:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #opencv_image = cv2.filter2D(opencv_image, -1, kernel)

        #sr = dnn_superres.DnnSuperResImpl.create()
        #path = 'EDSR_x4.pb'
        #sr.readModel(path)
        #sr.setModel('edsr', 4)
        
        with tab1:
            st.subheader("Original Image")
            st.image(opencv_image,use_column_width=True)
        
        model = YOLO('best.pt')

        result = model.predict(opencv_image)
        boxes = result[0].boxes.xyxy.tolist()

        
        for r in result:
            
            with tab2:
                st.subheader("Original Image with Detected Number Plates")
                st.image(r.plot(),use_column_width=True)
            
                x1, y1, x2, y2 = boxes[0]
                #res_plotted = r[0].plot()
                
                numplate_img = opencv_image[int(y1):int(y2), int(x1):int(x2)]
                numplate_img = cv2.cvtColor(numplate_img, cv2.COLOR_BGR2GRAY)
                #numplate_img = cv2.medianBlur(numplate_img,3)
                
                norm_img = np.zeros((numplate_img.shape[0], numplate_img.shape[1]))
                #numplate_img = cv2.normalize(numplate_img, norm_img, 0, 255, cv2.NORM_MINMAX)
                #numplate_img = cv2.threshold(numplate_img, 0, 255, cv2.THRESH_BINARY)[1]
              
                #numplate_img = sr.upsample(numplate_img)

                reader = easyocr.Reader(['en'])

            
            # Read text from an image
                #output = reader.readtext(sharpened_image)
                output = reader.readtext(numplate_img)

            # Print the extracted text
                for detection in output:
                    
                    st.subheader('Number Plate: '+ (detection[1]).upper())
                    
            with tab3:
                st.subheader("Number plate upscaled")
                st.image(numplate_img,use_column_width=True)
                
                

if __name__ == "__main__":
    application()
