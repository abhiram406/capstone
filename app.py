import cv2
import easyocr
import streamlit as st
import numpy as np
from ultralytics import YOLO

def application():
    st.header('Automatic Number Plate Detection')
    st.subheader('AIML Capstone Project - Group 3')
    st.write('Instructions: Please upload images of cars in the given link and switch through the tabs to check if the number plate gets detected')

    uploaded_file = st.file_uploader(label="Choose an image file",type=['png', 'jpg', 'jpeg'])
    
    tab1, tab2 = st.tabs(["Original", "Detected"])
    
    if uploaded_file:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) 

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
                
                img = opencv_image[int(y1):int(y2), int(x1):int(x2)]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #img = cv2.medianBlur(img,5)
                
                norm_img = np.zeros((img.shape[0], img.shape[1]))
                img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
                img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

                reader = easyocr.Reader(['en'])

            # Read text from an image
                output = reader.readtext(img)

            # Print the extracted text
                for detection in output:
                    
                    st.subheader('Number Plate: '+ (detection[1]).upper())


if __name__ == "__main__":
    application()
