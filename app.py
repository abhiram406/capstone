import cv2
import easyocr
import streamlit as st
import numpy as np
from ultralytics import YOLO

def application():
    st.header('Automatic Number Plate Detection')
    st.subheader('AIML Capstone Project - Group 3')
    #st.subheader('Powered by YOLOv8')
    st.write('Welcome!')

    with st.sidebar:
        
        uploaded_file = st.file_uploader(label="Choose an image file",
                 type=['png', 'jpg', 'jpeg'])
    
    tab1, tab2, tab3 = st.tabs(["Original", "Detected", "Number Plate"])
    
    if uploaded_file:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) 

        with tab1:
            st.subheader("Uploaded Image")
            st.image(opencv_image,channels = "RGB")
        
        model = YOLO('best.pt')

        result = model.predict(opencv_image)
        st.write(result)
        boxes = result[0].boxes.xyxy.tolist()

        
        for r in result:
            
            with tab2:
                st.subheader("Original Image with Detected Number Plates")
                st.image(r.plot())
            
            x1, y1, x2, y2 = boxes[0]
            #res_plotted = r[0].plot()
            
            img = opencv_image[int(y1):int(y2), int(x1):int(x2)]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.medianBlur(img,5)
            
            norm_img = np.zeros((img.shape[0], img.shape[1]))
            img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            
            #res_plotted = r.plot()[:, :, ::-1]
            with tab3:
                st.image(img,caption="Cropped Image",use_column_width=True)

                reader = easyocr.Reader(['en'])

            # Read text from an image
                output = reader.readtext(img)

            # Print the extracted text
                for detection in output:
                    st.write(detection[1])


if __name__ == "__main__":
    application()
