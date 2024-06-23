import cv2
import PIL
import streamlit as st
import numpy as np
from ultralytics import YOLO

def app():
    st.header('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome!')

    uploaded_file = st.file_uploader(label="Choose an image file",
                 type=['png', 'jpg', 'jpeg'])
    
    #image = PIL.Image.open(uploaded_file)

    
    


    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        
    
        model = YOLO('best.pt')

        result = model.predict(opencv_image)
        boxes = result[0].boxes.xyxy.tolist()
        for r in result:
            
            #x1, y1, x2, y2 = boxes
            #res_plotted = r[0].plot()
            #cropped_image = opencv_image[int(y1):int(y2), int(x1):int(x2)]
            res_plotted = r.plot()[:, :, ::-1]
            st.write(boxes)
            #st.image(cropped_image,caption="Cropped Image",use_column_width=True)



if __name__ == "__main__":
    app()
