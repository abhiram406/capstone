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
        for r in result:
            res_plotted = r[0].plot()
            #res_plotted = r.plot()[:, :, ::-1]
            #st.write(r.boxes.xyxy.tolist())
            st.image(res_plotted,
                    caption="Uploaded Image",
                    use_column_width=True
                    )



if __name__ == "__main__":
    app()
