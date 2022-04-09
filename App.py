from PIL import Image
import streamlit as st
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL

#Loading the Inception model
model= load_model('frames.h5',compile=(False))

#Functions
def splitting(name):
    vidcap = cv2.VideoCapture(name)
    success,frame = vidcap.read()
    count = 0
    frame_skip =1
    while success:
        success, frame = vidcap.read() # get next frame from video
        cv2.imwrite(r"img\frame%d.jpg" % count, frame) 
        if count % frame_skip == 0: # only analyze every n=300 frames
            #print('frame: {}'.format(count)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
            #st.image(pil_img)
        if count > 20 :
            break
        count += 1
    preprocessing()

def preprocessing():
    x = tf.io.read_file('img/frame2.jpg')
    x = tf.io.decode_image(x,channels=3) 
    x = tf.image.resize(x,[299,299])
    x = tf.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x
    
def predict(x):
    P = tf.keras.applications.inception_v3.decode_predictions(model.predict(x), top=1)
    return P


def main():
    st.title("Computer Vision.")

    selected = st.text_input("Search for an Object here....",)
    file = st.file_uploader("Upload video",type=(['mp4']))
    if file is not None: # run only when user uploads video
        vid = file.name
        with open(vid, mode='wb') as f:
            f.write(file.read()) # save video to disk

        st.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        cur_frame = 0
        success = True
    
    def generatesearchitems():
        for i in range(20):
            filename = ("img\frame%d.jpg" + str(i))
            x = tf.io.read_file(filename)
            x = tf.io.decode_image(x,channels=3) 
            x = tf.image.resize(x,[299,299])
            x = tf.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            P = tf.keras.applications.inception_v3.decode_predictions(model.predict(x), top=1)
            if (P[0][0][1]) == selected :
                st.success("Items Found")
                pic =  Image.open(filename)
                st.image(pic)
                return 0
        st.warning("Item not  Found")
        
    if st.button("Detect"):
        output1 = splitting(vid)
        output2 = preprocessing()
        output = predict(output2)
        #st.success('The Output is {}'.format(output))
        st.success("Successfuly detected all the objects now you can search for the item!")
        items =  generatesearchitems()
if __name__=='__main__':
    main()
