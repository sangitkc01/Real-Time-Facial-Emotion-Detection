import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
# import tensorflow as tf
# from tensorflow import keras
# import keras
from keras.models import Model, model_from_json
import time
from bokeh.models.widgets import Div
from keras_preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase



# class
class FaEmoModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model.compile()
        # self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        # set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FaEmoModel.EMOTIONS_LIST[np.argmax(self.preds)]


# importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FaEmoModel("model.json", "model.h5")
# model = model_from_json(open("model.json", "r").read())
# model.load_weights('best_accuracy_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


# facial expressions detecting function
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        fc = gray[y:y + h, x:x + w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img, faces, pred

    # the main function



def main():
    """Face Expression Detection App"""
    # setting the app title & sidebar

    activities = ["Home", "Recognize My Facial Expressions", "See Model Performance", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    st.sidebar.markdown(
        """ **Developed by Sangit KC, Sohel Mansuri, Narayan Pokherel**    
            Email : kc.sangeet680@gmail.com  
            """)
    st.sidebar.markdown("**GitHub:** https://github.com/sangitkc01")
    st.sidebar.markdown("**LinkedIn:** https://www.linkedin.com/in/sangeet-kc-6b26811b5")

    if choice == 'Home':
        html_temp = """ 
        <h2 style= "color: 	#330033  ; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">How are you feeling today? </h2>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                                    <h4 style="color:white;text-align:center;">
                                                    Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                                    </div>
                                                    </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
        st.markdown("____")
        Text ="""
        <h3 style="font-family: 'Times New Roman'; color: black; font-size: 30px;font-weight: 700">
        There are seven emotional categories they are: <br></br>
        1. Angry <br></br>
        2. Disgust <br></br>
        3. Fear <br></br>
        4. Happy <br></br>
        5. Sad <br></br>
        6. Surprise <br></br>
        7. Neutral <br></br>
        </h3>
        """
        st.markdown(Text, unsafe_allow_html=True)


    if choice == 'Recognize My Facial Expressions':
        st.title("Web Application For Facial Expression Recognition!")
        # html_choice = """
        # <marquee behavior="scroll" direction="left" width="100%;">
        # <h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Facial Expression Recognition Web Application </h2>
        # </marquee><br>
        # """
        # st.markdown(html_choice, unsafe_allow_html=True)

        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
        st.markdown("____")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        # if image is uploaded, display the progress bar and the image
        if image_file is not None:
            our_image = Image.open(image_file)
            st.markdown("**Original Image**")

            # Progress bar
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
            # End of progress bar

            st.image(our_image)

        if image_file is None:
            st.error("No image uploaded yet")
            return

        # Face Detection
        task = ["Faces"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):
            if feature_choice == 'Faces':
                st.markdown("**Processing...\n**")

                # Progress bar
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i + 1)
                # End of Progress bar

                result_img, result_faces, prediction = detect_faces(our_image)
                if st.image(result_img):
                    st.success("Found {} faces".format(len(result_faces)))

                    if prediction == 'Happy':
                        st.subheader("YeeY! You look **_Happy_** :smile: today, always be! ")
                    elif prediction == 'Angry':
                        st.subheader("You seem to be **_Angry_** :rage: today, just take it easy! ")
                    elif prediction == 'Disgust':
                        st.subheader("You seem to be **_Disgusted_** :rage: today! ")
                    elif prediction == 'Fear':
                        st.subheader("You seem to be **_Fearful_** :fearful: today, be couragous! ")
                    elif prediction == 'Neutral':
                        st.subheader("You seem to be **_Neutral_** today, wish you a happy day! ")
                    elif prediction == 'Sad':
                        st.subheader("You seem to be **_Sad_** :worried: today, smile and be happy! ")
                    elif prediction == 'Surprise':
                        st.subheader("You seem to be **_Surprised_** today! ")
                    else:
                        st.error(
                            "Your image does not seem to match the training dataset's images! Please try another image!")

    # if choice == "Webcam Face Detection":
    #     st.header("Webcam Live Feed")
    #     st.write("Click on start to use webcam and detect your face emotion")
    #     webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


    if choice == 'See Model Performance':
        st.title("Web Application For Facial Expression Recognition!")
        # html_choice = """
        # <marquee behavior="scroll" direction="left" width="100%;">
        # <h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Facial Expression Recognition Web Application </h2>
        # </marquee><br>
        # """
        # st.markdown(html_choice, unsafe_allow_html=True)

        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
        st.markdown("____")
        # st.subheader("\nWorking on bringing it to you...")
        # st.subheader("Will be back with it soon!")
        actions = ["Train and Test Accuracies", "Evaluation of the Model", "Confusion Matrix"]
        action = st.sidebar.selectbox("Choose One", actions)
        # st.markdown("____")
        # if st.checkbox('Train and Test Accuracies'):
        if action == 'Train and Test Accuracies':
            st.subheader('**Train Accuracy vs. Test Accuracy**')
            st.image('model_accuracy.png', width=700)
        # st.markdown("____")
        # if st.checkbox('\n\nEvaluation of the Model'):
        if action == 'Evaluation of the Model':
            st.subheader('**Evaluation of the Model**')
            st.image('model_metrics.JPG', width=700)
        # st.markdown("____")
        # if st.checkbox('\n\nConfusion Matrix'):
        if action == 'Confusion Matrix':
            st.subheader('**Confusion Matrix using all the classes**')
            st.image('model_cm.png', width=700)


    elif choice == 'About':
        st.title("Web Application For Facial Expression Recognition!")
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
        st.markdown("____")
        html_temp4 = """
                                     		<div style="background-color:#98AFC7;padding:10px">
                                     		<h4 style="color:white;text-align:center;">This Application is developed by Sangit KC using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at kc.sangeet680@gmail.com. </h4>
                                     		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                     		</div>
                                     		<br></br>
                                            <br></br>"""
        st.markdown(html_temp4, unsafe_allow_html=True)

        st.markdown(
            "**Dataset used for training:** https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge")



main()