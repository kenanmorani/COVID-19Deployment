# -*- coding: utf-8 -*-
"""covid_prediction_idu-cvlab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ltmPYJ5K88BxhNTcQ2aAH5VcPV3Rpts4
"""

#from google.colab import drive
#drive.mount('/content/drive/')

#model=tf.keras.models.load_model('https://drive.google.com/file/d/1fRfNE7LRKkRz0LN3WN6JuB9zkKr8wG34/view?usp=sharing')

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

#!pip install -q streamlit

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# 
# @st.cache(allow_output_mutation=True)
# def load_model():
#   model=tf.keras.models.load_model('/content/drive/MyDrive/IDU-CV Lab Work/COV19D_2nd - Trnasfer Learning/Saved Models/Modified_Xception.h5')
#   return model
# with st.spinner('Model is being loaded..'):
#   model=load_model()
# 
# st.write("""
#          # Image Classification
#          """
#          )
# 
# file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg", "png"])
# import cv2
# from PIL import Image, ImageOps
# import numpy as np
# st.set_option('deprecation.showfileUploaderEncoding', False)
# 
# def upload_predict(upload_image, model):
#     
#         size = (224,224)    
#         image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
#         image = np.asarray(image)
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
#         
#         img_reshape = img_resize[np.newaxis,...]
#     
#         prediction = model.predict(img_reshape)
#         if prediction > 0.5:
#          pred_class='NON-COVID'
#         else:
#          pred_class='COVID'
#         #pred_class=decode_predictions(prediction,top=1)
#         
#         return pred_class
# 
# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     predictions = upload_predict(image, model)
#     image_class = predictions
#     #score=np.round(predictions[0][0][2],5) 
#     st.write("The image is classified as", predictions)
#     #st.write("The similarity score is approximately",score)
#     print("The image is classified as ",image_class,)

!streamlit run app.py & npx localtunnel --port 8501
