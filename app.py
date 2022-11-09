import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import skimage
import cv2
from skimage import morphology
from skimage import segmentation
import scipy
# import pandas as pd 

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing, binary_dilation, binary_opening
from skimage.measure import label,regionprops, perimeter 
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
#import matplotlib.pyplot as plt

import scipy.misc


#from PIL import Image as im

@st.cache(allow_output_mutation=True)

## Segmentation
def load_model():
  UNet_model=tf.keras.models.load_model('UNet_model-3L-BatchNorm.h5')
  return UNet_model

## Classification for COVID-19
def load_model2():
  model=tf.keras.models.load_model('UNet-BatchNorm-CNN-model.h5')
  return model

with st.spinner('Model is being loaded..'):
  UNet_model=load_model()
  model=load_model2()

st.write("""
         # COVID-19 diagnosis via 2D Grayscale Slices - pre-trained on COV19-CT-DB
         """
         )
st.write("""
         The method details @ https://github.com/IDU-CVLab/COV19D_3rd
         
         """
         )
st.write("""
         By Kenan Morani @ Izmir Democracy University
         
         """
         )

#folder_path = st.text_input('Enter a file path:')
#st.write(folder_path)
#while not os.path.isfile(folder_path):
    #fileName = input("Whoops! No such file! Please enter the name of the file you'd like to use.")
#try:
    #with open(folder_path) as input:
        #st.text(input.read())
#except FileNotFoundError:
    #st.error('File not found.')
  
uploaded_files = st.file_uploader("Select all slices from one CT scan", accept_multiple_files=True)


from PIL import Image, ImageOps
import numpy as np
# st.set_option('deprecation.showfileUploaderEncoding', False)




def upload_predict(image, UNet_model ,model):
    
   
        #image = ImageOps.fit(upload_image, (224,224), Image.BICUBICS)
        
        image = np.asarray(image)
        n = image
        #image = cv2.resize(image, dim)
        image = image * 100.0 / 255.0  
        image = image / 255.0

        #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        image = image[None]
        #img_reshape = img_resize
        
        #img_reshape = img_resize[np.newaxis,...]

        ## Segemtnation
        image = UNet_model.predict(image) > 0.5
        image = np.squeeze(image)
        image = np.asarray(image, dtype="uint8")
        
        #img_reshape = img_reshape * 100.0 / 255.0
        #Prediction11 = Prediction11 * 100.0 / 255.0

        
        #Prediction11 = Unet_model.predict(img_reshape)
        image = np.squeeze(image)

        # Lung  Exctrction
        image = skimage.segmentation.clear_border (image)
        #label_image = cleared
        label_image = skimage.measure.label(image)

        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
         for region in regionprops(label_image):
          if region.area < areas[-2]:
            for coordinates in region.coords:                
             label_image[coordinates[0], coordinates[1]] = 0
        label_image = label_image > 0
        
        selem = disk(2)
        label_image = skimage.morphology.binary_erosion(label_image, selem)
 
        selem = disk(10)
        label_image = skimage.morphology.binary_closing(label_image, selem)

        label_image = roberts(label_image)
        label_image = scipy.ndimage.binary_fill_holes(label_image)

        label_image=label_image.astype(np.uint8)
        final = cv2.bitwise_and(n, n, mask=label_image)

        

        #final = im.fromarray(final)
        final = final / 255.0
        final = np.expand_dims(final, axis=-1)
        final = final[None]

        #Taking diagnosis Decision through CNN model
        #prediction = model.predict_proba(final)
        prediction = model.predict(final)
        #print('The probability that the image is heathy is', prediction)
        #prediction = prediction.astype(np.float)
        #print('The probability that the image is heathy is', prediction)

        #if prediction > 0.5:
         #pred_class='NON-COVID'
        #else:
         #pred_class='COVID'
        
        
        return prediction



extensions8 = []
extensions9 = []

results =1

for uploaded_file in uploaded_files:
    #if uploaded_file is not None:
            size = (224,224)
            st.write ('tupe is', type(uploaded_file))
            image_PIL = uploaded_file.convert("L")
            #image_PIL = Image.fromarray(uploaded_file)
            image = ImageOps.fit(image_PIL, size, Image.ANTIALIAS)
            #image = ImageOps.fit(uploaded_file, (224,224), Image.LANCZOS)
            result = upload_predict(image, UNet_model ,model)
            if result > 0.50:   
             extensions9.append(results)
            else:
             extensions8.append(results)           
        #print(sub_folder_path, end="\r \n")
        ## The majority voting at Patient's level
            extensions8=[]
            extensions9=[]
    #else:
            #path_in = None
      

#for filee in os.listdir(folder_path):
        #file_path = os.path.join(folder_path, filee)
        #st.write(file_path)
        #result = upload_predict(file_path, UNet_model, model)
    
        
       
        #if result > 0.50:   # Class probability threshod is 0.50
           #extensions9.append(results)
        #else:
           #extensions8.append(results)           
        #print(sub_folder_path, end="\r \n")
        ## The majority voting at Patient's level
        #extensions8=[]
        #extensions9=[]
    
if len(extensions9) >  len(extensions8):
      st.write("The Patient is NEGATIVE for COVID")
      
else:
      st.write("The Patient is POSITIVE for COVID")
      
       
st.write("Email @ kenan.morani@gmail.com, Webpage: https://github.com/kenanmorani")
