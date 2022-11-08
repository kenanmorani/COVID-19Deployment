import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import skimage

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
#import nibabel as nib

#from PIL import Image as im

@st.cache(allow_output_mutation=True)

## Segmentation
def load_model():
  UNet_model=tf.keras.models.load_model('/content/COVID-19Deployment-PipeLine/UNet_model-3L-BatchNorm.h5')
  return UNet_model

## Classification for COVID-19
def load_model2():
  model=tf.keras.models.load_model('/content/COVID-19Deployment-PipeLine/UNet-BatchNorm-CNN-model.h5')
  return model

with st.spinner('Model is being loaded..'):
  UNet_model=load_model()
  model=load_model2()

st.write("""
         # COVID-19 diagnosis via 2D Grayscale Slices - pre-trained on COV19-DT-DB
         """
         )

file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, UNet_model ,model):
    
        size = (224,224)  
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        
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
        print('The probability that the image is heathy is', prediction)
        prediction = prediction.astype(np.float)
        #print('The probability that the image is heathy is', prediction)

        if prediction > 0.5:
         pred_class='NON-COVID'
        else:
         pred_class='COVID'
        
        
        return pred_class

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, UNet_model, model)
    image_class = predictions
    #score=np.round(predictions[0][0][2],5) 
    st.write("The image is classified as", predictions)
    #st.write("The similarity score is approximately",score)
    print("The image is classified as ",image_class,)
