import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('model_best_weights.h5')
  return model
model=load_model()
st.title("Car Brand Model Classifier")
file=st.file_uploader("Choose a photo from your computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(224,224)
    image = ImageOps.fit(image_data,size, Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    img_reshape = np.reshape(image, (1, 224, 224, 3))
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Daiatsu_Core', 'Daiatsu_Hijet', 'Daiatsu_Mira', 'FAW_V2', 'FAW_XPV', 'Honda_BRV', 'Honda_city_1994', 'Honda_city_2000', 'Honda_City_aspire', 'Honda_civic_1994', 'Honda_civic_2005', 'Honda_civic_2007', 'Honda_civic_2015', 'Honda_civic_2018', 'Honda_Grace', 'Honda_Vezell', 'KIA_Sportage', 'Suzuki_alto_2007', 'Suzuki_alto_2019', 'Suzuki_alto_japan_2010', 'Suzuki_carry', 'Suzuki_cultus_2018', 'Suzuki_cultus_2019', 'Suzuki_Every', 'Suzuki_highroof', 'Suzuki_kyber', 'Suzuki_liana', 'Suzuki_margala', 'Suzuki_Mehran', 'Suzuki_swift', 'Suzuki_wagonR_2015', 'Toyota_HIACE_2000', 'Toyota_Aqua', 'Toyota_axio', 'Toyota_corolla_2000', 'Toyota_corolla_2007', 'Toyota_corolla_2011', 'Toyota_corolla_2016', 'Toyota_fortuner', 'Toyota_Hiace_2012', 'Toyota_Landcruiser', 'Toyota_Passo', 'Toyota_pirus', 'Toyota_Prado', 'Toyota_premio', 'Toyota_Vigo', 'Toyota_Vitz', 'Toyota_Vitz_2010']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
