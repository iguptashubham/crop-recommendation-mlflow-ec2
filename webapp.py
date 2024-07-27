import streamlit as st 
import pickle
import numpy as np
st.set_page_config(layout='centered',page_title='Crop Recommendation')


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background-image: url({"https://images.pexels.com/photos/540925/pexels-photo-540925.jpeg"});
  background-size: 180%;
  background-position: center; /* Center the image */
  background-repeat: no-repeat;
  background-attachment: local;
  /* Apply a darkening overlay */
  background-color: rgba(0, 0, 0, 0.5);
}}

[data-testid="stHeader"] {{
  background: rgba(0, 0, 0, 0);
}}

[data-testid="stToolbar"] {{
  right: 2rem;
}}
</style>
"""


st.markdown(f"# Crop Recommendation")
st.write('Crop Recommendation by machine learning model. \n Enter the input values and get the recommended crop.')

st.markdown(page_bg_img, unsafe_allow_html=True)



def load_model(run_id=None):
    model_path = f"mlruns\\0\\{run_id}\\artifacts\\model\\model.pkl"
    model = pickle.load(open(model_path, "rb"))
    return model

model = load_model(run_id="7f69131bfd3640df920f9c81fd13bb48")
c1,_,c2 = st.columns([0.4,0.2,0.4])

with c1:
  n = st.number_input(label='Enter the N')
  p = st.number_input(label='Enter the P')
  k = st.number_input(label='Enter the K')
  
with c2:
  temp = st.number_input(label='Enter the Temprature')
  humi = st.number_input(label='Enter the Humidity')
  ph = st.number_input(label='Enter the ph')
rain = st.number_input(label='Enter the Rainfall')

inputs = np.array([[n,p,k,temp,humi,ph,rain]]).reshape(1,7)
button = st.button(label = 'Recommend')

if button == True:
  recom = int(model.predict(inputs))
  
  crop_dict = {
        1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut',
        6: 'Papaya', 7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon',
        11: 'Grapes', 12: 'Mango', 13: 'Banana', 14: 'Pomegranate',
        15: 'Lentil', 16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans',
        19: 'Pigeonpeas', 20: 'Kidneybeans', 21: 'Chickpea', 22: 'Coffee'
    }
  crop_name = crop_dict.get(recom, f"Unknown ({recom})")
  st.success(f"{crop_name}")

