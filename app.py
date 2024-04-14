import streamlit as st
import numpy as np
import pickle

# Load the model and scaler from the .pth file
@st.cache_data
def load_model():
    with open('random_forest_model.pth', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']

model, scaler = load_model()

# Streamlit webpage setup
st.title("House.AI's Price")

# Form for input
with st.form(key='input_form'):
    city = st.text_input('City')  # Placeholder for future model updates
    rooms = st.number_input('Rooms', min_value=0, value=3, step=1)
    area = st.number_input('Area', min_value=10.0, value=50.0, step=5.0)
    level = st.number_input('Level', min_value=0, value=5, step=1)
    levels = st.number_input('Total Levels in Building', min_value=1, value=10, step=1)
    kitchen_area = st.number_input('Kitchen Area', min_value=5.0, value=10.0, step=5.0)
    building_type = st.selectbox('Building Type', ['Type0', 'Type1','Type2','Type3','Type4','Type5','Type6'])
    object_type = st.selectbox('Object Type', ['Object0', 'Object2'])
    distance = st.number_input('Distance to City Center', min_value=0.0, value=5.0, step=1.0)
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Encode inputs for categorical data using dictionaries
    building_type_encoding = {'Type0': 0, 'Type1': 1, 'Type2': 2, 'Type3': 3, 'Type4': 4, 'Type5': 5, 'Type6': 6}
    object_type_encoding = {'Object0': 0, 'Object2': 2}
    
    input_data = np.array([[rooms, area, level, levels, kitchen_area,
                            building_type_encoding[building_type],
                            object_type_encoding[object_type], distance]])
    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # Make prediction using the loaded model
    prediction = model.predict(input_scaled)
    output = round(prediction[0], 2)

    # Formatting the output to include commas
    formatted_output = f"{output:,.2f}"  # Adds commas as thousand separators and rounds to 2 decimal places

    st.success(f'The predicted house price is ₽{formatted_output}')

    

