from flask import Flask, render_template, request, jsonify
import uuid
import os
import time
import pandas as pd
from num2words import num2words
from bing_image_urls import bing_image_urls
from roboflow import Roboflow
import re
import base64

rf = Roboflow(api_key="9odmIusQuu9SiPLzm0Ur")
project = rf.workspace().project("food-ingredients-image-detection_team4")
model = project.version(1).model

try:
    os.mkdir('./static')
except OSError as error:
    pass

app = Flask(__name__, template_folder='./templates')
app.secret_key = '5765ho7Jeff'

global capture
capture = 0
classes=[]
def ingredient_detection(image):
    global classes
    classes.clear()
    recipes = []
    file = (model.predict(image, confidence=40, overlap=30).json())
    with open('static/uploads/detected_path.txt', 'r') as text_file:
        detect_path = text_file.read().rstrip()
    model.predict(image, confidence=40, overlap=30).save(detect_path)

    classes = [re.sub(r'[^a-zA-Z]', '', prediction["class"]) for prediction in file["predictions"]]

    if(len(classes) != 0):
        df = pd.read_csv("recipes/recipes.csv")
        filtered_df = df[df['Ingredients'].apply(lambda x: all(label in x for label in classes))]

        for index, row in filtered_df.head(6).iterrows():
            ingredients_string = row['Cleaned_Ingredients']
            ingredients_string = ingredients_string.replace('[', '').replace(']', '').replace("'", "")
            ingredients_list = ingredients_string.split(', ')

            instructions = row['Instructions']
            instructions_list = instructions.split('. ')

            num_index=num2words(index)
            num_index=num_index.replace(" ", "")

            recipes.append({
                'id': num_index,
                'title': row['Title'],
                'ingredients': row['Ingredients'],
                'image': row['Image_Name'],
                'cleaned_ingredients':ingredients_list,
                'instructions': instructions_list
            })
        for recipe in recipes:
            recipe['image'] = bing_image_urls(recipe['image'], limit=1)[0]
    else:
        recipes.append('none')
    return recipes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    print("task start")
    recipes=[]
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
    print("searching for path")
    with open('static/uploads/path.txt', 'r') as file:
        read_path = file.read().rstrip()
    with open('static/uploads/detected_path.txt', 'r') as file:
        detect_path = file.read().rstrip()
    print(read_path)
    recipes = ingredient_detection(read_path)
    print("task end")
    return render_template('index.html',captured_image=read_path,detect_image=detect_path,classes=classes,recipes=recipes)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    print("upload start")
    data = request.get_data()
    base64_string = data.decode('utf-8')
    cleaned_base64 = base64_string.split(",", 1)[-1]
    cleaned_base64_bytes = cleaned_base64.encode('utf-8')

    image_path = f"static/uploads/captured_{uuid.uuid4()}.png"
    original_uuid = image_path.split('_')[-1].split('.')[0]
    detected_image_path = f"static/uploads/detected_captured_{original_uuid}.png"
    print("path set")
    print("path is set to " + image_path)

    with open("static/uploads/path.txt", "w") as text_file:
        text_file.write(image_path)
    with open("static/uploads/detected_path.txt", "w") as text_file:
        text_file.write(detected_image_path)

    with open(image_path, "wb") as fh:
        fh.write(base64.decodebytes(cleaned_base64_bytes))
    print("upload end")
    return jsonify({'message': 'Image uploaded successfully'})

if __name__ == '__main__':
    app.run()