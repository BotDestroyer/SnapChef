from flask import Flask, render_template, Response, request
import os
import time
import cv2
import pandas as pd
from num2words import num2words
from bing_image_urls import bing_image_urls
from roboflow import Roboflow
import re

rf = Roboflow(api_key="9odmIusQuu9SiPLzm0Ur")
project = rf.workspace().project("food-ingredients-image-detection_team4")
model = project.version(1).model

classes=[]
global capture
capture = 0

try:
    os.mkdir('./static')
except OSError as error:
    pass

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)

captured_image_path = os.path.sep.join(['static', "shot.png"])
detect_image_path = os.path.sep.join(['static', "predicted_shot.png"])

def gen_frames():
    global capture
    while True:
        success, frame = camera.read()
        if success:
            if (capture):
                capture = 0
                cv2.imwrite(captured_image_path, frame)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


def ingredient_detection(image):
    global classes
    classes.clear()
    recipes = []
    file = (model.predict(image, confidence=40, overlap=30).json())

    if (os.path.isfile("static/predicted_shot.png")):
        os.remove("static/predicted_shot.png")

    model.predict(image, confidence=40, overlap=30).save("static/predicted_shot.png")

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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
    time.sleep(0.5)
    if(os.path.isfile("static/shot.png")):
        recipes = ingredient_detection(captured_image_path)
    return render_template('index.html',captured_image=captured_image_path,detect_image=detect_image_path,classes=classes,recipes=recipes)

if __name__ == '__main__':
    app.run()

camera.release()
if(os.path.isfile("static/shot.png")):
    os.remove('./static/shot.png')
if(os.path.isfile("static/predicted_shot.png")):
    os.remove("static/predicted_shot.png")
cv2.destroyAllWindows()