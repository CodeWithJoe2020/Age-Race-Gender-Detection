from flask import Flask, request, jsonify 
import cv2 
from deepface import DeepFace 
import os
import urllib.request

app = Flask(__name__)


@app.route("/detect/<path:url>")
def get_info(url):
    # Download image from URL and read it into memory
    urllib.request.urlretrieve(url, "temp.jpg")
    img = cv2.imread("temp.jpg")
    
    # Analyze image using DeepFace
    results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))
    print(results)
    
    # Remove temporary image file
    os.remove("temp.jpg")
    
    return jsonify(results)




if __name__ == '__main__':
    app.run()
