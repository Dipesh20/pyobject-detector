from flask import Flask, render_template, request
import os
from yolo import *
import time
import math

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def deleteFiles():
    dir = "./static"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        deleteFiles()
        file = request.files['file']
        if file and allowed_file(file.filename):
            extension = file.filename.rsplit('.', 1)[1].lower()
            filename = "image" + str(math.trunc(time.time()))+"." + extension
            filePath = "./static/"+filename
            file.save(filePath)
            new_File_path, outputFile = getYoloOutput(filePath, extension)
            return render_template("index.html", outputImage=new_File_path, inputImage=filePath, outputFile=outputFile)
    return render_template("index.html", outputImage="null", inputImage="null")


if __name__ == "__main__":
    app.run(debug=True)
