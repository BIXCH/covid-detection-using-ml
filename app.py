from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model(r'D:\new\Covid19-dataset\model\covid_xray_model.h5')



# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        test_image = image.load_img(filepath, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict the result
        result = model.predict(test_image)

        # Get prediction (COVID or Normal)
        if result[0][0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'COVID'

        return render_template('index.html', prediction=prediction, filename=filename)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
