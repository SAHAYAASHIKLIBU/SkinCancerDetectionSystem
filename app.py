import os
import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Set up the model
model = YOLO('best.pt')

# Set up paths
UPLOAD_FOLDER = 'static/images/uploads'
OUTPUT_FOLDER = 'static/images/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process the image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Secure the file name and save the image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform detection
        result = model(filepath)[0]
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + filename)

        # Plot the result and save it as an image
        annotated_image = result.plot()  # Get annotated image
        cv2.imwrite(output_image_path, annotated_image)  # Save the output image

        return render_template('index.html', uploaded_image=filename, output_image='output_' + filename)

    return 'Invalid file format. Please upload an image file (jpg, jpeg, png).'

# Serve the uploaded and output images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
