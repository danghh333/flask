from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from skimage import color
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
import spectral as sp

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU for Flask app




app = Flask(__name__)

# Load pre-trained model
model = load_model('/home/student4/Test/Flask/checkpoint.h5')

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = '/home/student4/Test/Flask/Upload' # Path to save RGB image after user uploaded
PREDICT_FOLDER = '/home/student4/Test/Flask/Predict' # Path to save predicted image after user uploaded
ALLOWED_EXTENSIONS = {'hdr', 'img'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def linear_stretch(image):
    min_val, max_val = np.percentile(image, (1, 99))
    stretched_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)
    return stretched_image

def preview(hdr_path, img_path):
    img = sp.open_image(hdr_path)
    red_band = img.read_band(50)
    green_band = img.read_band(30)
    blue_band = img.read_band(20)
    rgb_image = np.dstack((red_band, green_band, blue_band))
    stretched_red_band = linear_stretch(red_band)
    stretched_green_band = linear_stretch(green_band)
    stretched_blue_band = linear_stretch(blue_band)

    # Create a false color RGB image with stretched bands
    stretched_rgb_image = np.dstack((stretched_red_band, stretched_green_band, stretched_blue_band))
    return stretched_rgb_image
    
def crop_and_stack_bands(img, crop_size=144, start_x=None, start_y=None):
    # Get the number of bands
    num_bands = img.shape[2]

    # Initialize an empty list to store the cropped bands
    cropped_bands_list = []

    # Loop through each band
    for band in range(num_bands):
        # Read the current band
        current_band = img.read_band(band)  # Bands in Spectral start from 1

        # Set default starting points if not provided
        if start_x is None:
            start_x = current_band.shape[0] // 2 - crop_size // 2
        if start_y is None:
            start_y = current_band.shape[1] // 2 - crop_size // 2

        # Calculate the cropping boundaries
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # Crop the band
        cropped_band = current_band[start_x:end_x, start_y:end_y]

        # Append the cropped band to the list
        cropped_bands_list.append(cropped_band)

    # Convert the list of cropped bands to a NumPy array
    cropped_bands = np.stack(cropped_bands_list, axis=-1)
    return cropped_bands

def preprocess_and_predict(model, hdr_path, img_path):
    # Load the hyperspectral image
    img = sp.open_image(hdr_path)

    # Crop and stack bands
    cropped_array = crop_and_stack_bands(img, start_x=4000, start_y=4000)

    # PCA
    pca = PCA(30)
    reshaped_array = cropped_array.reshape(-1, cropped_array.shape[2])
    pca_result = pca.fit_transform(reshaped_array)
    pca_result_reshaped = pca_result.reshape(cropped_array.shape[0], cropped_array.shape[1], 30)

    # Reshape the data to match the model input shape
    input_reshaped = pca_result_reshaped.reshape(1, 144, 144, 30, 1)
    
    # Make predictions
    predictions = model.predict(input_reshaped)
    
    # Return the predicted mask
    return predictions[0]


@app.route('/preview')
def demo_preview():
    # Provide the HSI path parameters
    hdr_path = '/home/student4/HSI/Hyper-Spectral/hyper_20220326_3cm.hdr'
    img_path = '/home/student4/HSI/Hyper-Spectral/hyper_20220326_3cm.img'

    preview_image = preview(hdr_path, img_path)

    # Save the preview image as a PNG file
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_preview.png')
    plt.imsave(output_path, preview_image)

    # Return the path to the saved PNG file in the response
    response = {'demo_preview_path': output_path}
    return jsonify(response)

@app.route('/predict')
def demo_predict():
    # Provide the HSI path parameters
    hdr_path = '/home/student4/HSI/Hyper-Spectral/hyper_20220326_3cm.hdr'
    img_path = '/home/student4/HSI/Hyper-Spectral/hyper_20220326_3cm.img'

    # Perform model prediction
    predicted_mask = preprocess_and_predict(model, hdr_path, img_path)

    # Save the predicted mask as a PNG file
    predict_path = os.path.join(app.config['PREDICT_FOLDER'], 'demo_predict.png')
    plt.imsave(predict_path, np.argmax(predicted_mask, axis=-1), cmap='jet')

    # Return the paths to the saved PNG files in the response
    response = {
        'demo_predict_path': predict_path
    }
    return jsonify(response)

if __name__ == '__main__':
    port = 5555
    app.run(debug=True, port=port)
