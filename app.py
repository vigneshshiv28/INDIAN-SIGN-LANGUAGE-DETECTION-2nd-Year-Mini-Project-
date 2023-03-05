import cv2
import numpy as np
from flask import Flask, render_template, Response
from keras.models import load_model
import tensorflow as tf
from PIL import Image
#Loading isl prediction model
model = tf.keras.models.load_model('model.h5')
# Define a function to preprocess the input image
def preprocess_image(image):
    image = Image.fromarray(image).convert('L').resize((64,64))
    image_array = np.array(image).reshape(1,64,64,1)
    image_array = image_array/255.0
    return image_array
# Define a function to predict signs
def predict_sign(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_sign = np.argmax(prediction)
    signs = ['1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_sign = signs[predicted_sign]
    return predicted_sign
# Create a Flask app
app = Flask(__name__,template_folder='views')

# Create a Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')
# Define a function to capture frames from the web camera
def webcam_feed():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            # Display the video stream
            cv2.imshow('Webcam Feed', frame)

            # Wait for the 'q' key to be pressed to stop the program
            if cv2.waitKey(1) == ord('q'):
                break

            # Predict the ASL sign
            predicted_sign = predict_sign(frame)

            # Create a JPEG image from the frame
            _, jpeg = cv2.imencode('.jpg', frame)

            # Yield the JPEG image with the predicted sign
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + predicted_sign.encode() + b'\r\n')
        else:
            break

    # Release the web camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
# Create a Flask route for the prediction page
@app.route('/predict')
def predict():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)