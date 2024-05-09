from flask import Flask, render_template, request
from PIL import Image
from transformers import pipeline
from pathlib import Path

app = Flask(__name__)

# Load the image classification pipeline
pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        if file:
            # Save the image temporarily
            image_path = Path('temp.png')
            file.save(image_path)

            # Open and resize the image
            image = Image.open(image_path).resize((200, 200))

            # Make prediction using the model
            predictions = pipe(image)

            # Delete the temporary image file
            image_path.unlink()

            # Format the predictions
            formatted_predictions = []
            for prediction in predictions:
                label = prediction['label']
                score = prediction['score']
                formatted_prediction = f"{label} => 'score': {score}"
                formatted_predictions.append(formatted_prediction)

            # Apply conditional formatting for 'parkinson' label
            for i, prediction in enumerate(predictions):
                if prediction['label'] == 'parkinson' and prediction['score'] > 0.5:
                    formatted_predictions[i] = f"<span style='color:red'>{formatted_predictions[i]}</span>"

            return '<br>'.join(formatted_predictions)
    return 'Error'

if __name__ == '__main__':
    app.run(debug=True)
