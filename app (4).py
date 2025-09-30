
import pickle
import os
from flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Define the directory for the web application
app_dir = 'hiv_adherence_app'

# Define the paths to the saved model and preprocessor within the app directory
model_filename = os.path.join(app_dir, 'logistic_regression_model.pkl')
preprocessor_filename = os.path.join(app_dir, 'preprocessor.pkl')

# Load the trained model and preprocessor when the application starts
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {model_filename}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_filename}")
    model = None # Or handle the error appropriately

try:
    with open(preprocessor_filename, 'rb') as file:
        preprocessor = pickle.load(file)
    print(f"Preprocessor loaded successfully from {preprocessor_filename}")
except FileNotFoundError:
    print(f"Error: Preprocessor file not found at {preprocessor_filename}")
    preprocessor = None # Or handle the error appropriately


@app.route('/')
def index():
    return "HIV Adherence Prediction App"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)

        # Convert the incoming data to a pandas DataFrame
        # This assumes the incoming JSON has keys corresponding to your feature names
        # You might need to adjust this based on the exact format of your incoming data
        input_df = pd.DataFrame([data])

        # Preprocess the input data
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_processed)

        # Convert prediction to a human-readable format
        # Assuming your target variable was 'haspendingvl' with values 'Yes' and 'No'
        prediction_result = 'Yes' if prediction[0] == 1 else 'No' # Adjust based on your model's output

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # In a production environment, you would not run with debug=True
    app.run(debug=True, use_reloader=False) # Set use_reloader to False when loading external files at startup
