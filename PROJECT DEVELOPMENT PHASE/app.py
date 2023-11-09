from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('D:\VIT\CERTIFICATION COURSES\AI and ML\PROJECT - 5G RESOURCE ALLOCATION\result.html', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])
    input_data = input_data.drop('Application_Type', axis=1)  # Remove non-numeric column
    prediction = model.predict(input_data)
    return render_template('result.html', prediction_text='Predicted Allocated Bandwidth: {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True,port=5500)



# from flask import Flask, request, render_template
# import pickle
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# app = Flask(__name__)

# # Load the pickled model
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('"D:\VIT\CERTIFICATION COURSES\AI and ML\PROJECT - 5G RESOURCE ALLOCATION\result.html"', methods=['POST'])
# def predict():
#     data = request.form.to_dict()
#     input_data = pd.DataFrame([data])

#     # Encode the categorical column 'Application_Type'
#     onehot_encoder = OneHotEncoder(sparse=False)
#     encoded_data = onehot_encoder.fit_transform(input_data[['Application_Type']])
#     categories = onehot_encoder.categories_[0]
#     encoded_df = pd.DataFrame(encoded_data, columns=[f'Application_Type_{category}' for category in categories])

#     # Concatenate the encoded data with the rest of the input data
#     input_data_encoded = pd.concat([input_data.drop('Application_Type', axis=1), encoded_df], axis=1)

#     prediction = model.predict(input_data_encoded)
#     return render_template('result.html', prediction_text='Predicted Allocated Bandwidth: {}'.format(prediction[0]))

# if __name__ == '__main__':
#     app.run(debug=True, port=5500)


