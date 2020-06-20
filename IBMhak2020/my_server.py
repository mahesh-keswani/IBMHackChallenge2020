from flask import Flask, redirect, url_for, jsonify, request, render_template
import os

app = Flask(__name__, static_folder="./static/")  

def load_model():
	from tensorflow.keras.models import model_from_json
	# print("Loading model")
	with open('wind_turbine_architecture_20_6.json', 'r') as f:
		model = model_from_json(f.read())

	model.load_weight('wind_turbine_weights_20_6.h5')
	return model

def fetch_and_normalize_data_and_predict_data(number):
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.metrics import mean_squared_error
	from numpy.random import rand

	# print("In fetch_and_normalize_data_and_predict_data")
	n_steps_in = 144
	n_steps_out = 72
	n_features = 6

	df = pd.read_csv('integrated_data.csv')
	df.drop('end_date', axis = 1, inplace = True)
	df.drop(['precipMM', 'pressure', 'maxtempC', 'humidity'], axis = 1, inplace = True)

	# print("Loaded the file")
	sc = MinMaxScaler()
	scaled_data = sc.fit_transform(df.values[(number - n_steps_in) : (number + n_steps_out), :])

	scaled_x = scaled_data[:number,  :-1]
	scaled_y = scaled_data[-n_steps_out: , -1] 
	
	# print("scaling done!!!")
	model = load_model()
	print("Model loaded!!")
	yhat = model.predict(scaled_x.reshape(1, n_steps_in, n_features))
	y_hat_reshaped = yhat.reshape(yhat.shape[1])

	mse = mean_squared_error(y_hat_reshaped, scaled_y)
	# print("mse",mse)
	
	return  y_hat_reshaped, scaled_y, mse

def create_plot(prediction, true, error):
	import plotly
	import plotly.graph_objects as go
	import json

	fig = go.Figure()

	fig.add_trace(
	    go.Scatter(
	        x=prediction,
	        y=list(range(72)),
	        mode='lines', name='prediction',
                    opacity=0.8, marker_color='orange'
	    ))

	fig.add_trace(
	    go.Scatter(
	        x=true,
	        y=list(range(72)),
	        mode='lines', name='True data',
                    opacity=0.8, marker_color='blue'
	    ))

	fig.show()
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return graphJSON

@app.route('/',methods = ['GET']) 
def about():
	EPOCHS = 30
	BATCH_SIZE = 32
	LEARNING_RATE = 0.0001
	NUMBER_OF_STEPS_IN = 144
	NUMBER_OF_STEPS_OUT = 72
	CLIP_NORM = 0.5

	data = {
			'epochs':EPOCHS, 'batch_size':BATCH_SIZE, 'lr':LEARNING_RATE,
			'n_steps_in':NUMBER_OF_STEPS_IN,'n_steps_out':NUMBER_OF_STEPS_OUT,
			'clip_norm':CLIP_NORM
			}
	return render_template('about.html', data = data)

@app.route('/evaluate',methods = ['GET']) 
def evaluate(): 
	return render_template('evaluate.html')

@app.route('/show_result', methods = ['POST'])
def show_result():
	if request.method == "POST":
		number = int(request.form['number'])
		prediction, true, error = fetch_and_normalize_data_and_predict_data(number)

		plot = create_plot(prediction, true, error)
		return render_template('graph.html', plot = plot)

if __name__ == '__main__': 
   app.run()