from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model_sumatera_lstm.h5')

# Load dataset asli
df = pd.read_csv('sumatera.csv')
df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d/%m/%Y')

features = ['kode_provinsi']
targets = ['beras_medium', 'bawang_merah', 'bawang_putih_bonggol', 'cabai_merah_keriting', 'telur_ayam_ras',
           'gula_konsumsi', 'minyak_goreng_kemasan_sederhana']

# normalisasi
scaler = MinMaxScaler()
df[targets] = scaler.fit_transform(df[targets])

time_steps = 7
def prepare_data(df, time_steps):
    X = []
    for i in range(len(df) - time_steps):
        X.append(df[features + targets].iloc[i:(i+time_steps)].values)
    return np.array(X)

def calculate_accuracy(df, komoditas):
    df['rmse'] = abs(df[f'{komoditas}_real'] - df[f'{komoditas}_prediksi'])
    # df['rmse'] = np.sqrt((df[f'{komoditas}_real'] - df[f'{komoditas}_prediksi'])**2)
    average_price = df[f'{komoditas}_real'].mean()
    df['akurasi'] = 100 * (1 - df['rmse'] / average_price)
    return df.round(2)

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    provinsi = data['provinsi']
    komoditas = data['komoditas']
    future_day = data.get('future_day', 30)

    df_province = df[df['kode_provinsi'] == provinsi]

    last_date = df_province['tanggal'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_day, freq='D')

    X_test_prov = prepare_data(df_province, time_steps)
    future_input = X_test_prov[-future_day:]

    future_predictions = model.predict(future_input)
    future_predictions = scaler.inverse_transform(future_predictions)
    commodity_predictions = future_predictions[:, targets.index(komoditas)]

    future_df = pd.DataFrame({
        'tanggal': future_dates,
        'kode_provinsi': provinsi,
        komoditas: commodity_predictions
    })

    # Load harga sekarang (1 juni - 20 juni)
    real_df = pd.read_csv('hargaa_sekarang.csv')
    real_df = real_df.drop(['provinsi'], axis=1)
    real_df['tanggal'] = pd.to_datetime(real_df['tanggal'], format='%d/%m/%Y')
    real_df = real_df[real_df['kode_provinsi'] == provinsi]

    # real_df['is_real'] = True
    # future_df['is_real'] = False

    # real_predik = pd.concat([real_df, future_df], ignore_index=True)
    # real_predik = real_predik.sort_values('tanggal')

    merged_df = pd.merge(real_df[['tanggal', 'kode_provinsi', komoditas]],
                        future_df[['tanggal', 'kode_provinsi', komoditas]],
                        on=['tanggal', 'kode_provinsi'],
                        #  how='outer',
                        suffixes=('_real', '_prediksi'))

    merged_df = merged_df.sort_values('tanggal')

    result_df = calculate_accuracy(merged_df, komoditas)

    return jsonify({
        'status': {
            'code': 200,
            'message': 'Success predicting',
        },
                
        'predictions': result_df.to_dict(orient='records'),
        'accuracy': {
            'rmse_mean': float(f"{result_df['rmse'].mean():.2f}"),
            'accuracy_mean': float(f"{result_df['akurasi'].mean():.2f}")
        }
    }), 200

@app.route('/plot', methods=['POST'])
def plot():
    data = request.json
    provinsi = data['provinsi']
    komoditas = data['komoditas']
    future_day = data.get('future_day', 30)

    df_province = df[df['kode_provinsi'] == provinsi]

    last_date = df_province['tanggal'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_day, freq='D')

    X_test_prov = prepare_data(df_province, time_steps)
    future_input = X_test_prov[-future_day:]

    future_predictions = model.predict(future_input)
    future_predictions = scaler.inverse_transform(future_predictions)
    commodity_predictions = future_predictions[:, targets.index(komoditas)]

    future_df = pd.DataFrame({
        'tanggal': future_dates,
        'kode_provinsi': provinsi,
        komoditas: commodity_predictions
    })

    real_df = pd.read_csv('hargaa_sekarang.csv')
    real_df = real_df.drop(['provinsi'], axis=1)
    real_df['tanggal'] = pd.to_datetime(real_df['tanggal'], format='%d/%m/%Y')
    real_df = real_df[real_df['kode_provinsi'] == provinsi]

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(real_df['tanggal'], real_df[komoditas], label='Real', color='b')
    plt.plot(future_df['tanggal'], future_df[komoditas], label='Prediksi', color='orange')
    plt.title(f'Perbandingan {komoditas.replace("_", " ").title()} Provinsi {provinsi}', fontsize=14)
    plt.ylabel('Harga')
    plt.xlabel('Tanggal')
    plt.legend(loc='best')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
