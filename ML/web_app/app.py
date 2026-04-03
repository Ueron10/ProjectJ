from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'model_regresi.pkl'))

with open(os.path.join(BASE_DIR, 'models', 'city_district_mapping.json'), 'r', encoding='utf-8') as f:
    city_district_map = json.load(f)

with open(os.path.join(BASE_DIR, 'models', 'feature_columns.json'), 'r', encoding='utf-8') as f:
    feature_columns = json.load(f)

@app.route('/')
def index():
    cities = sorted(city_district_map.keys())
    return render_template('index.html', cities=cities)

@app.route('/api/districts/<city>')
def get_districts(city):
    districts = city_district_map.get(city, [])
    return jsonify(districts)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        kota = data.get('kota')
        kecamatan = data.get('kecamatan')
        luas_tanah = float(data.get('luas_tanah', 0))
        luas_bangunan = float(data.get('luas_bangunan', 0))
        kamar_tidur = int(data.get('kamar_tidur', 0))
        kamar_mandi = int(data.get('kamar_mandi', 0))
        
        if not kota or not kecamatan:
            return jsonify({'error': 'Kota dan kecamatan harus diisi'}), 400
        
        input_data = {col: 0 for col in feature_columns}
        input_data['luas_tanah'] = luas_tanah
        input_data['luas_bangunan'] = luas_bangunan
        input_data['kamar_tidur'] = kamar_tidur
        input_data['kamar_mandi'] = kamar_mandi
        
        kota_col = f"kota_{kota}"
        kecamatan_col = f"kecamatan_{kecamatan}"
        
        if kota_col not in input_data:
            return jsonify({'error': f'Kota "{kota}" tidak ditemukan dalam dataset'}), 400
        if kecamatan_col not in input_data:
            return jsonify({'error': f'Kecamatan "{kecamatan}" tidak ditemukan dalam dataset'}), 400
        
        input_data[kota_col] = 1
        input_data[kecamatan_col] = 1
        
        input_df = pd.DataFrame([input_data])
        
        input_df = input_df[feature_columns]
        
        prediksi = model.predict(input_df)
        prediksi_raw = float(prediksi.flatten()[0])
        
        # Threshold sederhana: minimum Rp 100 juta
        if prediksi_raw < 100000000:
            prediksi_final = 100000000
            is_threshold_applied = True
            warning_message = "Model tidak dapat memprediksi untuk spesifikasi ini. Data training tidak memiliki rumah sekecil ini di area tersebut."
        else:
            prediksi_final = prediksi_raw
            is_threshold_applied = False
            warning_message = ""
        
        return jsonify({
            'harga_prediksi': float(prediksi_final),
            'harga_formatted': f"Rp {prediksi_final:,.0f}",
            'prediksi_asli': float(prediksi_raw),
            'is_threshold_applied': is_threshold_applied,
            'warning_message': warning_message
        })
    except Exception as e:
        import traceback
        print(f"Error in predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
