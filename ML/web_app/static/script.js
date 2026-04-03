document.getElementById('kota').addEventListener('change', function() {
    const kota = this.value;
    const kecamatanSelect = document.getElementById('kecamatan');
    
    if (kota) {
        fetch(`/api/districts/${encodeURIComponent(kota)}`)
            .then(response => response.json())
            .then(districts => {
                kecamatanSelect.innerHTML = '<option value="">Pilih Kecamatan</option>';
                
                districts.forEach(district => {
                    const option = document.createElement('option');
                    option.value = district;
                    option.textContent = district;
                    kecamatanSelect.appendChild(option);
                });
                
                kecamatanSelect.disabled = false;
            })
            .catch(error => {
                console.error('Error fetching districts:', error);
                kecamatanSelect.innerHTML = '<option value="">Error loading districts</option>';
            });
    } else {
        kecamatanSelect.innerHTML = '<option value="">Pilih Kecamatan</option>';
        kecamatanSelect.disabled = true;
    }
});

document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = {
        kota: document.getElementById('kota').value,
        kecamatan: document.getElementById('kecamatan').value,
        luas_tanah: document.getElementById('luas_tanah').value,
        luas_bangunan: document.getElementById('luas_bangunan').value,
        kamar_tidur: document.getElementById('kamar_tidur').value,
        kamar_mandi: document.getElementById('kamar_mandi').value
    };
    
    if (!formData.kota || !formData.kecamatan) {
        alert('Silakan pilih kota dan kecamatan terlebih dahulu.');
        return;
    }
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('prediction-form').style.display = 'none';
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || 'Server error'); });
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        document.getElementById('result-section').style.display = 'block';
        
        document.getElementById('harga-prediksi').textContent = data.harga_formatted;
        document.getElementById('lokasi-rumah').textContent = `${formData.kecamatan}, ${formData.kota}`;
        
        // Show threshold warning if applicable
        const minPriceNote = document.getElementById('min-price-note');
        if (data.is_threshold_applied) {
            minPriceNote.style.display = 'block';
            minPriceNote.innerHTML = `<strong>${data.warning_message}</strong><br>Hasil prediksi model: Rp ${data.prediksi_asli.toLocaleString('id-ID')}<br>Ditampilkan: Rp 100.000.000 (minimum)`;
        } else {
            minPriceNote.style.display = 'none';
        }
        
        document.getElementById('detail-lokasi').textContent = `${formData.kecamatan}, ${formData.kota}`;
        document.getElementById('detail-tanah').textContent = `${formData.luas_tanah} m²`;
        document.getElementById('detail-bangunan').textContent = `${formData.luas_bangunan} m²`;
        document.getElementById('detail-tidur').textContent = formData.kamar_tidur;
        document.getElementById('detail-mandi').textContent = formData.kamar_mandi;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        document.getElementById('prediction-form').style.display = 'block';
        alert('Error: ' + error.message);
    });
});

function resetForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('prediction-form').style.display = 'block';
    document.getElementById('kecamatan').innerHTML = '<option value="">Pilih Kecamatan</option>';
    document.getElementById('kecamatan').disabled = true;
    document.getElementById('min-price-note').style.display = 'none';
}
