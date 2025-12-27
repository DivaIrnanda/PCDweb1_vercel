from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import shutil

# Import models
from models import db, User, Prediction

app = Flask(__name__)

# Konfigurasi
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flood_prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'unet_standar.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Silakan login terlebih dahulu untuk mengakses halaman ini.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Metrik evaluasi model
MODEL_METRICS = {
    'name': 'U-Net Standar',
    'architecture': 'U-Net',
    'iou_score': 0.8652,
    'dice_score': 0.9573,
    'pixel_accuracy': 0.9069
}

# Load model
def load_flood_model(model_path):
    """Load model dengan berbagai fallback options"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

model = load_flood_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(256, 256)):
    """Pra-pemrosesan citra sebelum prediksi"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded, img_resized

def create_overlay(original_img, mask, alpha=0.5):
    """Membuat overlay antara citra asli dan mask prediksi"""
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)
    
    overlay = original_img.copy()
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    if len(mask.shape) != 2:
        raise ValueError(f"Mask shape tidak valid: {mask.shape}")
    
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    red_mask = np.zeros_like(original_img)
    red_mask[:, :, 0] = mask
    
    overlay = cv2.addWeighted(overlay, 1-alpha, red_mask, alpha, 0)
    return overlay

def save_image(image, folder, filename):
    """Save image to disk dan return path"""
    filepath = os.path.join(folder, filename)
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Convert RGB to BGR for cv2.imwrite
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, image_bgr)
    
    return filepath

def image_to_base64(image):
    """Konversi numpy array ke base64 string"""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    pil_img = Image.fromarray(image)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# ==================== ROUTES ====================

# Route utama
@app.route('/')
def index():
    """Halaman utama - redirect ke login jika belum login"""
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html')

# Routes untuk autentikasi
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Halaman registrasi user baru"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            flash('Semua field harus diisi!', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Password tidak cocok!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar!', 'danger')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registrasi berhasil! Silakan login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Halaman login - menggunakan template standalone tanpa navbar"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Selamat datang, {user.username}!', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('Username atau password salah!', 'danger')
    
    return render_template('login_standalone.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user dan redirect ke halaman login"""
    logout_user()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login'))

# Routes halaman aplikasi (memerlukan login)
@app.route('/prediksi')
@login_required
def prediksi():
    """Halaman prediksi segmentasi banjir"""
    return render_template('prediksi.html', metrics=MODEL_METRICS)

@app.route('/riwayat')
@login_required
def riwayat():
    """Halaman riwayat prediksi user"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('riwayat.html', predictions=predictions)

# Routes halaman informasi (tidak perlu login)
@app.route('/informasi-model')
def informasi_model():
    """Halaman informasi model"""
    return render_template('informasi_model.html', metrics=MODEL_METRICS)

@app.route('/panduan')
def panduan():
    """Halaman panduan penggunaan"""
    return render_template('panduan.html')

@app.route('/tentang')
def tentang():
    """Halaman tentang aplikasi"""
    return render_template('tentang.html')

# API endpoints
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint untuk melakukan prediksi segmentasi banjir"""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG atau JPG'}), 400
    
    if model is None:
        return jsonify({'error': 'Model tidak tersedia'}), 500
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{current_user.id}_{timestamp}_{name}{ext}"
        
        # Simpan file original
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(original_path)
        
        # Pra-pemrosesan
        img_processed, img_original = preprocess_image(original_path)
        
        # Prediksi
        prediction = model.predict(img_processed, verbose=0)
        pred_mask = prediction[0]
        
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Buat overlay
        overlay_img = create_overlay(img_original, pred_mask)
        
        # Save hasil ke disk
        mask_filename = f"mask_{unique_filename}"
        overlay_filename = f"overlay_{unique_filename}"
        
        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        mask_path = save_image(pred_mask_rgb, app.config['RESULT_FOLDER'], mask_filename)
        overlay_path = save_image(overlay_img, app.config['RESULT_FOLDER'], overlay_filename)
        
        # Hitung statistik
        total_pixels = pred_mask.size
        flood_pixels = np.sum(pred_mask > 0)
        flood_percentage = (flood_pixels / total_pixels) * 100
        
        # Simpan ke database
        new_prediction = Prediction(
            user_id=current_user.id,
            filename=filename,
            original_image_path=original_path,
            mask_image_path=mask_path,
            overlay_image_path=overlay_path,
            flood_percentage=round(flood_percentage, 2),
            flood_pixels=int(flood_pixels),
            total_pixels=int(total_pixels)
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Konversi ke base64 untuk response
        original_b64 = image_to_base64(img_original)
        mask_b64 = image_to_base64(pred_mask_rgb)
        overlay_b64 = image_to_base64(overlay_img)
        
        return jsonify({
            'success': True,
            'prediction_id': new_prediction.id,
            'original_image': original_b64,
            'mask_image': mask_b64,
            'overlay_image': overlay_b64,
            'statistics': {
                'flood_percentage': round(flood_percentage, 2),
                'flood_pixels': int(flood_pixels),
                'total_pixels': int(total_pixels)
            }
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/prediction/<int:prediction_id>')
@login_required
def get_prediction(prediction_id):
    """API endpoint untuk mendapatkan detail prediksi"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if prediction.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Load images
    original_img = cv2.imread(prediction.original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    mask_img = cv2.imread(prediction.mask_image_path)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    
    overlay_img = cv2.imread(prediction.overlay_image_path)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    
    return jsonify({
        'success': True,
        'prediction': prediction.to_dict(),
        'original_image': image_to_base64(original_img),
        'mask_image': image_to_base64(mask_img),
        'overlay_image': image_to_base64(overlay_img)
    })

@app.route('/prediction/<int:prediction_id>/delete', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """API endpoint untuk menghapus prediksi"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if prediction.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Hapus files
        if os.path.exists(prediction.original_image_path):
            os.remove(prediction.original_image_path)
        if os.path.exists(prediction.mask_image_path):
            os.remove(prediction.mask_image_path)
        if os.path.exists(prediction.overlay_image_path):
            os.remove(prediction.overlay_image_path)
        
        # Hapus dari database
        db.session.delete(prediction)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Prediksi berhasil dihapus'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """API endpoint untuk mendapatkan informasi model"""
    return jsonify(MODEL_METRICS)

# Initialize database
with app.app_context():
    db.create_all()
    print("✓ Database initialized")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)