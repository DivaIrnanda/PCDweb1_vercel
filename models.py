"""
Database models untuk User dan Prediction History
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Model untuk user"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relasi ke predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Model untuk riwayat prediksi"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Info file
    filename = db.Column(db.String(255), nullable=False)
    original_image_path = db.Column(db.String(500), nullable=False)
    
    # Hasil prediksi
    mask_image_path = db.Column(db.String(500), nullable=False)
    overlay_image_path = db.Column(db.String(500), nullable=False)
    
    # Statistik
    flood_percentage = db.Column(db.Float, nullable=False)
    flood_pixels = db.Column(db.Integer, nullable=False)
    total_pixels = db.Column(db.Integer, nullable=False)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.filename}>'
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'flood_percentage': self.flood_percentage,
            'flood_pixels': self.flood_pixels,
            'total_pixels': self.total_pixels,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'notes': self.notes
        }