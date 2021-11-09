from .. import db, flask_bcrypt
import datetime
import jwt
from app.main.model.blacklist import BlacklistToken
from ..config import key

class Prediction(db.Model):
    """ Prdiction Model is used to store the Prediction Result"""
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    fixed_acidity =db.Column(db.Float, default=0, nullable=False)
    volatile_acidity = db.Column(db.Float, default=0, nullable=False)
    citric_acid = db.Column(db.Float, default=0, nullable=False)
    residual_sugar = db.Column(db.Float, default=0, nullable=False)
    chlorides = db.Column(db.Float, default=0, nullable=False)
    free_sulfur_dioxide = db.Column(db.Integer, default=0, nullable=False)
    total_sulfur_dioxide = db.Column(db.Integer, default=0, nullable=False)
    density = db.Column(db.Float, default=0, nullable=False)
    pH = db.Column(db.Float, default=0, nullable=False)
    sulphates = db.Column(db.Float, default=0, nullable=False)
    alcohol = db.Column(db.Float, default=0, nullable=False)
    quality = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)
    deleted_at = db.Column(db.DateTime, nullable=True)
