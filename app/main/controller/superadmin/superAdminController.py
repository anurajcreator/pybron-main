from app.main.service.stage_01_load_and_save import get_data, get_data_new, train, predict
from app.main.util.authDto import AuthDto
from flask import request
from flask_restx import Resource

from app.main.util.decorator import admin_token_required, token_required
from app.main.util.superadminDto import SuperAdminDto
api = SuperAdminDto.api
_get_data = SuperAdminDto.get_data
_get_inputs = SuperAdminDto.get_inputs

@api.route('/get_data')
class LoadData(Resource):
    """
    User Login Resource
    """
    @api.doc('load data')
    @api.expect(_get_data, validate=True)
    def post(self):
        """Logs in existing users"""
        #get post data
        post_data = request.json
        return get_data_new(data=post_data)

@api.route('/train_model')
class TrainModel(Resource):
    """
    User Login Resource
    """

    @api.doc(security='apikey')
    @api.doc('Train_model')
    @token_required
    def get(self):
        """Logs in existing users"""
        #get post data
        return train()

@api.route('/predict')
class LoadData(Resource):
    """
    User Login Resource
    """

    @api.doc(security='apikey')
    @api.doc('predict')
    @api.expect(_get_inputs, validate=True)
    @token_required
    def post(self):
        """Logs in existing users"""
        #get post data
        post_data = request.json
        return predict(data=post_data)