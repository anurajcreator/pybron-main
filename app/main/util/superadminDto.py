from flask_restx import Namespace, fields

class SuperAdminDto:
    api = Namespace('superAdmin', description='SuperAdmin Related operation')
    get_data = api.model('get_data', {
        'input_data_path': fields.String(required=True,description='location of Input data')
    })
    get_inputs = api.model('get_input',{
        'fixed_acidity': fields.Float(required=True, description='location of Input data'),
        'volatile_acidity':fields.Float(required=True, description='location of Input data'),
        'citric_acid': fields.Float(required=True, description='location of Input data'),
        'residual_sugar': fields.Float(required=True, description='location of Input data'),
        'chlorides': fields.Float(required=True, description='location of Input data'),
        'free_sulfur_dioxide': fields.Integer(required=True, description='location of Input data'),
        'total_sulfur_dioxide': fields.Integer(required=True, description='location of Input data'),
        'density': fields.Float(required=True, description='location of Input data'),
        'pH': fields.Float(required=True, description='location of Input data'),
        'sulphates': fields.Float(required=True, description='location of Input data'),
        'alcohol': fields.Float(required=True, description='location of Input data'),
        })