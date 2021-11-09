# app/__init__.py

from flask_restx import Api
from flask import Blueprint

from app.main.controller.auth_controller import api as auth_ns
from app.main.controller.superadmin.superAdminController import api as super_admin_ns
from app.main.config import authorizations, version
from app.main.util.decorator import token_required




blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='pybron BE APIs',
          version='1.0',
          description='Api Links are provided down below',
          authorizations=authorizations,
          )

api.add_namespace(auth_ns, path=version('auth'))
api.add_namespace(super_admin_ns, path='/load_data')
# api.add_namespace(path="/")



