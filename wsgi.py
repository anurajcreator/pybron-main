import os
import unittest

from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

from app.main import create_app, db
from app.main.model import user, prediction
from app.main.model import blacklist

from app import blueprint

app = create_app('prod')
app.register_blueprint(blueprint)

app.app_context().push()

application=app
