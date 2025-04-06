from flask import Flask, request, abort
from flask_cors import CORS
import flask_login
from RoutePlanner.default_router import PoliceRouter, ORS_PROFILES, DEFAULT_ORS_PROFILE
from markupsafe import escape
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from flasgger import Swagger, swag_from
import os
from database.models import Manager
from werkzeug.security import check_password_hash as check_pw
import datetime
import jwt

TOKEN_EXPIRE_MINUTES = 30
ALGORITHM = "HS256"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportException:
    pass

__version__ = "0.0.2"

app = Flask(__name__)
CORS(app)
api = Api(app)
app.config['SWAGGER'] = {
    'title': 'Patrol Routes API',
    'uiversion': 3,
}
swagger = Swagger(app)

app.secret_key = os.getenv('APP_SECRET_KEY')
if len(app.secret_key) == 0:
    raise EnvironmentError("No APP_SECRET_KEY provided.")
login_manager = flask_login.LoginManager()
login_manager.init_app(app)

@login_manager.request_loader
def request_loader(request):
    try:
        payload = jwt.decode(request.authorization.token,
                             app.secret_key,
                             algorithms=[ALGORITHM])
        return Manager.get(Manager.cedula == payload.get("sub"))
    except:
        return None

class RouteSuggestions(Resource):
    route_computer = PoliceRouter()
    parser: RequestParser
    def __init__(self):
        self.parser = RequestParser()
        self.parser.add_argument('cai',
                                 required=True,
                                 type=int,
                                 location='args')
        self.parser.add_argument('n',
                                 type=int,
                                 location='args')
        self.parser.add_argument('profile',
                                 type=str,
                                 choices=ORS_PROFILES,
                                 location='args')
        self.parser.add_argument('exclude_station',
                                 type=bool,
                                 location='args')
        self.parser.add_argument('threshold',
                                 type=float,
                                 location='args')
    
    @swag_from("doc/RouteSuggestions_get.yml")
    def get(self):
        args = self.parser.parse_args()
        # Todo: use real values here
        return self.route_computer.compute_routes(
            args['cai'],
            args['n'] if args['n'] is not None else 1,
            args['profile'] if args['profile'] is not None
                else DEFAULT_ORS_PROFILE,
            args['exclude_station'] is None,
            args['threshold'] if args['threshold'] is not None
                else 0.0
        )

class Route(Resource):
    def get(self, date, cai_id, assigned_to):
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            abort(400)
        return {
            'date': date.strftime('%Y/%m/%d'),
            'cai_id': cai_id,
            'assigned_to': assigned_to,
        }

    @flask_login.login_required
    def put(self, date, cai_id, assigned_to):
        return flask_login.current_user.cedula


@app.post('/api/admin/login')
def admin_login():
    # Use 'username' here for Oauth2 compliance
    cedula = request.form['username']
    password = request.form['password']

    try:
        manager = Manager.get(Manager.cedula == cedula)
    except:
        return {"error": "Wrong credentials"}, 401
    if check_pw(manager.password_hash, password):
        expires_at = datetime.datetime.now(datetime.timezone.utc) \
            + datetime.timedelta(minutes=TOKEN_EXPIRE_MINUTES)
        token = jwt.encode({"sub": manager.cedula, "exp": expires_at},
                           app.secret_key,
                           algorithm=ALGORITHM)
        return {
            'access_token': token,
            'token_type': "bearer"
        }
    else:
        return {"error": "Wrong credentials"}, 401

api.add_resource(Route,
    '/api/routes/<string:date>/<int:cai_id>/<int:assigned_to>')
api.add_resource(RouteSuggestions, '/api/routes')
if __name__ == '__main__':
    app.run(debug=True)
