from flask import Flask, request, abort
from flask_cors import CORS
import flask_login
from RoutePlanner.default_router import PoliceRouter, ORS_PROFILES, DEFAULT_ORS_PROFILE
from RoutePlanner.prediction.simple_wrapper import SimpleModelWrapper
from markupsafe import escape
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from flasgger import Swagger, swag_from
import json
import os
from database.models import Manager, Route as DBRoute
from werkzeug.security import check_password_hash as check_pw
import datetime
import jwt
import yaml

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

with open('doc/global.yml') as stream:
    doc_template = yaml.safe_load(stream)

swagger = Swagger(app, template=doc_template)

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
    route_computer = PoliceRouter(
    os.getenv('ORS_KEY'),
    SimpleModelWrapper(
        model_data_path = "crime_model_simple.pkl",
    ))
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
        self.parser.add_argument('hotspots',
                                 type=bool,
                                 location='args')
        self.parser.add_argument('requested_spots',
                                 type=str,
                                 location='args')
    
    @swag_from("doc/RouteSuggestions_get.yml")
    def get(self):
        args = self.parser.parse_args()
        requested_spots = json.loads(args['requested_spots']) \
            if args['requested_spots'] is not None \
            else []
        return self.route_computer.compute_routes(
            args['cai'],
            args['n'] if args['n'] is not None else 1,
            args['profile'] if args['profile'] is not None
                else DEFAULT_ORS_PROFILE,
            args['exclude_station'] is None,
            args['threshold'] if args['threshold'] is not None
                else 0.0,
            args['hotspots'] is not None,
            requested_spots
        )

class Route(Resource):
    @staticmethod
    def _get_route(date, cai_id, assigned_to):
        return DBRoute\
            .select()\
            .where((DBRoute.date == date)
               & (DBRoute.cai_id == cai_id)
               & (DBRoute.assigned_to == assigned_to))\
            .first()
    
    @swag_from("doc/Route_get.yml")
    def get(self, date, cai_id, assigned_to):
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return {"error": "Value %s is not of the YYYY-MM-DD date format." % date}, 400
        route = self._get_route(date, cai_id, assigned_to)
        if route == None:
            return {"error": "Route not found"}, 404
        return route.geometry

    @flask_login.login_required
    @swag_from("doc/Route_put.yml")
    def put(self, date, cai_id, assigned_to):
        if cai_id != flask_login.current_user.cai_id:
            return {"error": "Current manager is unauthorized to assign routes for %d." % cai_id}, 401
        route = request.get_json()
        route_id = (DBRoute
                        .insert(
                            geometry=route,
                            date=date,
                            cai_id=cai_id,
                            assigned_to=assigned_to,
                            assigned_by=flask_login.current_user
                        )
                        .on_conflict(
                            conflict_target=[DBRoute.date,
                                             DBRoute.cai_id,
                                             DBRoute.assigned_to],
                            preserve=[DBRoute.geometry],
                        )
                        .execute())
        return {"info": "route stored successfully."}

class Routes(Resource):
    @staticmethod
    def _get_routes(date, cai_id):
        return DBRoute\
            .select()\
            .where((DBRoute.date == date)
               & (DBRoute.cai_id == cai_id))

    def get(self, date, cai_id):
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return {"error": "Value %s is not of the YYYY-MM-DD date format." % date}, 400
        routes = self._get_routes(date, cai_id)
        if len(routes) == 0:
            return {"error": "No routes found for given date."}, 404
        return [route.toDict() for route in routes]
        

@swag_from("doc/login.yml")
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
            'token_type': "bearer",
            'user': manager.toDict()
        }
    else:
        return {"error": "Wrong credentials"}, 401

api.add_resource(Route,
    '/api/routes/<string:date>/<int:cai_id>/<int:assigned_to>')
api.add_resource(Routes, '/api/routes/<string:date>/<int:cai_id>')
api.add_resource(RouteSuggestions, '/api/routes')
if __name__ == '__main__':
    app.run(debug=True)
