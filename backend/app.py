from flask import Flask, request
from RoutePlanner.default_router import PoliceRouter
from markupsafe import escape
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from flasgger import Swagger, swag_from

__version__ = "0.0.1"

app = Flask(__name__)
api = Api(app)
app.config['SWAGGER'] = {
    'title': 'Patrol Routes API',
    'uiversion': 3
}
swagger = Swagger(app)

route_computer = PoliceRouter()

class RouteSuggestions(Resource):
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

    @swag_from({
        'summary': 'Retornar una lista de hostspots y rutas '
            + 'sugeridas para un determinado cai.',
        'description': 'Este endpoint recibe el id de un'
            + 'determinado cai y una cantidad de rutas a '
            + 'computar. Devuelve esta cantidad de rutas optimizadas '
            + 'para cubrir los hotspots indicados, todas partiendo '
            + 'del cai designado. ',
        'parameters': [
            {
                'in': 'query',
                'name': 'cai',
                'schema': { 'type': 'integer' },
                'description': 'Id del cai del cual partir.',
                'required': True
            },
            {
                'in': 'query',
                'name': 'n',
                'schema': {'type': 'integer'},
                'description': 'Número de rutas a generar.',
                'required': False
            }
        ],
        'responses': {
            '200': {
                'description': 'OK',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'cai': {
                                    'type': 'array',
                                },
                                'hotspots': {
                                    'type': 'array',
                                },
                                'routes': {
                                    'type': 'array',
                                }
                            }
                        },
                        'examples': {
                            'example1': {
                                'summary': 'Successful response',
                                'value': {
                                    'cai': [23.4, -28.09],
                                    'hotspots': [
                                        {
                                            'coordinates': [56.8, -38.3],
                                            'probability': 0.34
                                        },
                                        {
                                            'coordinates': [36.9, -18.5],
                                            'probability': 0.9
                                        },
                                    ],
                                    'routes': [
                                        [
                                            [56.3, 83.4],
                                            [90.3, 32.2],
                                            [482.3, 484.3]
                                        ]
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    })
    def get(self):
        args = self.parser.parse_args()
        # Todo: use real values here
        return {
            'cai': [23.4, -28.09],
            'hotspots': [
                {
                    'coordinates': [56.8, -38.3],
                    'probability': 0.34
                },
                {
                    'coordinates': [36.9, -18.5],
                    'probability': 0.9
                },
            ],
            'routes': [
                [
                    [56.3, 83.4],
                    [90.3, 32.2],
                    [482.3, 484.3]
                ]
            ] * (args['n'] if args['n'] is not None else 1)
        };

api.add_resource(RouteSuggestions, '/api/routes')
if __name__ == '__main__':
    app.run(debug=True)
