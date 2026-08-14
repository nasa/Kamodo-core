import os
from os import path

import numpy as np
import pandas as pd
from kamodo import kamodofy, Kamodo
from sympy.core.function import UndefinedFunction
from kamodo import get_defaults, getfullargspec
from kamodo import serialize, deserialize

import flask
from flask import Flask, render_template, jsonify
from flask_cors import CORS, cross_origin
from flask_restful import reqparse, abort, Api, Resource
from flask import request

from io import StringIO
import json

import yaml
import importlib

import plotly.graph_objects as go

def instantiate(conf):
    """Minimal native replacement for hydra.utils.instantiate"""
    if not isinstance(conf, dict) or '_target_' not in conf:
        return conf
    conf_copy = dict(conf)
    target = conf_copy.pop('_target_')
    module_name, class_name = target.rsplit('.', 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    return obj(**conf_copy)

from collections import defaultdict
from kamodo import from_kamodo

import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

app = Flask(__name__)

CORS(app) # enable Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.


def get_api(app):
    """get restful api for this app"""
    api = Api(app)
    return api

# models registered at startup
global_models = {}

# models registered by users, providing access to global models
user_models = {}

def main():
    """main entrypoint"""

    # 1. Load native YAML config instead of Hydra
    conf_path = 'conf/kamodo.yaml'
    if not path.exists(conf_path):
        print(f"Warning: Configuration file {conf_path} not found.")
        cfg = {'flask': {'host': '0.0.0.0', 'port': 5000}, 'models': {}}
    else:
        with open(conf_path, 'r') as f:
            cfg = yaml.safe_load(f)

    # 2. Handle override
    config_override_path = cfg.get('config_override')
    if config_override_path is not None:
        override_path = f"{os.getcwd()}/{config_override_path}"
        if path.exists(override_path):
            app.logger.info(f"found {override_path}")
            with open(override_path, 'r') as f:
                config_override_ = yaml.safe_load(f)
                cfg.update(config_override_)
                # Ensure models are replaced if specified
                if 'models' in config_override_:
                    cfg['models'] = config_override_['models']
        else:
            if cfg.get('verbose', 0) > 0:
                app.logger.info(f"could not get override: {override_path}")
                print(f"could not get override: {override_path}")

    if cfg.get('verbose', 0) > 0:
        app.logger.info(json.dumps(cfg, indent=2))

    # 3. Instantiate Global Models
    for model_name, model_conf in cfg.get('models', {}).items():
        try:
            model_ = instantiate(model_conf)
            global_models[model_name] = model_
        except Exception as m:
            print(m)
            if cfg.get('verbose', 0) > 0:
                print(f'could not start {model_name}')

    api = Api(app)

    api.add_resource(get_global_models_resource(), '/api', '/api/')

    for model_name, model_ in global_models.items():
        register_endpoints(api, model_name, model_, '/api')

    global user_models
    user_model_default = instantiate(cfg.get('user_model', {}))
    
    # Ensure it falls back to a Kamodo object if the yaml config was empty
    if not isinstance(user_model_default, Kamodo):
        user_model_default = Kamodo()
        
    user_models = defaultdict(lambda: from_kamodo(user_model_default))

    api.add_resource(get_user_models_resource(), '/kamodo/api', '/kamodo/api/')

    @app.route('/kamodo/api/<user_model_name>', methods=['GET', 'POST'])
    @app.route('/kamodo/api/<user_model_name>/', methods=['GET', 'POST'])
    @app.route('/kamodo/api/<user_model_name>/<user_func>', methods=['GET', 'POST', "DELETE"])
    def kamodo_model_func(user_model_name, user_func=None):
        user_model = user_models[user_model_name]
        print(user_model_name, user_func)
        print(user_model)

        if request.method == 'POST':
            func_key = request.form['signature']
            if 'expression' in request.form:
                func_expr = request.form['expression']
                if func_key in user_model:
                    del user_model[func_key]
                user_model[func_key] = func_expr
            elif 'model' in request.form:
                global_model_name = request.form['model']
                global_model = global_models[global_model_name]
                if 'model_func' in request.form:
                    global_model_func_name = request.form['model_func']
                    global_model_func = global_model[global_model_func_name]
                    user_model[func_key] = global_model_func

        if user_func is not None:
            parser = reqparse.RequestParser()
            if request.method == 'GET':
                func = user_model[user_func]
                defaults = get_defaults(func)
                for arg in getfullargspec(func).args:
                    if arg in defaults:
                        parser.add_argument(arg, type=str)
                    else:
                        parser.add_argument(arg, type=str, required=True)

                args_ = parser.parse_args(strict=True)
                args = dict()

                for argname, val_ in args_.items():
                    args[argname] = json.loads(val_, object_hook=deserialize)
                    if isinstance(args[argname], str):
                        args[argname] = json.loads(val_, object_hook=deserialize)
                    if isinstance(args[argname], list):
                        args[argname] = np.array(args[argname])
                print('{} {} passed {}'.format(
                    model_name,
                    user_func,
                    ' '.join(['{}:{}'.format(arg, type(arg_value)) for arg, arg_value in args.items()])))
                try:
                    result = func(**args)
                    print('{} {} function returned {}'.format(
                        model_name, user_func, type(result)))
                    return serialize(result)
                except:
                    return flask.Response("null", mimetype='application/json')
            if request.method == "DELETE":
                if user_func in user_model:
                    del user_model[user_func]

        return get_model_details(user_model)
        # return dict(user_func = user_func, user_model_name = user_model_name)

    @app.route('/kamodo/api/<user_model_name>/<user_func>/defaults', methods=['GET'])
    @app.route('/kamodo/api/<user_model_name>/<user_func>/defaults/', methods=['GET'])
    def kamodo_model_func_defaults(user_model_name, user_func):
        user_model = user_models[user_model_name]
        user_func = user_model[user_func]
        function_defaults = get_defaults(user_func)

        for arg in getfullargspec(user_func).args:
            if arg in function_defaults:
                pass
            else:
                function_defaults[arg] = None
        try:
            function_defaults_ = json.dumps(function_defaults, default=serialize)
        except:
            print('problem with {}.{}'.format(user_model_name, user_func))
            raise
        
        # Return exact JSON string with correct HTTP headers
        return flask.Response(function_defaults_, mimetype='application/json')

    @app.route('/kamodo/api/<user_model_name>/<user_func>/data', methods=['GET'])
    @app.route('/kamodo/api/<user_model_name>/<user_func>/data/', methods=['GET'])
    def kamodo_model_func_data(user_model_name, user_func):
        user_model = user_models[user_model_name]
        func = user_model[user_func]
        # assume function is kamodofied
        func_data_str = json.dumps(func.data, default=serialize)
        return flask.Response(func_data_str, mimetype='application/json')

    @app.route('/kamodo/api/<user_model_name>/<user_func>/plot', methods=['GET'])
    @app.route('/kamodo/api/<user_model_name>/<user_func>/plot/', methods=['GET'])
    def kamodo_model_func_plot(user_model_name, user_func):
        user_model = user_models[user_model_name]
        func = user_model[user_func]
        defaults = get_defaults(func)

        parser = reqparse.RequestParser()

        for arg in getfullargspec(func).args:
            if arg in defaults:
                parser.add_argument(arg, type=str)
            else:
                parser.add_argument(arg, type=str, required=True)

        app.logger.info('function plot resource called')
        args_ = parser.parse_args(strict=True)
        args = dict(indexing='ij') # needed for gridded data

        for argname, val_ in args_.items():
            if val_ is not None:
                args[argname] = pd.read_json(StringIO(val_), typ='series')
        figure = user_model.figure(variable=user_func, **args)
        
        # Plotly returns a JSON string, wrap it in a proper Response
        return flask.Response(go.Figure(figure).to_json(), mimetype='application/json')

    flask_cfg = cfg.get('flask', {})
    try:
        app.run(host=flask_cfg.get('host', '0.0.0.0'), port=flask_cfg.get('port', 5000))
    except OSError as m:
        print('cannot start with configuration', flask_cfg)
        raise


def get_model_details(model):
    detail = model.detail().astype(str)
    detail_dict = detail.to_dict(
                    # default_handler=str,
                    # indent =4,
                    orient='index',
                    )
    return detail_dict

def get_global_models_resource():
    '''registers signatures of multiple models'''
    class GlobalModels(Resource):
        def get(self):
            details = dict()
            for model_name, model_ in global_models.items():
                details[model_name] = get_model_details(model_)
            return details
    return GlobalModels

def get_user_models_resource():
    '''registers signatures of multiple models'''
    class UserModels(Resource):
        def get(self):
            details = dict()
            for model_name, model_ in user_models.items():
                details[model_name] = get_model_details(model_)
            return details
    return UserModels

def register_func_endpoints(api, model_name, model_, base_name, var_symbol):
    var_label = str(type(var_symbol))

    # /api/mymodel/myfunc
    func_resource_endpoint = '{}/{}/{}'.format(base_name, model_name, var_label)
    print('registering ' + func_resource_endpoint)
    try:
        api.add_resource(
            get_func_resource(model_name, model_, var_symbol),
            func_resource_endpoint,
            endpoint='/'.join([model_name, var_label]))

        # /api/mymodel/myfunc/defaults
        defaults_resource_endpoint = '{}/{}/{}/{}'.format(
            base_name, model_name, var_label, 'defaults')
        print('registering ' + defaults_resource_endpoint)
        api.add_resource(
            get_defaults_resource(model_name, model_, var_symbol),
            defaults_resource_endpoint,
            endpoint='/'.join([model_name, var_label, 'defaults']))

        # /api/mymodel/myfunc/data
        data_resource_endpoint = '{}/{}/{}/{}'.format(
            base_name, model_name, var_label, 'data')
        print('registering ' + data_resource_endpoint)
        api.add_resource(
            get_data_resource(model_name, model_, var_symbol),
            data_resource_endpoint,
            endpoint='/'.join([model_name, var_label, 'data']))

        # /api/mymodel/myfunc/plot
        func_plot_resource_endpoint = '{}/{}/{}/{}'.format(
            base_name, model_name, var_label, 'plot')
        print('registering ' + func_plot_resource_endpoint)
        api.add_resource(
            get_func_plot_resource(model_, var_symbol),
            func_plot_resource_endpoint,
            endpoint='/'.join([model_name, var_label, 'plot']))

        # /api/mymodel/myfunc/doc
        func_doc_resource_endpoint = '{}/{}/{}/{}'.format(
            base_name, model_name, var_label, 'doc')
        print('registering ' + func_doc_resource_endpoint)
        api.add_resource(
            get_func_doc_resource(model_, var_symbol),
            func_doc_resource_endpoint,
            endpoint='/'.join([model_name, var_label, 'doc']))

    except ValueError as m:
        print('warning: {}'.format(m))
        pass



def register_endpoints(api, model_name, model_, base_name, register_base=True):
    '''register all endpoints for this model'''
    print('registering endpoints for {}/{}'.format(base_name, model_name))
    logging.info('registering endpoints for {}/{}'.format(base_name, model_name))

    if register_base:
        model_resource_endpoint = '{}/{}'.format(base_name, model_name)
        print('registering ' + model_resource_endpoint)
        api.add_resource(
            get_model_resource(model_name, model_),
            model_resource_endpoint,
            model_resource_endpoint + '/',
            endpoint=model_name)

    for var_symbol in model_:
        if type(var_symbol) != UndefinedFunction:
            print('about to register {}'.format(var_symbol))
            register_func_endpoints(api, model_name, model_, base_name, var_symbol)

    # /api/mymodel/evaluate
    evaluate_resource_endpoint = '{}/{}/evaluate'.format(base_name, model_name)
    print('registering ' + evaluate_resource_endpoint)
    api.add_resource(
        get_evaluate_resource(model_name, model_),
        evaluate_resource_endpoint,
        endpoint='/'.join([model_name, 'evaluate'])
        )

    # /api/mymodel/doc
    model_doc_resource_endpoint = '{}/{}/doc'.format(base_name, model_name)
    print('registering ' + model_doc_resource_endpoint)
    api.add_resource(
        get_model_doc_resource(model_name, model_),
        model_doc_resource_endpoint,
        endpoint='/'.join([model_name, 'doc'])
        )

def get_model_doc_resource(model_name, model):
    '''retrieve documentation string for the model'''
    class model_doc_resource(Resource):
        def get(self):
            """get method for the doc string"""
            return model.__doc__
    return model_doc_resource

def get_model_resource(model_name, model):
    """Retrieve resource asscociated with this model"""

    class model_resource(Resource):
        """Resource for this model"""
        def get(self):
            """Get method for this model"""
            model_detail_dict = model.detail().astype(str).to_dict(orient='index')
            return model_detail_dict

    return model_resource

def get_func_resource(model_name, model, var_symbol):
    """Get resource associated with this function"""
    parser = reqparse.RequestParser()
    func = model[var_symbol]
    defaults = get_defaults(func)

    for arg in getfullargspec(func).args:
        if arg in defaults:
            parser.add_argument(arg, type=str)
        else:
            parser.add_argument(arg, type=str, required=True)

    class FuncResource(Resource):
        """Resource associated with this function"""

        def get(self):
            """get method for this resource"""
            args_ = parser.parse_args(strict=True)
            args = dict()

            for argname, val_ in args_.items():
                args[argname] = json.loads(val_, object_hook=deserialize)
                if isinstance(args[argname], str):
                    args[argname] = json.loads(val_, object_hook=deserialize)
                if isinstance(args[argname], list):
                    args[argname] = np.array(args[argname])
            print('{} {} passed {}'.format(
                model_name,
                var_symbol,
                ' '.join(['{}:{}'.format(arg, type(arg_value)) for arg, arg_value in args.items()])))
            result = func(**args)
            print('{} {} function returned {}'.format(
                model_name, var_symbol, type(result)))
            return serialize(result)

            # try:
            #     return result.tolist()
            # except:
            #     return result
    return FuncResource


def get_defaults_resource(model_name, model, var_symbol):
    """Get resource associated with this function's defaults"""

    # These defaults may change if the user redefines a variables
    # Therefore, the defaults need to be checked every time there's a get request

    class DefaultsResource(Resource):
        """Resource associated with this function's defaults"""
        def get(self):
            """get method for this resource"""
            app.logger.info('getting defaults for {}.{}'.format(model_name, var_symbol))
            func = model[var_symbol]
            function_defaults = get_defaults(func)
            try:
                function_defaults_ = json.dumps(function_defaults, default=serialize)
            except:
                print('problem with {}.{}'.format(model_name, var_symbol))
                raise
            return function_defaults_

    return DefaultsResource

def get_data_resource(model_name, model, var_symbol):
    """Get resource associated with this function's data"""

    # The data may change if the user redefines a variable
    # Data needs to come from the current state of the model

    class DefaultsResource(Resource):
        """Resource associated with this function's defaults"""
        def get(self):
            """get method for this resource"""
            print('getting data for {}.{}'.format(model_name, var_symbol))
            func = model[var_symbol]

            # assume function is kamodofied
            try:
                func_data = func.data
            except AttributeError:
                raise AttributeError('function {}.{} has no data!'.format(
                    model_name, var_symbol))

            try:
                func_data_ = json.dumps(func_data, default=serialize)
            except:
                print('problem with {}.{} data'.format(model_name, var_symbol))
                raise
            return func_data_

    return DefaultsResource


def get_evaluate_resource(model_name, model):
    """get resource associated with evaluate"""
    # not sure what to do if the model changes
    # get doesn't know what variables exist in the model
    # maybe move to route/<model_name>

    parser = reqparse.RequestParser()
    parser.add_argument('variable', type=str, required=True)

    for var_symbol in model:
        if type(var_symbol) != UndefinedFunction:
            func = model[var_symbol]
            for arg in getfullargspec(func).args:
                parser.add_argument(arg, type=str)


    class EvaluateResource(Resource):
        """Resource associated with evaluate"""

        def get(self):
            """get method for this resource"""
            args_ = parser.parse_args()
            args = dict()

            variable_name = ''

            for argname, val_ in args_.items():
                if argname != 'variable':
                    if val_ is not None:
                        args[argname] = pd.read_json(StringIO(val_), typ='series')

                else:
                    variable_name = val_

            try:
                result = model.evaluate(variable=variable_name, **args)
            except SyntaxError as m:
                return {'message': '{}'.format(m)}

            return {k_: v_.tolist() for k_, v_ in result.items()}

    return EvaluateResource


def get_func_plot_resource(model, var_symbol):
    """Get resource associated with this function's plot"""

    parser = reqparse.RequestParser()
    func = model[var_symbol]
    defaults = get_defaults(func)

    for arg in getfullargspec(func).args:
        if arg in defaults:
            parser.add_argument(arg, type=str)
        else:
            parser.add_argument(arg, type=str, required=True)


    class FuncPlotResource(Resource):
        """Resource associated with evaluate"""
        def get(self):
            """get method for this resource"""
            app.logger.info('function plot resource called')
            args_ = parser.parse_args(strict=True)
            args = dict(indexing='ij') # needed for gridded data

            for argname, val_ in args_.items():
                if val_ is not None:
                    args[argname] = pd.read_json(StringIO(val_), typ='series')
            figure = model.figure(variable=str(type(var_symbol)), **args)
            return go.Figure(figure).to_json()

    return FuncPlotResource

def get_func_doc_resource(model, var_symbol):
    """Get resource associated with this function's documentation"""
    class FuncDocResource(Resource):
        """Resource associated with doc"""
        def get(self):
            """get method for this resource"""
            return model[var_symbol].__doc__
    return FuncDocResource

if __name__ == '__main__':
    main()
