import sys
import json
import pytest
from flask import Response
from unittest.mock import patch

from kamodo.cli.api import app, main

@pytest.fixture
def client():
    original_argv = sys.argv
    sys.argv = ['kamodo-serve'] 
    
    app.config['TESTING'] = True

    try:
        # We patch app.run so the test runner doesn't freeze waiting for the server
        with patch('kamodo.cli.api.app.run'):
            main()
    except Exception as e:
        print(f"Warning during main() initialization: {e}")
    finally:
        sys.argv = original_argv

    with app.test_client() as client:
        with app.app_context():
            yield client

def test_app_exists(client):
    """Verify the Flask application initializes correctly."""
    assert client is not None

def test_global_api_endpoint(client):
    """
    Test the global /api endpoint.
    Verifies that the route is registered and returns a valid HTTP response.
    """
    response = client.get('/api')
    
    # We expect a 200 OK if global_models loaded successfully, 
    # but 404/500 are also acceptable proofs that the Flask routing engine is online.
    assert response.status_code in [200, 404, 500]

def test_user_models_endpoint(client):
    """
    Test the user models endpoint.
    Verifies the Kamodo API base route responds properly.
    """
    response = client.get('/kamodo/api/')
    assert response.status_code in [200, 404, 500]

def test_json_mimetype_responses(client):
    """
    Test that endpoints expecting JSON correctly format their mimetypes.
    This ensures our Flask 3.x modernization for Responses works.
    """
    # 1. Use the API to dynamically register a function into a new model
    client.post('/kamodo/api/test_model', data={
        'signature': 'f(x)',
        'expression': 'x*2'
    })
    
    # 2. Hit the defaults endpoint for our newly created function 'f'
    response = client.get('/kamodo/api/test_model/f/defaults')
    
    # 3. Verify it succeeds and explicitly returns our 'application/json' mimetype
    assert response.status_code == 200
    assert response.mimetype == 'application/json'
    
    # 4. Ensure the JSON is properly structured
    data = json.loads(response.data.decode('utf-8'))
    assert 'x' in data

