# import required packages
from flask import Flask, render_template, request, url_for
from gevent.pywsgi import WSGIServer

# create a web app
numanager = Flask(__name__)

@numanager.route('/', methods=['POST', 'GET'])
def home():
    context = {}
    if request.method == 'POST':
        cb_tasks = request.form.getlist('tasks')
        print('Tasks: ', cb_tasks)
        cbb_model = request.form.get('model')
        print('Model: ', cbb_model)
    
        context = {
            'train': True if 'train' in cb_tasks else False,
            'simulate': True if 'simulate' in cb_tasks else False,
            'open_nuboard': True if 'open_nuboard' in cb_tasks else False,
            'model': cbb_model,
        }
    print(context)
    return render_template('home.html', **context)


if __name__ == '__main__':
    # Debug/Development
    numanager.run(debug=True, host="localhost", port=4006)
    # Production
    # http_server = WSGIServer(('', 4006), numanager)
    # http_server.serve_forever()
