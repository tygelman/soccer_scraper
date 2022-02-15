import flask
from flask import render_template, url_for
from flask import Response

import io
import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_svg import FigureCanvasSVG as FigureCanvas

from config import Config
from models import model
from forms import QueryForm, ACTForm
from graphics import gen_fig, cluster_map
from utils import numbers
from utils.geocoding import (geocode)
from pymongo import MongoClient
import pickle
import pandas as pd
import os

from tqdm import tqdm

app = flask.Flask(__name__)
app.config.from_object(Config)
app.config['DEBUG'] = True


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Load stuff for ACT MODEL:
act_cluster_load = pickle_load('./models/act_model/data/act_clusters.pickle')
geo_cluster_load = pickle_load('./models/act_model/data/geo_clusters.pickle')
act_model_data = pd.read_csv('./models/act_model/data/act_model_data.csv')
ACT_MODEL = model.MultiBayesACT(act_model_data, geo_cluster_load, act_cluster_load)

MODEL = model.MultiBayes(data='./models/data.csv', name_matches='./models/name_matches.csv')
mongo_url = 'mongodb+srv://tygelman:testpass@cluster0.duycd.mongodb.net/soccer_scraper?retryWrites=true&w=majority'
client = MongoClient(mongo_url)


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    print(values)
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename.startswith('/'):
            filename = filename[1:]
        print(filename)
        if filename:
            file_path = os.path.join(endpoint, filename)
            print(file_path)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/plot-<string:college>-<string:position>.svg')
def plot_svg(college, position):
    fig = gen_fig(MODEL, college=college, position=position)
    output = io.BytesIO()
    FigureCanvas(fig).print_svg(output)
    return Response(output.getvalue(), mimetype='image/svg+xml')

@app.route('/cluster_map.svg')
def plot_cluster_map():
    fig = cluster_map(MODEL)
    output = io.BytesIO()
    FigureCanvas(fig).print_svg(output)
    return Response(output.getvalue(), mimetype='image/svg+xml')


@app.route('/', methods=['GET', 'POST'])
def home_page():
    return render_template('home.html')


@app.route('/relative-geo-prediction', methods=['GET', 'POST'])
def query():
    form = QueryForm()
    if form.validate_on_submit():
        pred, orig_admit_rate = MODEL.predict(form.college.data,
                                              {'Position': form.position.data, 'cluster': int(form.cluster.data)})
        pred = numbers.percent(pred, places=2, string_percent=True)
        orig_admit_rate = numbers.percent(orig_admit_rate, places=2, string_percent=True)
        if pred:
            return render_template('geoform.html', form=form, pred=pred, college=form.college.data,
                                   position=form.position.data, admit_rate=orig_admit_rate)
        else:
            df = MODEL.data
            valid = df[df.College == form.college.data]
            valid_clusters = [str(i) for i in sorted(valid.cluster.unique())]
            message = f'No players attended {form.college.data} from cluster {form.cluster.data}\nPlease select from the following clusters:\n{", ".join(valid_clusters)}.'
            return render_template('geoform.html', form=form, message=message)

    return render_template('geoform.html', form=form)


@app.route('/act_estimate', methods=['GET', 'POST'])
def act_estimate():
    form = ACTForm()
    if form.validate_on_submit():
        insert_reponse = client.soccer_scraper.soccer_act.insert_one(
            {'ACT Score': form.act_score.data, 'Position': form.position.data, 'Address': form.address.data})
        print(insert_reponse)

        geocode_result_dict = geocode(form.address)
        lat = geocode_result_dict['latitude']
        lon = geocode_result_dict['longitude']
        geo_cluster = ACT_MODEL.geo_clusters.predict([[lat, lon]])[0]

        act_cluster = ACT_MODEL.act_clusters.predict([[form.act_score.data]])[0]

        tokens = {
            'act_cluster': act_cluster,
            'geo_cluster': geo_cluster,
            'Class_Freshman': 1
        }

        position_map = {
            'Position_Defender': 0,
            'Position_Forward': 0,
            'Position_Goalkeeper': 0,
            'Position_Midfielder': 0
        }

        for k in position_map:
            if form.position.data.lower() in k.lower():
                update_map = position_map.copy()
                update_map[k] = 1
                tokens.update(update_map)

        print(tokens)

        preds_list = []
        for name in tqdm(ACT_MODEL.data.college_name.unique()):
            preds_list.append([name, ACT_MODEL.predict(name, tokens, relative=True)[0]])

        preds_df = pd.DataFrame.from_records(preds_list, columns=['college_name', 'relative_likelihood']).dropna()
        preds_table = preds_df.to_html()

        return render_template('actform.html', form=form, preds_table=preds_table)

    return render_template('actform.html', form=form)

app.run(host = "0.0.0.0", port = 8080)

