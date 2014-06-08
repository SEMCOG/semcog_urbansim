from bottle import route, response, run, hook, static_file
from urbansim.utils import yamlio
import simplejson
from jinja2 import Environment


@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = \
        'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

VIEWS = {}
DSET = None


def get_schema():
    global VIEWS
    return {name: list(VIEWS[name].columns) for name in VIEWS}

@route('/map_query/<table>/<filter>/<groupby>/<field>/<agg>', method="GET")
def map_query(table, filter, groupby, field, agg):
    global VIEWS, DSET
    # if table not in views:
    #    views[table] = DSET.view(table).build_df()
    cmd = "VIEWS['%s'].groupby('%s')['%s'].%s()" % \
          (table, groupby, field, agg)
    print cmd
    results = eval(cmd)
    results.index = DSET.zones.taz.loc[results.index]
    results = yamlio.series_to_yaml_safe(results.dropna())
    #print results
    return results

@route('/map_query/<table>/<filter>/<groupby>/<field>/<agg>', method="OPTIONS")
def ans_options(table, filter, groupby, field, agg):
    return {}


@route('/')
def index():
    index = open('index.html').read()
    config = {
        'center': str([42.322, -83.176]),
        'zoom': 10,
        'shape_json': 'data/tazs.json',
        'schema': simplejson.dumps(get_schema())
    }
    return Environment().from_string(index).render(config)


@route('/data/<filename>')
def data_static(filename):
    return static_file(filename, root='./data')


def start(dset, views, port=8765, host='localhost'):
    global VIEWS, DSET
    VIEWS = {str(k): views[k] for k in views}
    DSET = dset
    run(host=host, port=port, debug=True)


