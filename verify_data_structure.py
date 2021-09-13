import yaml
from collections import OrderedDict

# output OrderedDict in right order


def order_rep(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', list(data.items()), flow_style=False)


yaml.add_representer(OrderedDict, order_rep)


def preyaml_index_from_df(df):
    return [str(_) for _ in df.index.names]


def preyaml_columns_from_df(df):
    return {str(_): str(df[_].dtypes) for _ in df.columns.tolist()}


def preyaml_from_df(df, name):
    out = OrderedDict()
    out["name"] = name
    out["indexs"] = preyaml_index_from_df(df)
    out["columns"] = preyaml_columns_from_df(df)
    return out


def preyaml_from_store(store):
    return (preyaml_from_df(store[k], k[1:]) for k in sorted(store.keys()))


def yaml_from_store(store, **kwargs):
    return yaml.dump_all(preyaml_from_store(store), default_flow_style=False, **kwargs)
