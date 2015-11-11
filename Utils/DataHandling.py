import cPickle
import json
import os
import pandas as pd


def store_data(save_name, object_to_save):

    with open(save_name, 'wb') as file_to_save:
        cPickle.dump(object_to_save, file_to_save)
        print 'Saved object ' + save_name


def store_json(data, file_location):

    with open(file_location, 'w') as file_to_save:
        file_to_save.write(json.dumps(data, indent=4))


def load_json(file_location):
    with open(file_location, 'r') as file_to_load:
        text = file_to_load.read()
        return json.loads(text)


def load_data(load_name):

    if os.path.exists(load_name):
        with open(load_name, 'rb') as file_to_load:
            object_to_load = cPickle.load(file_to_load)
            print 'Loaded object ' + load_name
        return object_to_load
    else:
        print "Error loading file: " + load_name
        return None


def store_data_frame(save_name, df_to_save):

    df_to_save.to_pickle(save_name)
    print 'Saved dataframe.'


def load_data_frame(load_name):

    print 'Loading dataframe ' + load_name
    return pd.read_pickle(load_name)


