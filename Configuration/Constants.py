import os


class Constants():
    def __init__(self):
        pass

    cwd = os.getcwd()

    main_dir = cwd + os.sep + '..' + os.sep + '..' + os.sep + '..' + os.sep

    data_path = cwd + os.sep + 'data' + os.sep

    model_path = cwd + os.sep + 'models' + os.sep