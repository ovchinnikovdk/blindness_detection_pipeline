

class LoaderFactory(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def create_loaders(self):
        raise NotImplementedError