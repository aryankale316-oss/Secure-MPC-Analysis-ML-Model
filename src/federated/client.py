from src.model.train import train_model

class Client:

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None

    def train(self):

        self.model = train_model(self.data_path)

        return self.model
