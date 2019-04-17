from sklearn.metrics import accuracy_score


class Metrics:
    @classmethod
    def error(cls, data, output):
        return 1 - accuracy_score(data.data_y, output)
