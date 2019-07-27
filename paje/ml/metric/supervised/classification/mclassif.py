from sklearn.metrics import accuracy_score


class Metrics:
    @classmethod
    def error(cls, data):
        return 1 - accuracy_score(data.y, data.z)

    @classmethod
    def accuracy(cls, data):
        print(22222223333333333,(data.y, data.z), data.C)
        return accuracy_score(data.y, data.z)
