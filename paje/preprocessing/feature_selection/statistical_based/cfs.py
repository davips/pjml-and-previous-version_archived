from paje.preprocessing.feature_selection.filter import Filter

@PluginBase.register
class FilterCFS(Filter):
    def __init__(self, ratio=0.8):
        self.ratio = ratio


    def fit(self, X, y):
        print("fit")


    def transform(self, X, y):
        print("transform")


    def rank(self):
        print("rank")


    # def rank_values(self):
        # pass


f = FilterCFS()
f.fit([], [])
f.transform([], [])
f.rank_values()
