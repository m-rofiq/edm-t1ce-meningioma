class EarlyStopping:

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.stop_epoch = None

    def step(self, score, epoch):

        if self.best_score is None:
            self.best_score = score
            return

        if score <= self.best_score:
            self.counter += 1

            if self.counter >= self.patience:
                self.stop = True
                self.stop_epoch = epoch

        else:
            self.best_score = score
            self.counter = 0