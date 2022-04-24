import torch

class svm:
    @classmethod
    def __init__(self, trade_off: int = 100):
        self.w = None
        self.b = None
        self.trade_off = torch.tensor(trade_off)
        self.learning_rate = None
        self.best_w = None
        self.best_b = None
        self.best_loss = torch.tensor(1e9)
        

    @staticmethod
    def shuffle(train: torch.tensor, labels: torch.tensor):
        """
        Shuffles the data with a random permutation
        """
        n = train.shape[0]
        permutation = torch.randperm(n)
        train = train[permutation]
        labels = labels[permutation]
        return train, labels
    
    @staticmethod
    def iterate_minibatches(train : torch.tensor, labels: torch.tensor, batch_size : int):
        """
        returns a generator object for mini batches
        """
        for start_idx in range(0, train.shape[0] + 1 - batch_size, batch_size):
            excerpt = slice(start_idx, start_idx+batch_size)
            yield train[excerpt], labels[excerpt]

    @classmethod
    def loss(self, train: torch.tensor, labels: torch.tensor):
        """
        Calculates the loss
        """
        return self.w @ self.w + torch.mean(torch.maximum( torch.zeros(train.shape[0]) , torch.ones(train.shape[0]) - labels @ (train @ self.w + self.b)))

    
    @classmethod
    def fit_util(self, train: torch.tensor, labels: torch.tensor):
        """
        Utility function for fit
        """
        batch_size = train.shape[0]
        indices = (labels * (train @ self.w + self.b )) < 1.
        grad_b = -torch.sum(labels[indices])
        grad_w = 2 * self.w  - self.trade_off * (labels[indices] @ train[indices]) / batch_size
        self.b -=  self.learning_rate * grad_b
        self.w -= self.learning_rate * grad_w
        if self.best_loss > self.loss(train, labels):
            self.best_b = self.b
            self.best_w = self.w
            self.best_loss = self.loss(train, labels)
        

    @classmethod
    def fit(self, train: torch.tensor, labels: torch.tensor, learning_rate : int = 0.1 ,epochs : int = 10, batch_size = 4):
        """
        Trains the weights
        """
        self.learning_rate = torch.tensor(learning_rate)
        s = train.shape[1]
        self.w = torch.normal(torch.zeros(s)).to(torch.float64)
        self.b = torch.tensor(0.).to(torch.float64)
        

        for epoch in range(epochs):
            self.learning_rate = 1/(epoch+1)
            train, labels = self.shuffle(train, labels)
            for x,y in self.iterate_minibatches(train, labels, batch_size):
                self.fit_util(x, y)
        self.b = self.best_b
        self.w = self.best_w

    @classmethod
    def predict(self, test: torch.tensor):
        """
        returns the predicted label
        """
        logits = test @ self.w + self.b
        return torch.sign(logits)
    
    @classmethod
    def accuracy(self, test : torch.tensor, ground_truth: torch.tensor):
        """
        Returns the accuracy of the model given
        test datapoints, and ground truth for those
        datapoints
        """
        predicted = self.predict(test)
        return torch.mean((predicted == ground_truth).float())


