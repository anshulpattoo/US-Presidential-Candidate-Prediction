 from sklearn.neural_network import MLPClassifier
 
  def train(self, inputPatterns, outputs):
    #Initializing the MLPClassifier. 
    self.classifier = MLPClassifier(hidden_layer_sizes=(100, ), max_iter = 50, momentum = 5,
    learning_rate_init = 1, activation = 'relu', solver = 'adam', random_state = 1)

    #Fitting the training data to the network
    self.classifier.fit(inputs, outputs)
    