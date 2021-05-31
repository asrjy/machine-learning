# Algorithm needs to predict a random outcome as observed in the training data
# Dumbest Prediction ALgorithm Ever

from random import seed
from random import randrange

def random_algorithm(train, test):
    train_output_vals = [instance[-1] for instance in train]
    unique_outputs = list(set(train_output_vals))
    predicted = []
    for row in test:
        index = randrange(len(unique_outputs))
        predicted.append(unique_outputs[index])
    return predicted

seed(69)

train = [[0], [1], [0], [1], [0], [1]]
test = [[None], [None], [None], [None]]

preds = random_algorithm(train, test)
print(preds)
