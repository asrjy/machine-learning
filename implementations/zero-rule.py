# Returns the most likely value based on count/frequency
from random import seed

def zero_rule_classification(train, test):
    output_vals = [instance[-1] for instance in train]
    prediction = max(set(output_vals), key=output_vals.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

def zero_rule_regression(train, test):
    output_vals = [instance[-1] for instance in train]
    mean = sum(output_vals)/float(len(output_vals))
    predictions = [mean for i in range(len(test))]
    return predictions

def moving_average_zero_rule_regression(train, test, n):
    output_vals_window = [instance[-1] for instance in train[-n:]]
    mean = sum(output_vals_window)/float(n)
    predictions = [mean for i in range(len(test))]

seed(69)

train = [['0'], ['1'], ['1'], ['0'], ['1'], ['1']]
test = [[None], [None], [None], [None]]

predictions = zero_rule_classification(train, test)
print(predictions)

train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]

predictions = zero_rule_regression(train, test)
print(predictions)