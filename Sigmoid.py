import math

def sigmoid(input1 = 0.05, input2 = 0.10, weight1 = 0.15, weight2 = 0.20, bias = 0.35 ):

  y = weight1*input1 + weight2*input2 + bias
  print y
  return 1 / (1 + math.exp(-y))




print("sigmoid is", sigmoid(0.5932, 0.60288, 0.50, 0.55, 0.60))



#print (0.5*math.pow((0.7735-0.99), 2))