from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
X = [[0., 0., 0.0], [1., 1., 1], [2, 2, 2]]
y = [0, 1, 2]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
reg = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
clf.fit(X, y)
reg.fit(X, y)
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
print(reg.predict([[0.0, 0.0, 0.0], [-1., -2., 0.0], [1, 2, 0.0]]))
print(clf.predict([[0.0, 0.0, 0.0], [-1., -2., 0.0], [1, 2, 0.0]]))