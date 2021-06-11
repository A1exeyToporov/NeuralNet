#Fully Connected Neural Network

1) Import package <b>NeuralNet</b>
2) Define class <b>NeuralNetwork( )</b> with input size equals number of features
3) Stack some layers with method <b>.add_layer(units, activation)</b>
4) Use method fit to train your model <b>.fit(X_train, y_train, num_iterations, learning_rate, loss)</b> 
5) Predict test data with <b>.predict(X_test)</b> and get accuracy using <b>.accuracy(Y_true, Y_predict) if you train for binary classification</b>

##Some tips and issues

1) <b>Sigmoid</b> and <b>ReLu</b> activations is only available
2) Better use learning rate less than 0.01, because you can get vanishing gradients issue
3) Only has <b>binary_loss</b> for binary classification and <b>MSE</b> for regression tasks
