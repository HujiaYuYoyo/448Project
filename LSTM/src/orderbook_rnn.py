from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
import tensorflow as tf

class OrderBookRNN:
    '''
    Inputs:
        - timesteps: number of time sequence inputs
        - layer_neurons: number of neurons in each LSTM layer
        - input_shape: shape of input
        - output_shape: shape of output (e.g. num classes)
        - num_hidden_layers: number of 'vertical' hidden LSTM layers
        - dropout: dropout rate
    '''
    def __init__(self, timesteps, layer_neurons, input_shape, output_shape, num_hidden_layers, response_type, dropout = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timesteps = timesteps
        self.layer_neurons = layer_neurons
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.response_type = response_type
        self.model = self.createRNN()

    def createRNN(self):
        tf.reset_default_graph()
        if self.response_type.upper() == 'CLASSIFICATION':
            print('Building classification model...')
            model = Sequential()
            model.add(LSTM(self.layer_neurons, input_shape=self.input_shape, return_sequences=True))
            for i in range(self.num_hidden_layers):
                if i == self.num_hidden_layers-1:
                    model.add(LSTM(self.layer_neurons, return_sequences=True)) # False?
                else:
                    model.add(LSTM(self.layer_neurons, return_sequences=True))

            if self.dropout is not None:
                model.add(Dropout(self.dropout))
            model.add(Flatten())
            model.add(Dense(self.output_shape, activation='softmax'))

            print('Compiling model...')
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

        elif self.response_type.upper() == 'REGRESSION':
            print('Building regression model...')
            model = Sequential()
            model.add(LSTM(self.layer_neurons, input_shape=self.input_shape, return_sequences=True))
            for i in range(self.num_hidden_layers):
                if i == self.num_hidden_layers-1:
                    model.add(LSTM(self.layer_neurons, return_sequences=True)) # False?
                else:
                    model.add(LSTM(self.layer_neurons, return_sequences=True))

            if self.dropout is not None:
                model.add(Dropout(self.dropout))

            model.add(Flatten())
            model.add(Dense(1))

            print('Compiling model...')
            model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def get_model(self):
        return self.model
