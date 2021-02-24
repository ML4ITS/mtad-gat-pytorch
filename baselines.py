from typing import Tuple
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, LSTM, Dense, Flatten, ReLU, LeakyReLU


def create_cnn(input_shape: Tuple[int, ...],
			   activation: str = 'relu',
			   kernel_size: int = 7,
			   dropout: float = 0.2,
			   filters: int = 32,
			   out_dim: int = 8
			   ) -> keras.Model:

	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dropout(dropout))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(out_dim))

	return model


def create_lstm(input_shape: Tuple[int, ...],
				units: int = 64,
				activation: str = 'relu',
				dropout: float = 0.2,
				out_dim: int = 8
				) -> keras.Model:

	model = Sequential()
	model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
	model.add(LeakyReLU(0.5))
	model.add(LSTM(units, return_sequences=True))
	model.add(LeakyReLU(0.5))
	model.add(Dropout(dropout))
	model.add(LSTM(units//2, return_sequences=False))
	model.add(Dense(out_dim))

	return model


