import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, LeakyReLU, Dropout

# Import Data
# ------------------------------------------------------------------
df = pd.read_csv('data\daily.csv')
print(df.head())

price = df['Close']
plt.figure(figsize=(15, 9))
plt.plot(price)
plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
plt.title("Bitcoin Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.show()

# Normalization
normalize = MinMaxScaler()
df_normalized = normalize.fit_transform(price.values)

# Build the model
# ------------------------------------------------------------------
num_units = 64
learning_rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(lr=learning_rate)
loss_function = 'mse'
batch_size = 5
num_epochs = 50

# Initialize the RNN
model = Sequential()
model.add(LSTM(units=num_units, activation=activation_function,
          input_shape=(None, 1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.1))
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer=adam, loss=loss_function)

# Train the model
# ------------------------------------------------------------------
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle=False
)

# Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Prediction
# ------------------------------------------------------------------
original = pd.DataFrame(min_max_scaler.inverse_transform(y_test))
predictions = pd.DataFrame(
    min_max_scaler.inverse_transform(model.predict(x_test)))

# Plot prediction vs real data
ax = sns.lineplot(x=original.index,
                  y=original[0], label="Test Data", color='royalblue')
ax = sns.lineplot(x=predictions.index,
                  y=predictions[0], label="Prediction", color='tomato')
ax.set_title('Bitcoin price', size=14, fontweight='bold')
ax.set_xlabel("Days", size=14)
ax.set_ylabel("Cost (USD)", size=14)
ax.set_xticklabels('', size=10)
