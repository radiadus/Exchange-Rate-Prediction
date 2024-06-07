# Exchange-Rate-Prediction
using LSTM-RNN to predict USD to INR from 2010 to 2017.

![output](https://github.com/radiadus/Exchange-Rate-Prediction/assets/55176713/8d783e9c-648d-4077-8165-b6a61b35266b)

I got this project as my mid term exam for my Artificial Neural Network course back at 4th semester. In this project I have a dataset of US Dollar to Indian Rupee price from 1980 until 2017. I was using Tensorflow, Keras and Sklearn to me at this project. First of all I read the dataset that was in a .csv file. I was using pandas to read the csv file and print it as in the image below.

    data = pd.read_csv('USD_INR.csv', index_col='Date')
    data.index = pd.to_datetime(data.index)
    data = data.iloc[::-1]
    print(data)

![image-5](https://github.com/radiadus/Exchange-Rate-Prediction/assets/55176713/256dde3b-52ae-41cf-8d3a-a01e21ed87e6)

_(USD to INR kurs dataset from 1980 until 2017)_

    plt.plot(data['Price'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

![image-6](https://github.com/radiadus/Exchange-Rate-Prediction/assets/55176713/994550d8-af7b-4a02-b519-3929a8bdd784_)

_(Plotting of the dataset)_

Then split the data into 80% for training dataset and 20% for testing dataset.

    data = data.filter(['Price'])
    dataset = data.values
    training_data = math.ceil(len(dataset) * 0.8)

After that, I was using MinMaxScaler() from Sklearn as my scaler for the dataset.

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

Right after scaling the dataset, we will use sliding window method so that the model could correctly forecast the testing dataset. I will append every 50 data of training dataset into new array named x_train as our features. Also I will append the next data as our target named y_train.

    train_data = scaled_data[0:training_data,:]
    x_train = []
    y_train = []
    for i in range(50, len(train_data)):
      x_train.append(train_data[i-50:i, 0])
      y_train.append(train_data[i, 0])
      if i<51:
        print(x_train)
        print(y_train)

Then lets reshape x_train and y_train so we could use them in keras model.

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

Build our model with Keras and fit it with x_train and y_train. Here I was using LSTM from Keras. I was used 2 layers of LSTM with 50 inputs because the sliding window size was 50. Then I added 2 fully connected layers with 25 nodes for the first one and 1 node for the last layer.

    model = keras.Sequential()
    model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.LSTM(50, return_sequences=False))
    model.add(keras.layers.Dense(25))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))
    
    model.fit(x_train, y_train, batch_size = 1, epochs = 1)

Then lets prepare our test data. We will need to implement sliding window method and reshape as well to our test data.

    test_data = scaled_data[training_data - 50:, :]
    x_test =[]
    y_test = dataset[training_data:, :]
    for i in range(50, len(test_data)):
      x_test.append(test_data[i-50:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

Last but not least make the predictions and plot them so we can see the result.

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_data]
    valid = data[training_data:]
    
    plt.figure(figsize=(16,8))
    valid['Predictions'] = predictions
    plt.title("USD to IRP")
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.plot(train['Price'])
    plt.plot(valid[['Price', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    plt.show()

Here is the result of the model predictions and plot of it.

![image-13](https://github.com/radiadus/Exchange-Rate-Prediction/assets/55176713/d61e1d1b-d517-4352-ab9b-e38dc65bae2e)

_(actual price and prediction price side by side)_

![image-14](https://github.com/radiadus/Exchange-Rate-Prediction/assets/55176713/3c79079d-a3d5-4bc7-90b9-b3d3d21e72e0)

_(plot of prediction price (green) and actual price (orange))_

The result was excellent which the predictions were very close to the actual price. Here we can conclude that LSTM are very powerful to do regression prediction.
