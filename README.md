🚗 Traffic Volume Prediction using LSTM 📈

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to predict metro interstate traffic volume. The project utilizes a Jupyter Notebook to perform data preprocessing, model building, training, and evaluation.
📝 Project Overview

The goal of this project is to build a time-series prediction model that can accurately forecast traffic volume. The model is an enhanced version of an LSTM network 🧠, a type of recurrent neural network particularly well-suited for sequence prediction problems.

🛠️ Key Libraries Used

    pandas 🐼: for data manipulation and analysis.

    numpy 🔢: for numerical operations.

    matplotlib.pyplot 🎨: for data visualization.

    sklearn.preprocessing.MinMaxScaler ⚖️: for scaling the data.

    tensorflow.keras.models.Sequential 🔗: to build the neural network model.

    tensorflow.keras.layers.Dense, LSTM, Dropout 🧱: to define the layers of the model.

    sklearn.metrics 📏: for evaluating the model's performance.

💾 Dataset

The project uses the Metro Interstate Traffic Volume dataset. The initial data contains the following columns:

    holiday 🎉: A categorical variable for holidays.

    temp 🌡️: The temperature in Kelvin.

    rain_1h 🌧️: Amount of rain in the hour.

    snow_1h 🌨️: Amount of snow in the hour.

    clouds_all ☁️: Percentage of clouds in the sky.

    weather_main 🌦️: A short categorical description of the weather.

    weather_description 📝: A more detailed categorical description of the weather.

    date_time 📅: The timestamp for the data point.

    traffic_volume 🚙: The target variable, representing the number of vehicles per hour.
<img width="940" height="274" alt="image" src="https://github.com/user-attachments/assets/b52df1ac-9bf1-4690-81b4-e6b6ae9ee0fe" />



⚙️ Data Preprocessing

The raw data undergoes several preprocessing steps to prepare it for the LSTM model:

    Load Data 📂: The Metro_Interstate_Traffic_Volume.csv file is loaded into a pandas DataFrame.

    Feature Engineering & Cleaning 🔧:

        The date_time column is converted to datetime objects and set as the DataFrame's index.

        The holiday and weather_description columns are dropped.

        The weather_main column is converted into numerical format using one-hot encoding.

        New features, hour and day_of_week, are extracted from the date_time index.

    Scaling ⚖️: The numerical features are scaled to a range between 0 and 1 using MinMaxScaler. This is a crucial step for neural networks to ensure stable and efficient training.

    Sequence Creation 🔄: The data is transformed into sequences of 48 hours to predict the traffic volume for the next hour.

🤖 Model Architecture

The prediction model is a Sequential Keras model with an enhanced LSTM architecture:

    Layer 1️⃣: An LSTM layer with 100 units (return_sequences=True).

    Layer 2️⃣: A Dropout layer with a rate of 0.2 to prevent overfitting.

    Layer 3️⃣: A second LSTM layer with 100 units.

    Layer 4️⃣: Another Dropout layer with a rate of 0.2.

    Layer 5️⃣: A Dense layer with 50 units.

    Output Layer 🎯: A final Dense layer with 1 unit to output the prediction.

The model is compiled using the adam optimizer and mean_squared_error as the loss function.

🏋️‍♂️ Training and Evaluation

    The dataset is split into training (80%) and testing (20%) sets.

    The model is trained for 25 epochs with a batch size of 64.

    Validation is performed on 10% of the training data.

After training, the model's performance is evaluated on the test set.

🏆 Performance Metrics

The model's performance on the test set is measured using the following metrics:

    Mean Absolute Error (MAE): 1.48

    Root Mean Squared Error (RMSE): 1.88

    R-squared (R2) Score: 0.9815 ✨

The high R2 score suggests that the model explains a large portion of the variance in the traffic volume data, indicating strong predictive performance.
<img width="940" height="361" alt="image" src="https://github.com/user-attachments/assets/3f10f682-88fb-42f8-999d-d6fc0192b702" />

📊 Visualization

A plot comparing the actual and predicted traffic volume on the test set provides a visual representation of the model's accuracy. The plot shows a strong alignment between the predicted and actual values, further supporting the high evaluation scores.
<img width="940" height="471" alt="image" src="https://github.com/user-attachments/assets/708c2cee-85a4-4c66-a4fc-7defeaeb994b" />
