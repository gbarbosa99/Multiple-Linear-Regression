import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head(20))
print(df.dtypes)

# Using the 'BreakPointsOpportunities' column as our feature and 'Wins' as our output to train the model. 
features=df[['BreakPointsOpportunities']]
outcomes=df[['Wins']]

# Split our data into 80% training data and 20& testing data.
train_x, test_x, train_y, test_y = train_test_split(features, outcomes, train_size = 0.8, test_size = 0.2)

# Instantiate the LinearRegression object from SKlearn and train the model using our training data.
lr = LinearRegression()
lr.fit(train_x, train_y)

# Print the coefficient of determination of the prediction, with a score of .82, the model fits the data well.
print(lr.score(test_x, test_y))

# Predict the outcome values using our test data.
prediction = lr.predict(test_x)

# Plot the predicted outcome against our actual outcome.
plt.figure(figsize=(6, 6))
plt.scatter(test_y, prediction, alpha=0.5, marker = 'o', label='Data Points')
plt.title('Predicted vs. Actual Outcome')
plt.xlabel('Actual Outcome')
plt.ylabel('Predicted Outcome')
plt.legend()
plt.show()


### Multi Linear Regression
# Using the 'BreakPointsOpportunities' and 'DoubleFaults' column as our feature and 'Winnings' as our output to train the model. 
mlr_features = df[['DoubleFaults', 'BreakPointsOpportunities']]
mlr_outcomes = df[['Winnings']]

# Split our data into 80% training data and 20& testing data.
mlr_train_x, mlr_test_x, mlr_train_y, mlr_test_y = train_test_split(mlr_features, mlr_outcomes, train_size = 0.8, test_size = 0.2)

# Instantiate the LinearRegression object from SKlearn and train the model using our training data.
mlr=LinearRegression()
mlr.fit(mlr_train_x, mlr_train_y)

# Print the coefficient of determination of the prediction, with a score of .84, the model fits the data better than the single linear regression model.
print(mlr.score(mlr_test_x, mlr_test_y))

# Predict the outcome values using our test data.
mlr_prediction = mlr.predict(mlr_test_x)

# Plot the predicted outcome against our actual outcome.
plt.figure(figsize=(6, 6))
plt.scatter(mlr_test_y, mlr_prediction, c='orange', alpha=0.5, marker = 'o', label='Data Points')
plt.title('Predicted vs. Actual Outcome')
plt.xlabel('Actual Outcome')
plt.ylabel('Predicted Outcome')
plt.legend()
plt.show()
