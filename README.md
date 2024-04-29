Payment fraud refers to illegal or deceptive transactions conducted by criminals with the intent to secure goods or services without proper payment or to steal funds from an individual or organization. 
This type of fraud can occur across various platforms including online transactions, credit card transactions, wire transfers, and mobile payments. I have created a Machine learning framework to detect such transactions from the publicly available data at Kaggel "https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download" .The machine learning models like XGBoost and RandomForest were used to create the classification models for fraud predictions that achieved an accuracy score of 1.  The details related to the data set are as follows:

step: represents a unit of time where 1 step equals 1 hour
type: type of online transaction
amount: the amount of the transaction
nameOrig: customer starting the transaction
oldbalanceOrg: balance before the transaction
newbalanceOrig: balance after the transaction
nameDest: recipient of the transaction
oldbalanceDest: initial balance of recipient before the transaction
newbalanceDest: the new balance of recipient after the transaction
isFraud: fraud transaction

