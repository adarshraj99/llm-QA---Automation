<img width="1819" height="586" alt="image" src="https://github.com/user-attachments/assets/31d166dc-8990-46e7-92d4-f4af3f14df45" />

## Types of testing:
responsible ai testing, post production testing, shift left testing, Prompting testing, Memory Testing ,Context testing,

## Examples of ML models:
Spam filters in emails, Promotion recomendation engines by user preferences, Predective maintainance for industrial equipments.

#### Strengths and Limitations:
Flexiblity: Can get results even when data is not feeded to the ML model, because it already learned from the previous data and it analyzes with predections.
Scalablity: Can give broader and narrower results depending on model's **temprature** than the conditional results from an expret based traditional systems.
Interpretability: ML models have less interpretability and we don't know the outputs (As it's a prediction by machine). In expert based systems, Reactions are defined.

#### ML Model Life cycle:
<img width="1723" height="856" alt="image" src="https://github.com/user-attachments/assets/761983c5-58fe-4ed6-a655-23ab1d30e23c" />

* Devs use Training data and testers have validation data.
* Validation data have 2 problems: Underfitting(Need more data training on seen data) and Overfitting(If a high accuracy model is trained on more related but un-useful data than needed(ex: giving house colour,designs,facing directions. model can't predict price accurately) efficiency goes down).
* GoodFit: after UnderFitting, when model starts giving better results. Itis said a goodfit. It is just before overfitting.
* Overfitting: Caused by: - over training from unuseful data (called **noise & randomeness**). - Can be caused by more model complexity (not giving generalised output). - Can be caused by non-regularization. To be corrected by **Regularization techniques** like - Lasso(L1) ,Ridge(L2), Elastic Net. - Cross validation not implemented.

* Labeled data vs Unlabeled data:
  When ML model have only input data and don't have output data for training called Un-labelled data .When ML model have both Input and Output data for training called labelled data.

* Regularization:
A tehnique used to prevent overfitting by adding a penalty for the loss function. This penalty discourages the overly complex models making them more generalizable to new data. Non-regularization ,otherwise means tarining models without penalty which can cause overfitting, **High Variance**, Increased Complexity (capturing noise alongwith signals).

* Cross Validation:
Dataset is divided into sets of data. Some set of data is used for training the model and other sets are used for verification of the tranined model.

* High Variance:
Caused due to overfitting ,Noise in training data, Complex modeling. Here, model performs exceptionally well on training dataset but fails to generalize and performs poorly on the data.


#### Supervised Learning:
It is type of machine learning algorithm that learns from labeled data. Here, ML model needs a trainer to train model on the labeled data(with correct answer or tagged data).

* Advantages: Learns from previous experiences, solves real time computational problems.
* Disadvantages: Classifying big data can be challenging, Needs lots of computation time and resourses, requires labelled data set, needs training mentor. ex : This is usefull when want to predict outcome based on input features. ex: Email spam detection, image recognition, Predicting house prices, weather forecasting.

###### Types of supervised learning:

Regression supervised learning:

For contineous data prediction like house prices, stock prices, customer behaviour. These learn to predict numbers out of infinitely possible numbers. Some common types are:

Linear regression. Frameworks used for linear regression: Scikit Learn.
Polynomial Regression
Support Vector Machine Regression
Decision Tree Regression
Random Forest Regression
Classification supervised learning:
Solves a classification problem where the output variable is a categtory like any colour, any disease, any other condition, email spam. Here, the output is not infinite possiblities but only a set of outputs. It learns from input data and by using probablity distribution over output groups. Some common types are :
