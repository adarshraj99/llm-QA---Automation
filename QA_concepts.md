<img width="1819" height="586" alt="image" src="https://github.com/user-attachments/assets/31d166dc-8990-46e7-92d4-f4af3f14df45" />

## Types of testing:
responsible ai testing, post production testing, shift left testing, Prompting testing, Memory Testing ,Context testing,

## Examples of ML models:
Spam filters in emails, Promotion recomendation engines by user preferences, Predective maintainance for industrial equipments.

## Strengths and Limitations:
Flexiblity: Can get results even when data is not feeded to the ML model, because it already learned from the previous data and it analyzes with predections.
Scalablity: Can give broader and narrower results depending on model's **temprature** than the conditional results from an expret based traditional systems.
Interpretability: ML models have less interpretability and we don't know the outputs (As it's a prediction by machine). In expert based systems, Reactions are defined.

## ML Model Life cycle:
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


## Supervised Learning:
It is type of machine learning algorithm that learns from labeled data. Here, ML model needs a trainer to train model on the labeled data(with correct answer or tagged data).

* Advantages: Learns from previous experiences, solves real time computational problems.
* Disadvantages: Classifying big data can be challenging, Needs lots of computation time and resourses, requires labelled data set, needs training mentor. ex : This is usefull when want to predict outcome based on input features. ex: Email spam detection, image recognition, Predicting house prices, weather forecasting.

#### Types of supervised learning:

- Regression supervised learning:

For contineous data prediction like house prices, stock prices, customer behaviour. These learn to predict numbers out of infinitely possible numbers. Some common types are:

* Linear regression. Frameworks used for linear regression: Scikit Learn.
* Polynomial Regression
* Support Vector Machine Regression
* Decision Tree Regression
* Random Forest Regression

- Classification supervised learning:
Solves a classification problem where the output variable is a categtory like any colour, any disease, any other condition, email spam. Here, the output is not infinite possiblities but only a set of outputs. It learns from input data and by using probablity distribution over output groups. Some common types are :
* Logistic regression: Used for Binary classification tasks like : mail Spam Detection ,loan approvals by learning probablity of binary outcome. Used for both. Framework used : Scikit Learn.
* Support vector machines(SVMs): Effective in high dimentional spaces. Used for both regression and classification.
* Decision Trees: Suits both Classification and Regression supervised learning.
* Random Forests: Suits both Classification and Regression supervised learning.
* Naive Bayes: It is particularly usefull for text classification tasks, such as spam detection, sentiment analysis, and document categorization.
* K-Nearest Neighbors (KNN)
* Neural Networks :Highly capable model capable of handling various types of data and tasks including image and speech recognition by learning complex patterns from data. Library used : Tensor Flow
* Gradient Boosting Machines(GBM):

- Time Series Forecasting:

#### Ways to evaluate supervised learning models:

- For Regression: 
* Mean Squared Error (MSE):
* Root Mean Squared Error (RMSE):
* Mean Absolute Error (MAE):
* R-Squared (Coefficient of Determination)

- For Classification: 
* Accuracy
* Precison
* Recall
* F1 Score
* Confusion Matrix 

## UnSupervised Learning: 
Here learning happens from unlabeled or uncategorized data. Goal is to discover the pattern and categories in the Unlabeled data without explicit guidance. No training is given to the model. So, machines are supposed to find the hidden pattern ,actions or structures to categirize data.

Types of unsupervised learning: 

#### Clustering: 
Grouping similar data points together. It is a way to move silimar data points in nearer to the same clusters and away from the other non-similar data clusters. 
Techniques and methods are used to group data points into clusters based on their similarities: 

* Exclusive (partitioning)
* Agglomerative
* Overlapping
* Probabilistic

Types of clustering: 
* Hierarchial Clustering
* K-means clustering
* Principal Component analysis(PCA)
* Singular Value Decomposition
* Independent Component Analysis(ICA)
* Autoencoders
* Gaussian Mixture Models (GMMs)
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

#### Association : 
Here, relationship between different data in clusters are found. Such as ,people who buy x item may also buy y item. Common types of association Unsupervised learning include: 
* Apriori algorithm
* Eclat Algorithm
* FP-Growth Algorithm


## Testing UnSupervised data models(BlackBox functional testing):

#### Cross validation testing method:
1st step is Training on Training set. Then Training on Validation set. Here, can train the dev build with many inout data and see if build is able to segrigate the input data into different clusters and test it by asking questions like: where is my sales highest in which city ,product, questions on type of customers to target marketing.

Here, We have only un-labeled data. In **training data set**, model learns the needed patterns and structure in data. In **validation data set** ,input data(Training data set) is seprated into clusters of matching data.

#### Silhouette Scores(For Whitebox testing): 
This score is from -1 to 1. This checks the new untrained data score of model. If data is matching to the similar data cluster and not matching to the other nearby clusters, it is higher score. If score is matching to the nearby cluster, it is low score and shows data far away from matching cluster.
Can ask developers ,the methods and utilities where to pass data with some python methods to get the Silhouette score for different test data.

If the silhoutte scores are high with training data and suddenly score dips after more training ,it reached training Overfitting. So, should stop there. 

#### Calinski-Harabasz score: 
The Calinski-Harabasz score measures the ratio between the variance between clusters and the variance within clusters. Ranges from 0 to infinity. Higher score is better clustering.

#### Variance:
It is a measure of inconsistancy in the model's prediction when the models is trained on different subset of the same data. If variance is high it often leads to overfitting.

#### Adjusted Rand index: 
measures the similarity between two clusterings. It ranges from -1 to 1. higher scores indicating more similar clusterings.

#### F1 score: 
This is harmonic mean of precision and recall ,which are 2 matrics used in supervised learning to evaluate classification model. F1 score can also be used for Unsupervised learning.  Higher F1 score tells models can classify data better.

`F1 = 2 * (precision * recall) / (precision + recall)`
It is from 0-100% .Higher F1 score means better model.

### Applications of Supervised Learning: 
* Spam filtering, Image classification, Medical Diagnosys, Fraud Detection,

### Applications of unsupervised leaerning: 
* anomaly detection: Can identify unusual(total new) patterns or deviations from normal behaviour in data, enabling fraud detection, system failure etc.
* Scientific discovery: Can find hidden relation,pattern in data.
* Recommendation systems : on products, moview, songs etc
* Customer segmentation: Can cluster/group similar customers together.
* Image, audio, video segmentation

### Disadvantages of unSupervised Learning: 
* Difficult to measure accuracy or effectiveness due to lack of labeled data.
* Lesser accuracy.
* Noisy data can be difficult to cluster.
* Number of classes not known

### Disavantages of supervised learning: 
* Cannot process very large and more complex data from supervised learning.

## Hybrid Learning: 
* Combination of Supervised and Unsupervised learning. eg: ChatGPT. 

## Reinforcement Learning: 
Here model learns from it's surrounding ,actions taken by other users, here model's decision depends on the current state not the history. This needs trial and error for learning. 
How it works: 
The agent interacts with the environment and receives rewards for performing actions. The agent learns from the feedback of each action and discovers the best processing paths. The agent learns the optimal behavior in an environment to obtain maximum reward. 
ex: In Robotics for task performing, Gaming for responding to player's actions, Autonomous cars for teaching navigation to self driving cars.

Process:
Define the environmentand reward for the agent.
Create the agent that specifies the policies involved.
Use neural networks or lookup tables to represent the policy.
Choose the suitable RL training algorithm.

Types of Reinforcement learning:
* Positive: Here model learns to create a positive behaviour or step as a result of environment update to get goals.
* Negative: Here model learns to undo a negative behavious or step as a result of the env change to get goals.

Some Elements of RL: 
Policy: Defines the agent’s behavior at a given time. 
Reward Function: Defines the goal of the RL problem by providing feedback.
Value Function: Estimates long-term rewards from a state.
Model of the Environment: Helps in predicting future states and rewards for planning.

Famous Frameworks: 
* Scikit Learn: Mainly for Linear regresion and Logistic regression.
* Tensor Flow: Mainly for Neural Networks, Reinforcement Learning.
* Keras: High Level Neural Network API ,runs on top of TensorFlow for better UI.  
* PyTorch:For DL models. Used mainly in research.
* XGBoost:For Gradient Boosting algorithm. 
* Light BGM: A Gradient Boosting Framework for large scale data. 
* CatBoost: A Gradient Boosting Framework, specialized in categorial data. 
* OpenAI Gym: For Developing and comparing Reinfrcement Learning Algorithms. 
* Stable Baselines:



## QA Questions  
1. Which type of learning team is using from supervised, Unsupervised, Hybrid(Supervised+Unsupervised), Reinforcement ?
2. Which Algorithm Team is using for the type of Learning? As there are multiple for every type of learning mentioned above.
3. Which Framework(Library) team is using to implement algorithms?
4. Can  you give me utilities/methods where i can generate overfitting and Underfitting graphs for Linear regression.
5. Can you give me utilities/methods where i can generate scores for unsupervised learning ? 


## QA Testing from evaluation phase:  

#### Temprature Testing: 
Temprature testing is used to control the randomness of predictions in the generative models. Lower temprature makes model more deterministic while higher temprature makes model more unpredictive and creative. In testing we can test models stablity at different tempratures.

#### 0 Shot Testing prompts:
Testing how the model handles requests without prior training on similar Category of data. Ex: asking medical domain trained model about geography questions. Model is expected to generalize the outputs and also use the other categories training data. 
Here, model relies on its broader understanding of the domain, like semantic relationships between concepts, to make predictions. 

#### Chain of thought testing prompts: 
Checking how well the model can follow a sequence of logical steps or reasoning. ex: giving a maths questions. Here, model can give single word answer or answer with all the steps to solve the question. There can followup cross questions.  
Also after refreshing the chat  should delete the old chat memory and after that any new chat should not followup context with the old chat.

#### Test whether model stays relevant to the topics or goes away when in a conversation.

#### Fantasy Claims Definition:
Testing model doesnot go beyond the needed resonable answer and it does not makes wrong assumptions. ex: asking 'Can human eat rocks' model makes a story and gives benifits of eating rocks.  

#### Accuracy Testing Definition: 
It is needed in Image identification. Suppose business criteria for model image identification is 90%. Model gives 93% of times correct answer, it is pass. Less then 90% is fail.  

#### Repeatablity testing: 
ask same question again and again and rephrase the same questions and ask. Model should not deviate from the trained data doubting the previous answer. 

#### Style Transfer Testing: 
Test whether the tone or the style of output have been changed. ex: ask llm to make formal and other casual mail on same topics. 
Another style change to test can be: text 'Please let us know if you need more information' at the end of every output.  This can be performed on the dev build platform like: platform.openai.com/playground.
Another style recognition is to read the intent of chat input. Model should see diffrence between humor, sarcasam.
Another is, if user is asking wide question model should not self decide and refer to narrow answer like 'suggest top restaurants in bangalore'. Model answer should have all varities of top restaurants including chinese,italian,french,northIndianetc..

#### User Location Based filtered answers: 
If user is based in Indian Hydrabad and asking about city hotels. Model should not refer to the Pakistan's Hydrabad city.

#### Invarience Testing: 
The output of model should not change for a question after adding irrelevant points to the same question. Medical ex: By changing Patient's name, DOB it should not change models prediction of patient's heart attacks chances. 

#### BiDirectional Testing: 
In some models when the input data gets reversed (ex: input data about the 2 different lungs of same person) the modles prediction about lung cancer should not change.
In some models reversing the input data should change the output. 

#### Responsible AI testing:
The model output must be ethical. Types: Bias Testing, Explainability, Safety Testing, Privacy protection, Red Teaming. 

#### Transperancy testing: 
To check any business criteia correctly by model. ex: If a loan being distributed. model should predict correctly about the loan taker ability to return loan and other criterias. And, model should give the loan rejection/approval points in detail to share with the customer.

#### Ethical Testing: 
To test ethical implications of AI 

#### Data Privacy and Security testing: 
model should not give personal outputs about any public figures even if have that knowledge. In testing people try to confuse model and try to get answer, but model should not answer it.

#### Model Generalization Testing: 
If training data is limited to certain group or geolocation. 

#### Societal Impact Testing: 
Answers for guidance should be ethically answered and with positive socitical impacts. like: drugs gelling guy address, how todo illegal stuffs etc.

#### Integration testing: 
To check if the model/build is working correctly with the production servers. 

#### Latency Testing: 
Testing for the model answers in a timeframe. Check in normal and peak load. 

#### Drift Testing: 
* Data Drift Testing: if real life data is changing w.r.t. time for training model. Model should be updated accordingly. Here, way of asking questions to model is changed like 'I have this symptoms, do i have Covid?', here input symptoms have changed. Drifts can be of many types : sudden drift, gradual drift, incremental drift, and reoccurring drift(happens after some time).

* Concept Drift Testing: Similar as data drift, here any working method or concept changed. Here question to model is same but the output must change as concept is changed. like selenium way of invoking webdriver have changed. 

* Monitoring: a trained model can degrade over time (maybe bug). So, we need contineous monitoring and testing of the builds. There are tools which can contineously monitor change in the version or any other model's input training data like **Evidentlyai.com**, **Amazon SageMaker Model Monitor** . These tools generate alarm when some data changes so that we can do the monitoring for Drift Testing.

#### Shadow Testing: 
When developed model is trained on new data. We can test this new version of model with live data questions asked by end customers and check outputs. 

#### A/B Testing: 
This is same as shadow testing in live model data. Here, data from live users are diverted into new model (updated non-released version). Can do shadow and A/B testing for Patch releases. 

