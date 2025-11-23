Convolutional Neural Networks also called Computer vision.

transformers: 

bias: 

variance: 

### Neural Networks: 
Network of neurons. Neuron is a **Function to be created** if we know some Actual Inputs and Actual Outputs. Neuron is a fundamental processing unit, analogous to brain neurons which receives input signals, **applies calculations(in weights & Biases)** and produce output signals. It gets trained with more and more Input data and later by **decreasing the error (Cost) following the gradient descent concept.** 

<img width="404" alt="image" src="https://github.com/user-attachments/assets/b11cd770-ac24-47bb-8978-e403256a9e92">

In a linear 2D neuron like **F(x)=W(x)+b** with W as weight and b as bias. If we change W ,the outputs on 2D graph rotates and if we change the b the output on the 2D graph changes position as up and down. 

But, Linear function can only combine to make linear functions. But, we need non-linear more complex functions. 


### 👉 Convolutional Neural Network(CNN): Convolutional means twisting, coiling togetner.
A type of Neural Network with 

**input layers**:
- Passed data (Mainly images in CNN) to other layers.

**Convolutional Layers**:
- Filters input image. **Extracts features from input dataset**. Applied a set of learnable filters called **Kernels**. These Kernels are small matrices usually 2*2 to 3*3 or 5*5 shape. Different layers checks different parts of input data with the layer available image part. Output of this layer called **future maps**.

**Activation Layer**:
- Takes input from convolutional layers. Activation function adds non-linearity to the network. It will apply an **element-wise activation function** to the output of the convolution layer. The data volume dosen't change in this layer.

**Pooling Layer**: 
- This layer is periodically inserted. It's main function is to downsample data for faster computation reduces memory.  2 main types of poolings are: max pooling and average pooling. 
Fully connected layers and Output Layers.

Convolutional Neural Network is used for **Pattern recognition to match Images or videos**. Each layer of CNN checks for a particular pattern in a part of image(in certain pixels only).

Uses: For Image and video segmentation and detection. Used in CV, selfDriving cars, medical image analysis.


### 👉 Recurrent neural network(RNN): 
Used in processing sequential data like words ,time series. So, used in **Speech recognition, language translation, voice Sentiment analysis**. 

Cons: These are being replaced by LLMs and transformer based AI. 
These work by remembring past info by feeding output from one layer to other layer. Ex: for predicting the next word in sentance RNN uses previous words to help decide most likely next word. 


### Fully connected neural network(FCNN): 
All major types of NNs are FCNNs like CNN, RNN etc. 
artificial neural network where the architecture is such that **all the nodes, or neurons, in one layer are connected to the neurons in the next layer**.
FCNNs are more dense and have more connections than CNNs, but basic design is same.


### 👉 Generative Adversarial Neural Network (GANN): 
GANs train 2 Neural Networks compete against each other. Used for Deep larning. 1 network(the generator) creates new data and other network(the discriminator) identifies real data from the fake data created by the generator. The discriminator learns to diffrentiate between the 2 types of data. 
This is used for **Image generation from text, generate music, New Videos, Cybersecurity, Can generate real training data for other models**.  


## Transformer Neural Networks:
Replaced RNNs for llm models. It works on Self-Attention Layers which Helps the model focus on relevant words in context to understand relationships between words in a sentence, regardless of their position. 
- Input Embeddings: Converts words into numerical vectors.
- Positional Encoding: Adds word order information to embeddings.
- Self-Attention: Lets each word focus on other relevant words in the sentence.
- Multi-Head Attention: Applies attention multiple times in parallel to capture different patterns.
- Feedforward Layers: Processes each token through a fully connected neural network.
- Layer Normalization & Residual Connections: Stabilize learning and improve performance.
- Stacked Encoder/Decoder Blocks: Multiple layers help the model learn complex patterns.


### Deep Q Learning: 
Technique in reinforcement learning. 
It trains an agent to maximize rewards by learning which actions to take in different situations. 


### 👉 Underfitting vs Overfitting: 
Underfitting (Undertrained or trained on unrelevant data):
The model is too simple to learn the underlying patterns in the data.
Poor performance on both training and test data ,High bias, low variance.
- Not enough model complexity (e.g. linear model for nonlinear data)
- Too few features or inadequate training
- Needs more training on relevant features

OverFitting (model is too complex, memorizing noise instead of learning general patterns):


#### 👉 Labelled data: 
Answer or output of the target variable that a model is trying to predict. This is used to learn and make predictions. Input data is labeled pre-processing of data. 


#### 👉 Temprature: 
Temperature is a parameter in generative AI (GenAI) that controls the randomness of the output. Higher temprature means more creative output and lower temprature gives more predictive output. General range is 0-1. 


#### 👉 Tokens: 
Smallest fundamental independant data blocks made by genAI model from the given input. Model makes small blocks to find which block suits with which block to make a correct output. These can be words, punctuation marks, or even sub-words.


#### Embeddings : 
**Encodes the tokens** with it's meanings based on that token data. As, input passes through the model layers ,tokens gets updated Embeddings. These are a way to represent real-world objects as mathematical vectors. 
**Embeddings capture the meaning and grammatical properties of words. For example, words with similar meanings have similar embeddings.** 
Embeddings convert high-dimensional data into low-dimensional vectors. This makes the data easier to process. 
Used for finding similar texts, images, helps ChatsBots to understand relationship between words and images, Fraud detection by identifying certain similar pattern.


#### Context: 
amount of text data a model can process at a given time.
it is the surrounding information, circumstances, and **relevant factors that influence the interpretation** and decision-making process of an AI algorithm. It is like understanding the whole picture before making a judgement.
ex: In image recognition, the surrounding objects in a scene can provide context about an object.
    In text analysis, previous sentences in a paragraph can give context to understand the current sentence.
    In a recommendation system, a user's past purchase history can provide context for suggesting new items.
- Tokens keeps updating itself based on the input data which helps in the output data needed.


#### Hallusination: 
AI bot gives incorrect answers because of limited data training or having multiple outputs, lenghty complex prompts, human phrase languase non-understanding. ex: prompt: "my schoolteacher head is on fire .What should i do ?"  ,  "she is as blind as a bat. What should i do ?"(here i meant she dosen't care about things around).

#### Banned Contents: 
Some contents are banned from GPT. like : info on political figure location ,info on global wars like :who will isreal kill next?   

#### Text classification and sorting:
the process of automatically categorizing text data into predefined classes or categories.

#### Sentiment analysis: 
he process of analyzing written text to determine if it has a positive, negative, or neutral sentiment. 

#### Information extraction(IE): 
Types :Named-entity recognition(NER)&,...

#### Speech recognition:
the process of converting spoken language into text using machine learning models. 

#### Natural language understanding and generation(NLU & NLG):

#### Computer Vision: 

#### Image classification: 

#### Target detection: 
the process of identifying and locating specific objects of interest within an image or video, essentially pinpointing their position and classifying what they are, often using computer vision techniques. Uses: autonomous driving, surveillance systems, and image analysis.


#### Image segmentation: 
a computer vision technique that involves **dividing a digital image into multiple segments, where each pixel within a segment shares similar characteristics like color, texture, or brightness**, allowing for detailed analysis of objects within an image by assigning a label to each pixel. It's like creating a "map" of the image by identifying and separating different parts of it.
Unlike object detection which identifies objects within an image, image segmentation assigns a class label to each individual pixel, providing a more precise outline of objects.
Techniques for Image segmentation: 
**Thresholding**: Simple method where pixels are classified based on intensity values exceeding a set threshold.
**Clustering algorithms**: Grouping pixels with similar features using techniques like **K-means clustering**.
**Region growing**: Expanding regions from seed points based on pixel similarity.
CNNs are widely used for Image segmentation. 


#### Significance test:
a statistical method used to determine whether an observed effect (like a relationship between features and target variable) is likely due to a real phenomenon or simply random chance.


#### Overfitting:

#### Wrap Up:


#### Parameter tuning: 
the process of adjusting a model's settings to improve its performance. Different parameters settings are done to see and verify their results to select the best parameter giving best results.
parameter tuning techniques: 
Grid search: An exhaustive method that evaluates every possible combination of parameter is checked for output.
Bayesian optimization: A **probabilistic method** that uses a model to predict which parameter values will perform best.
**Random search**: A method that randomly selects a combination of parameters to test.


#### Confusion Matrix: 
It is a table that **compares predicted values to actual values** for a dataset. It's a way to evaluate how well a machine learning model is performing. 

#### Feature Extraction: 
process that transforms raw data into new features that are more useful for machine learning models. it is one of dimensionality reduction technique. 
How it works: 
Combines or modifies original data to create new features
highlights the most meaningful information
Simplifies the model's task while retaining as much relevant information as possible

#### Model Training: 

#### Dimensionality Reduction: 
a technique that transforms a large dataset with many features into a smaller dataset with fewer features, still preserving the important information, effectively reducing the complexity of the data and improving the efficiency of machine learning models by removing redundant or irrelevant information. 
Common techniques are :
Principal Component Analysis (PCA), 
Linear Discriminant Analysis (LDA), 
Independent Component Analysis (ICA)


#### Classification(Binary) vs Regresion ML Techniques: 
Classification is used when the output variable is a category. Email Spam Detection, Image detection(Dog/Cat/Other), Sentiment Analysis. Algorithms: Logistic Regression, SVM, Random Forest, Decision Tree, CNN. Checks by : Accuracy, F1 Score, 
Regression is used when the output variable is a continuous value. The goal is to predict a numerical value based on the input data. ex: House Price Prediction, Stock Market Forecasting, Weather Prediction. Algorithms: Linear Regression, Polynomial Regression, Support vector regression, RNN, Checked by: Mean Absolute Error, ean Squared Error, R-squared(Coefficient of Determination). 


#### Multiclass classification: 
There is only Binary and Multiclass classification types of classification in ML. Unlike Binary Classification it is for categorize data into more than 2 groups. It handles more complex scenarios. ex: Handwriting recognition(can be matching to one or more person), Text classification (Text can be classified onto multiple groups). 


#### Different types of Regression: 
In ML different types of regression are designed for working with different types of data and predictions. Most commonly used regression types are : Linear regression, Polynomial regression, Ridge Regression, Lasso Regression, Elastic Net Regression, Logistic regression, Support Vector Regression, Decision Tree Regression, Random Forest Regression, Step wise Regression.


#### Single Linear Regression: 
It is most basic regression and it is for predictive analysis. linear regression model is instantiated to fit a linear relationship between input features (X) and target values (y).
ex: predicting house prices based on square footage, forecasting sales based on advertising spend, estimating a patient's blood pressure based on their age.
```
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X_new)
```


#### Multiple Linear Regression:
Most common for of regression. Multiple Linear Regression basically describes how a single response variable Y depends linearly on a number of predictor variables. 
ex: selling price of house can depend on location, number of bedrooms and bathrooms, built year, area ,etc..


#### Polynomial Regresion: 
Polynomial Regression is a form of linear regression in which the relationship between the independent variable(Input) x and dependent variable(output) y is modelled as an nth-degree polynomial. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y. 
<img width="455" alt="image" src="https://github.com/user-attachments/assets/82ceccd1-bf35-46fb-907e-a2e71441709a" />
here, independant variable means input and dependant variable means output.

A higher degree allows the model to fit the training data more closely, but it may also lead to overfitting. 


#### Logistic regression:
It is a classification type not a logistic regresion type of training.  Types: 
Binomial: only two possible types of the dependent variables, such as 0 or 1, Pass or Fail, etc.
Multinomial: In multinomial Logistic regression, there can be 3 or more possible unordered types of the dependent variable, such as “cat”, “dogs”, or “sheep”
Ordinal: there can be 3 or more possible ordered types of dependent variables, such as “low”, “Medium”, or “High”.


#### Evaluation Matrix: 
In every ML model training of any type, the goal is to minimise the loss (diffrence between the predicted and actual values).
This is a measure to check performance of a machine learning model. Different models require different evaluation metrics, depending on the specific goals and nature of the problem. some types for regression, classification:

###### Classification: 

* Accuracy:
  
  <img width="217" alt="image" src="https://github.com/user-attachments/assets/971e4b91-f82c-407c-b6ae-b361ef019685" />
  
* Precision:
  
  <img width="228" alt="image" src="https://github.com/user-attachments/assets/a2b90d6e-5fb1-42b2-9db2-110867df6ad0" />
  
* Recall (Sensitivity):
  
  <img width="202" alt="image" src="https://github.com/user-attachments/assets/cde439df-72e4-4aaf-b950-7675ae3a65ef" />
  
* F1 Score:
  
  <img width="184" alt="image" src="https://github.com/user-attachments/assets/375d1d3d-ed72-4af3-9fad-6c8a0157eee9" />
  
* ROC-AUC (Receiver Operating Characteristic - Area Under Curve):

* Confusion Matrix:

###### Regression Matrix: 

* Mean Absolute Error (MAE):
  <img width="451" alt="image" src="https://github.com/user-attachments/assets/62f97ab5-c323-4198-a5c0-f345949159bf" />

  
* Mean Squared Error (MSE):
  <img width="454" alt="image" src="https://github.com/user-attachments/assets/c158fc3d-b647-4791-ba01-035a8db58816" />

  
* Root Mean Squared Error (RMSE):
<img width="344" alt="image" src="https://github.com/user-attachments/assets/2e370af8-4ebb-4cd7-a63e-b2ba4972d2db" />

  
* R-squared (R²):
<img width="500" alt="image" src="https://github.com/user-attachments/assets/483d5a23-af8b-48e3-9b6f-312f2901d072" />

  
###### Clustering Matrix: 

* Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to 1, with higher values indicating better clustering.

* Davies-Bouldin Index: Measures the average similarity ratio of each cluster with respect to the clusters that are most similar to it. Lower values indicate better clustering.

* Adjusted Rand Index (ARI): Measures the similarity between the clustering results and a ground truth class assignment. Values range from -1 to 1, with higher values indicating better clustering.


#### SoftMax Regression: 

#### Naive Bayes: 
supervised machine learning algorithms used for classification tasks, based on Bayes’ Theorem to find probabilities. 

#### Support Vector Machines:

#### Decision Tree: 

#### Random Forest: 

#### Clustering: 

#### K means clustering: 

#### Types of Clustering: 

#### Dimensionality Reduction: 

### Datasets:

#### Training data set: 

#### Validation data set:

#### Test data set:
