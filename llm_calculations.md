## Denotions in ML algorithms:

* #### Superstrip:
  <img width="264" alt="image" src="https://github.com/user-attachments/assets/0145e054-40ed-4f22-9d58-c107601902fb" />

  It looks like power in circle bracket. It means number of the training example from a list of training examples and their outputs. ex: 1st or 2nd or .....nth size in feet^2 example.
  
  <img width="269" alt="image" src="https://github.com/user-attachments/assets/c8e8911e-3b9c-4e0d-ac35-61c766446a09" />
 
* #### Hypothesis, hat:
  ML model's function is called hypothesis. This function takes input data x and predicts the output called Y-Hat(ŷ). This prediction ŷ is the estimated value of data in y-axis for value in x-axis with a found out function f.  

* #### Function:
  It is calculated by input learning data in ML model.  
  <img width="110" alt="image" src="https://github.com/user-attachments/assets/7b17dd96-5a5e-41f3-9cca-027042e53a0f" />

  Here, w&b is called parameters or Weights or co-efficients.
  if w = 0 ,y = b.
  w is slope and b is point on y-axis.

<img width="237" alt="image" src="https://github.com/user-attachments/assets/a9c8f567-bf2e-41a5-a7c7-7f5f38401531" />

* #### Error:
  It is diffrence between the ML model prediction and the value of y for given x input data. 
  error = ŷ - y
  
* #### Cost Function:
  When square of all errors values are summed from 1 to m  x value and this sum is divided by the number of occurances m, it makes Cost function. Then it is divided by 2 by convention to be useful in later calculations.  
    
  <img width="382" alt="image" src="https://github.com/user-attachments/assets/b9a178b1-a1e3-45c5-8979-ed1445b7a6b6" />

  Also called Squared error cost function represented by J(w,b). which can also be written as :
  
  <img width="296" alt="image" src="https://github.com/user-attachments/assets/557ef641-5cf2-42d1-bc2f-165a27175139" />

Parameters w and b are calculated to reduce cost function J(w,b).

<img width="326" alt="image" src="https://github.com/user-attachments/assets/e870009c-dbea-464e-87bf-0e5215df01ba" />

Here, error is (model's prediction) - (actual value) . i.e.:

<img width="142" alt="image" src="https://github.com/user-attachments/assets/31c012d6-e39a-422b-be5b-a7a406ec27ec" />

In Squared error cost function, the cost function does not have multiple local minimum ,it have only 1 local min anyttime. So, squared error cost function have only global min ,not local min.


#### Multiple Linear Regression: 
Applying cost function to get the house prediction output with multiple parameters like: Number of bedrooms, size, floors, age.

<img width="895" alt="image" src="https://github.com/user-attachments/assets/2255d95d-6c59-4a93-b4ef-8bf249fb3e70" />

here, all 'w' and 'b' are vectors and symbol for itis w or b with arrow above it. 


#### Vectorization: 
This is used for writing faster executable code for model's prediction. 

If n is less number eg: n=3. We can do it without vectorization. like: 

<img width="286" alt="image" src="https://github.com/user-attachments/assets/713be1d0-8d54-4124-9a77-528233f3411a" />

Can also do it without vectorization with **summation and python range()**. But, this is not much efficient:

<img width="302" alt="image" src="https://github.com/user-attachments/assets/bd3e965f-7b27-4159-ae57-7190949069fa" />

But, if n very big number, need vectorization with python dot() .adv is, It is shorter code and it runs faster as **dot() uses muliprocessing** .

<img width="206" alt="image" src="https://github.com/user-attachments/assets/28a15f88-745d-46c0-a3bd-a36f46ee975f" />

Vectorization can be used in Gradient Descent for parallel processing.  


#### Multiple linear regression with vectorization for gradient descent: 

In vector notation :

model function <img width="371" alt="image" src="https://github.com/user-attachments/assets/0bf7cd73-9092-44a8-b8ea-9b8f706656a0" />    is written as    <img width="154" alt="image" src="https://github.com/user-attachments/assets/8c80eb1a-fb9d-4c41-bae4-eed2e0a5642a" />

Cost function <img width="279" alt="image" src="https://github.com/user-attachments/assets/0eb49a31-aae5-41c4-aef4-1d0111d65ae8" />   is written as  <img width="91" alt="image" src="https://github.com/user-attachments/assets/9a7c44d0-0bb7-4e42-a074-f1a1691ccbf1" />

In Gradient descent  <img width="371" alt="image" src="https://github.com/user-attachments/assets/ae45226a-9df6-4361-99fa-5e93dd659c9a" />  can be written as   <img width="221" alt="image" src="https://github.com/user-attachments/assets/b87945ee-c3cc-430e-a102-10c06b9164aa" />

For multiple linear regression: 

W is : 

<img width="276" alt="image" src="https://github.com/user-attachments/assets/76a267d7-0a94-45a7-bf90-fc4b8bc08996" />

multiple w is calculated : 

<img width="327" alt="image" src="https://github.com/user-attachments/assets/487f7eae-ac40-43f6-94db-ea7465791299" />

b is: 

<img width="326" alt="image" src="https://github.com/user-attachments/assets/7aa6659d-3acc-46a4-acfe-ab3250200a48" />

and , simultaneously update wj (for j=1 ,......,n) and b


#### Normal Equation: 

This is an alternative way to get w and b. This method works only for linear regression and does not need iterations to get w and b values, it can do in 1go with an advanced linear algebra library.  

#### Gradient Descent Feature scaling : (with feature size vs parameter size):
This helps achiving global min in gradient descent much faster. ex:  In Guessing house price, suppose x1 and x2 are size and bedrooms. so, size takes wide range of numbers and bed rooms 

<img width="823" alt="image" src="https://github.com/user-attachments/assets/9cab9159-ecef-41f2-91ee-926b3ce56639" />

Here, For for bigger value of x ,w is smaller. If x1 and x2 are 2000k and 5 then w1 and w2 are 0.1 and 50 respectively. This calculates house price nearby the actual house.

<img width="443" alt="image" src="https://github.com/user-attachments/assets/55bfdd01-eeb1-4160-9ada-08f31519bd28" />

graph and contour plots of these x and w values: 

If x1 is very big and x2 is very small & w1 is very small and w2 is very big. 

<img width="853" alt="image" src="https://github.com/user-attachments/assets/8bdd5572-ce42-489a-bc5f-fadcfd9cf567" />

here, if w1 is very small, it gets multiplied with x1 and becomes a very large number and big change to the house price estimate. 
however, w2 is very big, It gets multiplied with x2 and becomes a very small number and does not effect house price much. 
With this training data is used it will take much time to find global min. in gradiend descent (bouncing back and forth) before finally finding global min. 

<img width="470" alt="image" src="https://github.com/user-attachments/assets/a1e1666a-e4f1-49af-801f-548eaf21a334" />

**Having different features with various range of values makes it difficult to get the global min from gradient descent.** 
So, should making smaller numbers bigger or bigger numbers smaller (**Ultimately need to put all features in same scale for getting global min faster**) to get a clear scattered plot and contour plot. 

<img width="806" alt="image" src="https://github.com/user-attachments/assets/3e2bec0d-2b12-437c-bc47-fa2d7f3a3923" />

To achieve it ,we need Feature Scaling : 

#### Feature Scaling:
It should be performed if features are not around -1 to +1. Rahter they are very small and very big numbers or they are not around 0 in 2D axis plane.

- ##### Dividing by max:
   
<img width="597" alt="image" src="https://github.com/user-attachments/assets/8d5aaa0c-0f8a-4950-8e25-8f09fbbdfc5d" />

As value of x1 is very big. It can be divided with it's max value and that can be plotted on graph in x1 from 0 to 1 range. 

- ##### Mean Normalization:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/73874f74-7db2-4ac0-9d02-de7868308286" />

Here, To calculate x1 and x2. x1-(avg. of x1)/(max-min) and the values we will get between -1 and +1 around 0 in graph. 

- ##### Z Score normalization:

<img width="605" alt="image" src="https://github.com/user-attachments/assets/54fe4d62-7c9e-43a7-ad82-893db34286b2" />

It is found from standard deviation of each feature and mean from the gallatian distribution or mean curve. 


#### Checking Gradient Descent Conversion:

Make a graph j(vector w, b) vs iterations. Here, j() should decrease after every iteration. If cost function J() is not decreasing then alpha(Learning rate) must be coosen poorly (too large) or potential code bug. After some iterations the curve gets flat and it is not decreasing .Ths is called **convergense**. 

#### aplha (Learning Rate):
alpha is a key parameter to optimize weights and biases while calculating the gradient descent.

#### Choosing alpha correctly:

If alpha coosen is too small, it will run very slowly and go up and down or can keep increasing.  If it is too large ,it may not converge . 

<img width="626" alt="image" src="https://github.com/user-attachments/assets/27211fd8-0dec-41f2-8cd9-36ac98dc6936" />

If not getting global min learning rate in gradient descent. One may set alpha very very small and then if it is not working. Then there must be a bug in code. 
Very small alpha also then steps to find min will increase taking more time. 

So, should try with a very small alpha and then try with a bigger alpha to find where alpha is not decreasing and keep choosing smaller than the bigger value to get correct alpha. 

#### Feature Engineering: 

Feature Engineering is about creating new feature to solve problem from pre-existing knowledge or intution. ex: if house plot have length and width, we can add 1 extra feature area if this feature is effecting price. 

<img width="607" alt="image" src="https://github.com/user-attachments/assets/fa5a95f8-93ce-402a-b4e2-e76088489f4b" />

#### Polynomial Regression (feature engineering): 

Adding to feature engineering we can add power to the f() function to get better house pricings. ex: adding house vsize in meter cube. 

<img width="617" alt="image" src="https://github.com/user-attachments/assets/817a1812-2650-4b24-bc1e-763b2e65c8c7" />

or, we can also use sq. root instead of sq. 

<img width="608" alt="image" src="https://github.com/user-attachments/assets/0387a0f6-fa5b-4e81-b823-4867992910fa" />

This depends on the model ,which one to use. Based on that we decide which feature suits better.

#### Logistic regression: 
For predictive analysis. Used more than linear regression. Outputs between 0 and 1 with formula <img width="132" alt="image" src="https://github.com/user-attachments/assets/e39b24b4-ca38-4925-a169-c8399d6719ea" />   called **sigmoid function**.

<img width="575" alt="image" src="https://github.com/user-attachments/assets/6b4eefc2-9896-4efc-81c9-6b88ad436a3e" />

##### Decision Boundaries: 
These are the boundaries outside and inside which True(1) and False(0) outputs exists respectively. Here, f(x)=g(z) .

For non cancer tumor vs cancer tumor ex: Consider w1 and w2 as 1 and b as -3.
<img width="565" alt="image" src="https://github.com/user-attachments/assets/8526c487-3936-4a59-a639-389ce063fec0" />

These decision boyndaries can be linear ,circular, oval, any shape curves(non linear) with more powers of x.

<img width="581" alt="image" src="https://github.com/user-attachments/assets/c234c478-3551-4beb-8001-2a66be060cea" />
<img width="570" alt="image" src="https://github.com/user-attachments/assets/93b55ef5-088e-4fa0-96d7-8a8ff7e16bf9" />

##### Cost Function for Logistic regression: 
With cost function calculation we can choose better parameters. 

In tumor Training example ,where there are 1,...,m training examples and 1,....,n training features(tumor size, patient's age, etc). The output/Target can be 0 and 1 only. 
The Logistic regression is defined by equation: <img width="182" alt="image" src="https://github.com/user-attachments/assets/028ca048-64e9-48d2-bbcb-eb4aa1338eb6" />

In Linear regression the Loss function curve is convex and given by f(x)=w.x+b  and J(vector w,b) is <img width="302" alt="image" src="https://github.com/user-attachments/assets/ef0d25d8-aef0-415a-af33-76784d2ad12f" />   The inside summation part is written as L(f(x,y))  <img width="334" alt="image" src="https://github.com/user-attachments/assets/a9b089e8-5395-4d8f-8e59-09e3b4365dee" />  and the loss function curve is not convex. 

<img width="608" alt="image" src="https://github.com/user-attachments/assets/8024adfc-dd4d-46f3-b711-56065cf86ab3" />

Logistic loss function when y=0 and when y=1:

<img width="493" alt="image" src="https://github.com/user-attachments/assets/8a3df1a0-f4bc-43ed-bb27-bd5dee96087e" />

f(x) output is always between 0 and 1. 
* Squared error cost function is not suitable for logistic regression.

##### Simplified loss function for logistic regression: 
If y is only 1 or 0. Then the simplified loss function is: 
<img width="593" alt="image" src="https://github.com/user-attachments/assets/ca015d87-9332-4ee4-b077-104d9ea61fd2" />

Simplified cost function for logistic regression: 
<img width="612" alt="image" src="https://github.com/user-attachments/assets/0704b336-fae1-466c-81b4-21002f51ccca" />

this is convex with single global minimum. 

To calculate the gradient Dedcent, we need to find the multiple new w&b parameters of cost j() function repetedly to keep minimising the J(w,b) cost function.  
<img width="587" alt="image" src="https://github.com/user-attachments/assets/9759e712-fc3b-4883-9251-533b1a497911" />

here `Wj` is the number of features. <img width="26" alt="image" src="https://github.com/user-attachments/assets/6d3c6c95-d7b3-4d34-8ebb-16cdb41fcf41" /> is the Jth feature of the of the training example i. 

<img width="615" alt="image" src="https://github.com/user-attachments/assets/24a77989-6a22-47a9-82c6-3c73f9e19753" />

Here, need to simultaneously calculate the right hand side values and update the left hand side values and keep updating. 

same can be written as below and then updated in the logistic regresion function f(x) of w,b. 

<img width="529" alt="image" src="https://github.com/user-attachments/assets/8786a6f4-6636-4e1c-a0d2-76c06eb157ea" />

Further for **Convergence** and **Vectorization** and **feature scaling** in Gradient descent, can be done same way it is done in linear equation.   


### Over Fitting: 
**Bias** in ML can be due to under-fitting. Where model is not trained much over some data or because ML have some pre-learning or pre-conceptions (bias) over a set of data. 
**Generalization** is training a model for knowing general idea so that for untrained data also model can predict the general outputs. This is possible with model with average training data with x-y equations not linear equation ,may be quadratic equation.  
If the model is trained on multiple data sets(multi-polynomial equation) then model can become the overtrained model called **over-fitting** and not good for a general prediction.  Over-fitted model can give wrong outputs. This type of model is called to have **high variance** as with very small change in single input in training data can make the very different predictions for multiple ML engineers.   

So, overfit model have high variance and underfit model have high bias. 

<img width="612" alt="image" src="https://github.com/user-attachments/assets/1a4a7947-4562-4780-a1eb-c27b07d52ac0" />

similarly:
<img width="602" alt="image" src="https://github.com/user-attachments/assets/9dcdf4d4-d5a6-4df0-bed5-eda28f33caab" />

#### Adressing overfitting:
Select only needed/most important features.
Train model with more data: best way
Regularization: model is trained to self make some `w` values of the polynomial minimal or 0 (ignoring the training data feature) and see if the curve fits the training data better without it. 

<img width="602" alt="image" src="https://github.com/user-attachments/assets/0ecc6e42-484c-4a85-a56c-012de2a979be" />

Adding all the small `w` values to make a number for simplicity we use Lambda (always lambda > 0) called regularization parameter. 

<img width="549" alt="image" src="https://github.com/user-attachments/assets/e4b30f78-d56c-40b4-99b8-647578fe6a6b" />

first part is called `mean squared error` and other called `regularization term`. Here, we keep Wj small and choose Lambda to balance to banance between mean squared error and  regularization term. 
So, If Lambda = 0. regularization term = 0 and J(w,b) is only mean squared error which gets overfitted curve. 
and, If Lambda is very very high number eg. 10^10. Then, since regularization term is very big number, mean squared error nead to belance it out and learning algorithm will choose w1,w2,w3,w4.... to be very very near to zero. so, f(x) becomes equal b and gives a linear graph and cause underfitting. 

<img width="553" alt="image" src="https://github.com/user-attachments/assets/755f1217-4c56-46db-9c46-f9c32f0c9765" />

So, need to choose good values of Lambda. 


#### Regularized Linear Regression:

Keep updating w for j=1 to n and updating b with updated w. 

<img width="639" alt="image" src="https://github.com/user-attachments/assets/c1008e6b-7cb2-4e62-ac89-b5bc6acaeb55" />

Formula to calculate the gradient descent :
<img width="484" alt="image" src="https://github.com/user-attachments/assets/e4b39a44-9b78-4b5b-bd3c-6ae79c738ff7" />

COst function for logistic regression was: 

<img width="752" alt="image" src="https://github.com/user-attachments/assets/ada14ab7-863d-4d2c-a66b-1b42edf827f1" />

Adding regularization to it: 

<img width="926" alt="image" src="https://github.com/user-attachments/assets/05411f84-af69-4db5-a0c7-157459334031" />

and, by resetting the w and b parameters here to get the least J(w,b) and gradient descent. here, we get the high order polynomial but it is not overfitting due to generalization. 
 

### Can set Gemini GPT token and search here: 
1. https://console.cloud.google.com/vertex-ai/studio/freeform?project=carbon-feat-280309
