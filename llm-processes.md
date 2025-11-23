# Keywords: 


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




# Processes:


## Neurons and how it learns: 

### neurons: 
These are mathamatical functions that work together to solve a problem. A neuron just hold a value between 0 and 1. This number inside neuron is called **activation**. Some groups sets firing cause some other groups to fire. 


![image](https://github.com/user-attachments/assets/566eadae-4a26-4cee-be07-88b8209186ef)

Here, splited parts of the input image is checked in second layer. These splited parts may look like a straight vertical or tilted line(|or/) and a straight horozontal line(-) to make '7'. 
But, how 2nd layer will know which is vertical line or horizontal line ? Basically how it will find the parts of the image ? It is calculated by **weights**


### Weights: 
These are numbers assigned as strength between neurons connection. Basically it is number between -n and +n (different for different models). Higher weight will confirm better connection and vise versa. Neuron data with Higher weight will have more chance of passing to next neuron layer for further processing. 

![image](https://github.com/user-attachments/assets/da0dff38-e94c-46aa-8501-93fed23a8975)

It is called **weighted sum**. Here, the w1,w2,w3,w4,......wn are weights and a1,a2,a3,.......an are **activations**. When, we calculate all the weights in a matrix of pixels below. It is seen that the most positive weights for part(-) of input(7) is at the correct position.

Here the weighted sum is equal to the value of pixels of the needed part. 

![image](https://github.com/user-attachments/assets/1f6279e2-aff0-4cf2-a2ca-81b76048c425)

The activation range depends on the activation function used. A common function used called **sigmoid function or logistic curve** does this conversion from big scale (-n to +n) to 0 to 1 scale value update. More positive values gets near to 1.

![image](https://github.com/user-attachments/assets/0df9289a-29d1-4df3-8d5c-0a4625f4dbc6)



## Weight Quantization: 

**Size** of a model depends on model weight size and data type's precision. 
To save memory weights can be stored with lower precision data types by process known as quantization. There are 2 main major mathods : 

#### Post-Training Quantization (PTQ): 
weights of an already trained model are converted to lower precision without any retraining. It is easier todo and causes potential performance degradation.

#### Quantization-Aware Training (QAT): 
weight conversion happens in process during the pre-training or fine-tuning stage ,resulting in enhanced model performance. It is computationally expensive and demands **representative** (data that has the features or data points that the application is designed to predict or classify) **training data**. 

Mostly Floating point data are used due to precision. Typically floating point numbers uses n bits to store a numerical value. These n bits are partitioned into 3 distinct components:
`sign`: +ve or -ve number. It uses one bit where 0 indicates a positive number and 1 signals a negative number.

`exponent` : The exponent is a segment of bits that represents the power to which the base (usually 2 in binary representation) is raised. The exponent can also be positive or negative, allowing the number to represent very large or very small values.

`Significand/Mantissa` : The remaining bits are used to store the significand or mantissa. This represents the significant digits of the number. The precision of the number heavily depends on the length of the significand.

Formula used for this representation is : $(-1)^{sign} * (base)^{exponent}$ * significand


#### Most commonly used data types in DL:  `float32 (FP32), float16 (FP16), and bfloat16 (BF16/brain floating point)`. 

Both float16 and bfloat16 differs in precision. 

Float16: FP16 uses 16 bits to store a number. It uses 1 bit for the sign, 5 bits for the exponent, and 10 bits for the mantissa. More memory-efficient which accelerates computations but less accuracy.

bFloat16: also 16bit. It uses 1 sign bit, 8 exponent bits, and 7 mantissa bits. This is more accurate.

FP32(full precision): one bit for the sign, eight for the exponent, and the remaining 23 for the significand. It provides a high degree of precision, the downside of FP32 is its high computational and memory footprint.
![image](https://github.com/user-attachments/assets/7adf9588-0e78-4894-9647-1acb4279a31a)

#### 8-bit Quantization: 
2 major ways to 8-bit quantization. 
- absolute maximum (absmax) quantization: the original number is divided by the absolute maximum value of the tensor and multiplied by a scaling factor (127) to map inputs into the range [-127, 127].  To retrieve the original FP16 values, the INT8 number is divided by the quantization factor, as there is some loss in precision due to rounding.
  ex: If we have an absolution maximum value of 3.2. A weight of 0.1 would be quantized to `round(0.1*(127/3.2)) = 4`. If we dequantize it, we would get 4*(3.2/127)=1.008. So, error =     `0.008`.Can be done with python tourch library.
  
![image](https://github.com/user-attachments/assets/26984fc3-a785-43bc-a0b3-db83cbcc7deb)

- asymmetric one with zero-point quantization: This uses a scale factor and a zeropoint. Scale is total range of values(255)  divided by the difference between the maximum and minimum values.
![image](https://github.com/user-attachments/assets/46cf1201-b436-4f0e-b31b-2b5d48aa9e56)

These variables are used to quantize or dequantize weights.

![image](https://github.com/user-attachments/assets/705d1312-92a3-41a0-9e14-61b5bd513abc)



### Activation:
Activation of the neurons are bascically a measure of how positive the relevant weighted sum is. More activation makes the neuron more light up. Activations are not directly controlled for neural network trainings. We train weights and biases only. Activations are influenced only. 
Activation functions determine the output of a neuron based on its weighted inputs.

![image](https://github.com/user-attachments/assets/e7aeb112-28aa-418f-a6ff-c93c70086d62)

In cases there  may be condition to not light up pixel when weighted sum < 10. This is adding a **bias** for inactivity. 

![image](https://github.com/user-attachments/assets/2467496f-fc28-4eaf-bce9-0050413bd3f0)
![image](https://github.com/user-attachments/assets/d86447c9-cabe-43ed-8185-68fc5707e186)

![image](https://github.com/user-attachments/assets/2ab5d8f3-7678-4684-9a5d-ea32db974ff8)

Here, 1st layer have 784 neurons, 2nd layer have 16 neurons and 3rd have 16 neurons and 4th have 10 neurons. Every neuron have it's own biases. Here, it is 13,002
weights and biases. Here, 1st layer neurons dosen't have any biases as they only receive raw data so, it is not 784+16+16+10 for bias adding. 
The secons layer is expected to pick up on edges and 3rd layer picks up the patterns. 

![image](https://github.com/user-attachments/assets/75a29064-ccde-4303-acb4-a405a76cb8e0)
Note: Neurons that work together, fire together.
Target is to make the weights for expected output neuron more (to make connected neuron more active) than the non-expected (incorrect) output neuron.  


### Deep Learning : 
Finding the correct weights and biases by the incorrect outputs comparision with correct output.

Weights ,Activations, biases are passed in the sigmoid function to calculate the forward transition of activations from one to next layer.  

![image](https://github.com/user-attachments/assets/c63bdce7-3137-4016-9ffa-3d840521bef0)

Sigmoid function is old schoold now and new one is ReLU(a) = max(0,a) where a is activation. ReLU is Rectified Linear Unit. 

### Four types of machine learning: supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning

### Supervised Learning: 
* A type of machine learning where the model is trained on **labeled data**. Here, model is trained to get a generalized answer for an new input data.  
* More accurate than un-supervised learning. Needs human intervention.
* Supervised learning models are able to predict future.
* More commenly used.
* These can be used with non-labled data as input.
* Can find hidden pattern in data which un-supervised learning cannot find.
* $\hat{y}$  is symbol for prediction on y-axis for input x to the model.
* y is the actual value on y-axis for the training data x. and  $\hat{y}$ is prediction value after model training.
#### Linear regression:   
* If curve is in straight line(linear regression) `f{w,b}(X) = wX + b` is the function (hypothesis) here. Here, w and b are adjusted to find the correct function which can make predictions. w and b are called parameters or co-efficients or weights. 
* In Linear regression, 1 input variable is provided for the prediction outcome. 
* b is the height on y-axis and w gives angle to x-axis.
  <img width="617" alt="image" src="https://github.com/user-attachments/assets/ef0320ad-9793-4712-8d61-22b605de83fa" />

  
<img width="53" alt="image" src="https://github.com/user-attachments/assets/d6bb8c17-4d5e-40f9-b95f-bbcc74beab16" /> <img width="176" alt="image" src="https://github.com/user-attachments/assets/45c373ad-8ce2-4066-9954-1c244b25fee5" />

#### Cost Function: 
The $\hat{y}$ is checked with y. i.e. diffrence between actual and predicted foundout to get error ($\hat{y}$  -  y).  This then squared and added upto number of training examples. This gives total error. So,it is divided by m to get avg error. This is divided by 2 for furthur usefullness in later calculations. 
<img width="379" alt="image" src="https://github.com/user-attachments/assets/55b35319-77c1-4af6-ae0a-e4e9eade6177" />

For diffrent values of input x and w ,we get different values of $\hat{y}$ (assuming b=0). Calculate j(w) vs w graph. Then, find the w to get minimum j(w) possible. 

<img width="602" alt="image" src="https://github.com/user-attachments/assets/fb1ce8c7-c563-4455-bde6-cd0f2695f419" />

In more practical term we need to find values of w and b(where b!=0) to get the min j(w) possible. 

#### Gradient descent: 
It can be used to reduce any function not just linear regression. For Linear regression with sqared error cost function the 3D graph of j(w,b),w,b is always a bow or a hammock shape graph like: 

<img width="497" alt="image" src="https://github.com/user-attachments/assets/fc896a31-11bf-4f32-9f0a-56d28e1125a6" />

<img width="380" alt="image" src="https://github.com/user-attachments/assets/9b42f62d-7c10-4ab9-abaa-7b628886d254" />

But, for NL training normal graphs look like: 

<img width="824" alt="image" src="https://github.com/user-attachments/assets/5106a039-08ed-41b5-b0ff-72fcca001e0c" />

Going downhill from any one hilltop can have multiple ways to multiple local minimum.

<img width="950" alt="image" src="https://github.com/user-attachments/assets/a0f5ccd8-ca45-424b-82ff-4b62baeba127" />

When Gradient Descent is calculated from i=1 to i=m(a set or batch) .It is called a Batch gradient descent. 

#### Gradient Descent algorithm:

<img width="307" alt="image" src="https://github.com/user-attachments/assets/7c0dd2ec-cffc-473e-a833-89dcaf626d8e" />

Here, 
alpha is called learning rate. It is between 0 & 1. 
δ/δw relates to direction and magnitude of the steps to take for going downhill.
If alpha(learning rate) is too small calculation of 'w' will take very tiny tiny steps to go downhill and will take more time. 
If alpha(learning rate) is too large, the calculation of 'w' can surpass the local minimum and go beyond the needed value. Never finding the local minimum.
In calculating local minimum, the value of cost function J(w) will be largest for the 1st value if the slope is decreasing. Later the cost function value will keep minimising to smaller and even smaller until it reaches the local minimum. And , it works with any cost function J() ,not just mean squared error cost function for linear regression. 

<img width="436" alt="image" src="https://github.com/user-attachments/assets/97b76243-f6e3-403f-8f41-48aae245fa49" />

#### Linear regression algorithm: 






b is found with: 

<img width="296" alt="image" src="https://github.com/user-attachments/assets/46132fb9-be23-4812-b2e5-602e979b6e9b" />

We need to **keep changing the w and b simultaneously** to find min of w&b for finding the local minimum. Here, the value of 'w' in formula for 'b'is not updated with equation for 'w' rather it is previous 'w'. So, calculation of w&b happens simultaneously here. Like this we get to a point called **Convergence** after this model's prediction stops improving and the error becomes almost constant.












### Unsupervised Learning: 
* A type of machine learning where the model is trained on unlabeled data to find hidden patterns. 
* Un-supervised learning models are less accurate than supervised learning and these models cannot make future predictions.
* These can only find and group data together.  
* Can process larger volumes of data but cannot be trusted.  

### Semi- supervised learning: 
* Can be used for both labled and unlabled data for training.
* Most useful when we need only few supervised data training and later data can be easily predicted eg: medical x-ray examination model.

### Reinforcement Learning: 


### Training steps :
Training starts with input number to a untrained model and after a wrong output. The model should get error response as penalty. Here, Penalty is output of a cost function which returns expected data. Mathamatically:
It is addition of the squares of diffrences between untrained model output and the expected output. 

![image](https://github.com/user-attachments/assets/df4eb7e2-e1c0-4940-bc7a-b99b0cba0d8c)
This sum is small when model identifies the input image correctly and the sum is bigger when model cannot identifies the image correctly. 

Here, average of these penalty (sum) is calculated and this is the updated to network as error(How bad the trained model should feel). 

Now, Need to change these weights and biases to fix the errors. So, We try to find the multiple Local minimums and then the Global minimum from these with help of the function. ex, ideally a function which takes single input and gives single output. here, need to find the input which gives minimum output for the function with calculas. 

![image](https://github.com/user-attachments/assets/71114fe4-f590-45ea-bc07-4393b1265a05)

But, generally we don't have single input ideal condition. We have multiple inputs and multiple outputs. so, the global minimum is found mathamatically with gradient descent of function. This gradient is direction of the steepest accent(increase the function output) most quickly. And, going opposite is the steepest descent(decrease the function output) most quickly. The length of the vector signifies how steep is the sopes are.

![image](https://github.com/user-attachments/assets/e085aa4a-5425-49fd-a1f7-2e6665c1bef8)

getting vectors at all the points of different directions and lengths will make it easier to idetify the needed point. 

![image](https://github.com/user-attachments/assets/676b824f-7b71-4cf8-935d-039790c5c91a)

This is for 1 x-y plane only. so, keep repeating it and keep going downhill to get vectors of different planes.

![image](https://github.com/user-attachments/assets/aeb8c10b-5515-426a-bcec-68dd0b39a58f)

The negative gradient of the cost function is just a vector(direction). This is going to cause the most decrease to the cost function. 
Minimizing the avg of the training data is done here to actual output keeps getting nearer to the expected output.

### Gradient Descent: 
Machine learning is 'minimizing the cost function'. So. Neurons have contineously changing activations(between 0 to 1) rather 0(Inactive) or 1(Active). 
This process of contineously finding the descending vector is called gradient descent. 

Negative and positive in gradient represents up and down and magnitude tells which change(weights i.e. Neuron's connections) matters more and which matters less. 

![image](https://github.com/user-attachments/assets/58f230d3-4b30-4824-aad0-fe2f215eb39b)

#### Gradient descent Algorithm and it's varients: 

#### Stochastic Gradient Descent (SGD): 

#### Mini-Batch Gradient Descent with Python: 

#### Optimization techniques for Gradient Descent:

#### Introduction to Momentum-based Gradient Optimizer:

#### 



### Back propagation: 
Each of the neurons here have its own thoughts in this 2nd to last layer. And, we want all other than needed neuron (max positive neuron) to be less active. 
![image](https://github.com/user-attachments/assets/ecbde5f1-ca16-4d15-aa8a-00a1b1c66646)  

So, the thoughts(weights) of all the outputs neurons are added to see the list of +ve and -ve weights which should happen to the 2nd last layer.

![image](https://github.com/user-attachments/assets/62256e2a-6ec0-4a56-8bbb-aa91d31e24ca)

Once we have these, knowing which one should be stronger weight ,we can back propagate and update weights and biases and moving to the initial neural layer. And, same back propagation is used for multiple training examples. 

Then, average of all neuron weights are found for different training data. This collection is negative gradient of the cost function.  

![image](https://github.com/user-attachments/assets/a4b4f274-df72-44a9-8dfb-ab8e5e25040b)

### Stochastic gradient descent:
Gradient descent is very slow and computatinally difficient. So, in Stochastic gradient descent training the training data is divided into multiple datas. Each bach of these data is fed up one by one. We find gradient descent of each of these training data. 

### Backpropagtion Calculas:
Consider connection between 2 nodes ((last neuron) and (2nd last neuron)) namely a(L) and a(L-1) by a single neural connection. Consider the desired output to be y with value 1. 
Here, $(a(L)-y)^{2}$ is cost c. 
also, last neuron activation is determinded by previous neuron activation multiplied by weights plus some bias. 
`a(L) = σ (w(L)*a(L-1)+b(L))`
* Note: here, (L) and (L-1) means last and second last neurons.
So, calculating `w(L)*a(L-1)+b(L)` and y will give the function cost.

 Think these on a number line: adjusting w(L) will adjust a(L) value to some number in a number line. and a(L) adjustment will adjust c in number line. Saying diffrently : it is derivative of c  w.r.t.  w(L)   i.e.  `δ(c)/δw(L)`.  
To calculate : 
```
δ(c)            δ( w(L)*a(L-1)+b(L) )         δ( σ (w(L)*a(L-1)+b(L)) )                              δ c
---------  =   ---------------------------  x  -----------------------------------  x  -----------------------------------
δw(L)                  δ(w(L))                          δ( w(L)*a(L-1)+b(L) )             δ( σ (w(L)*a(L-1)+b(L)) )   
```
![image](https://github.com/user-attachments/assets/a11528af-f3a9-4b6e-bc6c-aeba2ce31119)
```
  δ(c)  
--------- = 2 (a(L)-y)
  δ(a(L))  
```
This means it is 2 times the diffrence between netwotrk's output(a(L)) and what we wanted to be y.
```
  δa(L)         
----------  =      σ‘(z(L))   i.e. sigmoid function
  δz(L)             
```
```
  δz(L)         
----------  =      a(L-1)   # so the amount weight δw(L) can influence the last layer is ,depends how strong the previous neuron is.
  δw(L)             
```

The big calculation of δ(c)/δ(w(L)) can be written as below (for a multi layer cum multi neuron example) :

![image](https://github.com/user-attachments/assets/8ad1f9e5-edd0-4d23-b7bb-1871a90e3bc6)

This is just one component of the gradient vector :∇c

![image](https://github.com/user-attachments/assets/3494aea2-ad3b-4aee-b8a9-7a7ef1089694)

from gradient vector image. To calculate  δ(b) for bias. just replace w(L) with b(L)
```
δ(c)            δ( w(L)*a(L-1)+b(L) )         δ( σ (w(L)*a(L-1)+b(L)) )                              δ c
---------  =   ---------------------------  x  -----------------------------------  x  -----------------------------------
δb(L)                  δ(b(L))                          δ( w(L)*a(L-1)+b(L) )             δ( σ (w(L)*a(L-1)+b(L)) )
```

This method is used for backpropagation to check how sensitive the cost funtion is to previous weights and biases. 

This same calculation is done on different neural layers each with multiple neurons: 

![image](https://github.com/user-attachments/assets/a751fe0b-6bac-4a3c-8ad5-8a2e74dfb685)

Here the only difference is: 1 single neuron from one layer influences all neurons of the other layer. so, should sum over layer L. 
These solves the way to find the bottom most point. 


### How LLM works: 


### Cons:
- After learning ,if machine is fed with a random non-sense (non-number) image. It confidantly gives output as a fixed number.
- Even it can recognizes the number. It cannot draw the numbers.
- Model is actually not lwarning anything it is actually memorizing from the input dataset and giving output data. ex: if a trained model is fed with new wrong data set repetedly. It starts recognizing with the wrong data.
    
