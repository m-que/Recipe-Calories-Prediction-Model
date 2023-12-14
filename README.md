# Recipe-Calories-Prediction-Model
**Author**: Maya Que

## Project Overview
This is a data science project on investigating the relationship between details of the recipe and the time it takes for completion, in order to construct a model that classifies recipes in terms of time consumption. This project is for DSC80 at UCSD. The dataset used is originally scrapped from [this source](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf). This project is for DSC80 at UCSD. The exploratory data analysis on this dataset can be found [here](https://m-que.github.io/Recipes-Ratings-Project/).

---
## Framing the Problem
With health and wellness becoming a growing trend, understanding the caloric content of a dish is not only pivotal for investigating the nutritional composition of a meal but also contributes to making informed and health-conscious dietary choices. The caloric content of a recipe can serve as a valuable tool for dieticians, chefs, home cooks, and more, providing users with the information needed to create meals that are not only delicious but also mindful of their nutritional impact. 

Recognizing this need, this project aims to develop a predictive regression model that uses a number of factors, such as the number of steps (`n_steps`), number of ingredients (`n_ingredients`), and nutritional information (`total fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `saturated fat (PDV)`, `carbohydrates (PDV)`) to predict the number of calories in a recipe. By doing so, this project seeks to equip individuals with the knowledge they need to make informed choices that align with their dietary preferences, health goals, and overall well-being. 

**Response Variable and Information at Time of Prediction:**
This project uses a regression model since the response variable, `calories` is a numeric variable that indicates the amount of calories in a recipe. At the time of prediction, the user would already have the ingredients, steps, and nutrition table in the recipe. Through these details on the recipe, one can reasonably count the number of ingredients and steps used in the recipe and estimate its various nutritional values from the nutrition table. Thus, all the features are available at the time of prediction and can be leveraged.

**Measuring Metrics:**
The model will be evaluated using R^2 and Root Mean Square Error (RMSE) as the key metrics. These metrics were chosen over others because R2 offers insight into the proportion of variance in calories that can be explained by the features of the model. Meanwhile, RMSE measures the average magnitude of errors between predicted and actual values, making it a suitable metric for assessing how well the model does in estimating calorie content. R2 is essential for understanding how well the model captures variability in the calorie content, while RMSE emphasizes accuracy by penalizing larger errors. Together, these metrics ensure that this model delivers accurate and reliable estimates, which is valuable since delivering accurate and consistent caloric information is crucial for users dedicated to making health-conscious choices.

---
## Baseline Model

**Features Used:**
For the baseline model, three quantitative features were used as described below:

- `n_steps` (discrete): The steps of recipes can be a useful feature in predicting the calories of a meal since typically the more steps a recipe has, the larger and richer the meal will be. These recipes are likely to have a higher calorie count. Since n_steps is a numerical feature, I apply standardization to it so the distribution will be more standardized. 

- `n_ingredients` (ordinal discrete): The number of ingredients can also be related to the calories of a meal, since the more ingredients a recipe has, the more nutrition it will contain. Thus, it can result in a higher calorie count. Since n_ingredients is a numerical feature, standardization is applied to ensure it is scaled properly

- `total fat (PDV)` (continuous): The total fat (PDV) column is also likely to predict the calories of a recipe since fat is known as one of the most concentrated sources of calories and is most present in calorie-dense foods. Total fat is also a numerical feature, so standardization is applied to this column as well. 


**Model Construction**:
With the above transformers for each feature, a ColumnTransformer is used to allocate StandardScaler transformer to the columns and combined with the LinearRegression model in one pipeline as the Baseline model. The linear regression model will predict the calories of a recipe with a polynomial of at most degree of one. 

**Model Performance**:
In our investigation, the datasets are separated into a training set and a test set with a 75:25 ratio. The baseline model is fit on the training set to test its performance on both seen and unseen data. The performance of the baseline model on the testing set (unseen data), as indicated by the R2 value, is 0.6621877442139298, implying that about 66.22% of the variability in the calorie count can be explained by the three features used in this model. Since this score is still a bit low, this suggests that these features may not be the most influential factors for predicting calorie content, or that the relationship between these features and calories is not completely linear, which was assumed by the linear regression model. 

In addition, the RMSE is 20.273383172586406. As the measure of differences between the predicted and actual values, this RMSE is still too high and not low enough to ensure an accurate model. 

A residual plot of the predicted values against the residuals is shown below. 

<iframe src="assets/fig1.html" width=800 height=600 frameBorder=0></iframe>

In a well-fitted model, residuals should be randomly scattered around the zero line to indicate the model captures variation in the data, which is not the case here. In this plot, many of the points are clustered on the left-hand side, indicating that most of the recipes have predicted calories in that region. As the plot moves from left to right, the broader scattering range of points seem to suggest that the model makes bigger mistakes for larger predicted values.

---
## Final Model
In the final model, in order to improve the accuracy of the model, five more quantitative features were added to the model. 
- `minutes` (continuous): The recipe minutes can be useful for predicting the calories of recipes since longer cooking times are usually related to more complex and larger meals, which is also linked to higher calorie counts. 
- `sugar (PDV)` (continuous): The amount of sugar is also an important factor that can affect the calculation of calories. Typically, recipes with higher sugar levels tend to also have a higher calorie count to the sugar's high energy content
- `sodium (PDV)` (continuous): High levels of sodium are generally associated with highly processed, high-calorie foods, so high sodium levels may indicate high calorie count. 
- `protein (PDV)` (continuous): Protein-rich ingredients, such as meat and dairy products, can potentially have more calories, and their presence can contribute significantly to the total calorie count of a recipe, 
- `carbohydrates (PDV)` (continuous): As a primary energy source, carbohydrates can contribute significantly to a recipe's calorie count. Dishes with high carbohydrate content, such as pasta and baked goods, tend to provide more calories. 

**Model Construction:**
The Lasso regression model is chosen to be the final model. Lasso, short for Least Absolute Shrinkage and Selection Operator, is a type of linear regression that performs shrinkage to shrink some coefficients to zero, effectively removing less important features and preventing overfitting. This makes the model simpler and interpretable, which is important for understanding the relationships between recipe details and its caloric content. To fit the Lasso model, StandardScaler transformer is applied to the new nutritional features, and QuantileTransformer is applied on `minutes`. This ensures that is dataset is appropriately standardized for the Lasso regression model since it will be shrinking coefficients and balancing the weights of the variables before training. 

**Choice of Hyperparameters:**
The final model is fit with alpha returned by GridSearchCV. The alpha for the model is important since it controls the amount of shrinkage, impacting the coefficients for the parameter. The higher the alpha value, the more the model would shrink the coefficient to make it less overfitted while keeping RMSE and R2 high with fewer features. The dataframe below reveals the validation accuracies with columns corresponding to hyperparameter combinations of 'alpha': [0.01, 0.1, 0.3, 0.5, 0.7, 1, 10, 100] and the rows corresponding to folds.

|   index |        0 |        1 |        2 |        3 |        4 |        5 |        6 |        7 |
|--------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
|       0 | 0.994527 | 0.994527 | 0.994528 | 0.994529 | 0.99453  | 0.994531 | 0.994555 | 0.994629 |
|       1 | 0.994369 | 0.994369 | 0.994369 | 0.994369 | 0.994368 | 0.994368 | 0.994348 | 0.994129 |
|       2 | 0.996329 | 0.996329 | 0.996329 | 0.99633  | 0.99633  | 0.99633  | 0.996326 | 0.996238 |
|       3 | 0.989093 | 0.989093 | 0.989092 | 0.989091 | 0.98909  | 0.989089 | 0.989059 | 0.988773 |
|       4 | 0.994128 | 0.994127 | 0.994127 | 0.994127 | 0.994127 | 0.994127 | 0.994117 | 0.993992 |

 After conducting GridSearch, we found that the best value for alpha is 0.3. 

**Model Performance:**
The R2 of the final model is 0.9972176806474021, and the RMSE is 6.107438781217623. The performance of the model is greatly improved by introducing new features that are related to caloric content and using the Lasso regression model instead. The performance on the test dataset is high, with the R2 being 99%, indicating our model can explain almost 100% variability in calories from its features. The RMSE is also low, demonstrating our model has minimal residuals and higher predictive accuracy. It should be noted that since calories are essentially calculated by taking the weighted sum of nutritional values, the high-performance score in our final model due to the added nutritional value features is reasonable. Compared to the baseline model, the R2 has increased, suggesting that the newly introduced features can better explain the calories of recipes. The smaller RMSE also indicates that the predicted values are closer to the actual values, so our model is better at predicting accuracy.

---
## Fairness Analysis

While a model has been constructed allowing prediction into the caloric content of recipes, observing the metrics of the model alone cannot be a good indicator as to whether the model is fair when placed in recipes with different numbers of ingredients. A way to test the fairness of the regression model is to take the difference between the predicted values' RMSE on recipes with fewer ingredients and recipes with greater ingredients for comparison. Hence, a permutation test is conducted on the RMSE of the model in predicting the calories of recipes with fewer ingredients and more ingredients to see if there is an actual bias in the model's performance in terms of ingredient level. 

**Group A and Group B:**
The two groups that we are going to use in the permutation test are extracted from the column `n_ingredients`, in which a Binarizer with a threshold of the median ingredients is used to separate the two groups into recipes with fewer ingredients and recipes with more ingredients. The median of `n_ingredients` is 9, which will serve as the threshold to classify recipes into two groups.  

*Group A*: Recipes with ingredients <= 9
*Group B*: Recipes with ingredients > 9

**Null Hypothesis**: Our model is fair. Its RMSE for Group A (fewer ingredients) and Group B (more ingredients) are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: Our model is unfair. Its RMSE for Group A (fewer ingredients) is different from Group B (more ingredients). 

**Significance Level**: The significance level set for this permutation test is 0.05, ensuring that a strong permutation test is performed with a high confidence level. 

**Evaluation Metric and Test Statistic**: Since RMSE is used as a performance metric in our final model, we continue to use this as the evaluation metric in our permutations. Since we hypothesize that the RMSE may be different for testing on recipes with fewer ingredients compared to recipes with more ingredients, the absolute difference in RMSE between the predictions of Group A and Group B is used as the test statistic. 

**P-Value:** To test the hypothesis, we run the permutation test by permuting the `n_ingredients` column and using our fitted model to predict corresponding calories. The RMSE differences in the two groups are calculated, and the process is repeated 1000 times. The p-value of the 1000 simulations is 0.001, which is less than the significance level of 0.05. Thus, we reject the null hypothesis that the model is fair. 

Below is a graphic visualization of the outcome of the permutation test. 

<iframe src="assets/fig2.html" width=800 height=600 frameBorder=0></iframe>
