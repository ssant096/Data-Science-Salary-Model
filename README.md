# Data Science Salary Prediction
Final project for Introduction to Data Science class. Used data collected from aijobs.net to made a 10 fold CV model for evaluating the most important variables in determining the salary of a data scientist.

# Introduction and Project Descripiton
Our group wanted to study something relevant, so we choose to focus on what variables affect the salary of data scientists the most. We found a
dataset on Kaggle which contained 3,755 rows and 11 columns of data on data scientist salaries. The data was not very large, but came from a
site called ai-jobs.net which is a recruiting site for jobs in data science and related fields, so we felt this dataset was appropriate to use for our
project.
Our research question was “What are the factors that lead to the highest salary for a data scientist?” We wanted to find out the most important
factors for attaining the highest possible salary as a data scientist. We would look at factors such as experience, job title, employee location,
company size and location. We would then look to determine which factor impacts salary the most, and if there are multiple factors, we will try to
find out which group of factors is the most important.  
  
Questions we would like to answer include:  
  
Are there multiple factors that affect the salary for a data scientist?  
If there are multiple factors, then what group of factors is the most valuable in increasing salary?  
In order of least to most impact, what are the factors?  
How strong is the correlation between each factor and salary?  
Are there correlations between variables?  
Are there any factors that lower salaries for data scientists and should be avoided?  
Are these factors different in different countries?  
If they are different for different countries, then we can find out the most important factors in certain countries.

# Data Exploration and Visualization
Sample of the Data:

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/34b3406d-781e-4652-ba96-0c950b60a27e">

Exploratory Data Analysis (boxplots):

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/75a0bed9-bd8b-4e5b-ad8c-60b10af96610">

From the above box plots, we can see that those with the title of applied scientist, working in a medium sized company, working full time, with
executive level experience, and working in person full time make the most money.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/b78ca790-ba3c-4011-a77e-8ea7e2cc2bbe">

From this bar graph, we can see that our data comes from the past 4 years and is fairly recent, so the information we get from our project will not
be outdated and might prove to be quite useful.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/bba269a4-aea7-436b-b06c-a91a821ce877">

We can also see that our data is normally distributed from this Q-Q plot.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/46dda8aa-0130-46cf-8f51-d02ed613f3ca">

Additionally our data contains no NA values.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/7697eb79-52bb-446a-8842-f08a1ed69453">

From this heat map we can see that countries in the southern hemisphere appear to make less than countries in the northern hemisphere, with the
exception of Australia, so this leads us to believe that location would be a good predictor of salary, which we would like to add to our analysis.

# Data Analysis, Modeling and Predictions
For our first model, we wanted to use the data we had to see which variable would have the most impact on data scientists’ salaries so we decided
to use a linear model to analyze this. However, our original data contained mostly categorical data, so we converted some of the categorical data
into numerical data so we could use a linear model to model our data.  

We chose to leave out job titles as a predictor since we did not feel there was an appropriate order for job titles as data scientist, data engineer
and machine learning engineer are all very similar and it is hard to decide what order they should be placed in.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/89ea1cf6-70af-41f4-8562-c7e07afe45bf">

From our initial lm model, using experience level, employment type, and company size as predictors, we got an adjusted R^2 value of 0.1988,
which is a good start.  
After visualizing our heat map, we originally wanted to create models of countries with the top 5 highest mean salaries, however we were not able
to since we did not have enough data for Israel, Puerto Rico, or Russia.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/d8fa96d1-78ea-42cd-8c19-b95a9c16f860">

So we decided to make models with the top 3 countries with the we had the most data on, which is US, Britain, and Canada. Hopefully this will
improve our adjusted R^2 value, since by focusing on a specific country, we can eliminate location as a variable which will effect salary.
Additionally we can also compare the effect of each variable between countries to see if certain countries value certain qualities more than other
countries.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/90707ec1-98d0-4c5b-8ce7-8101389f98db">

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/ce14d6f3-e775-4ca4-bb14-b2a78cf62b76">

We wanted to conduct another round of visualizations for just the US data to see if there were any changes, and everything appeared to be the
same as in our original plots.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/16213723-91e4-4639-b6cd-3f0a77db96ac">

We did the same for Britain and it followed a similar trend, but for employment type, we only had data on full-time employment in Britain.

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/93897123-40c1-485a-a8c3-ac7e853c38ae">

In Canada we do see different results than what we saw in the previous 2 countries. Machine learning software engineer is the highest paid title,
and individuals with senior level experience actually make more than executive level, and those that work remote full time make the most and
those that work 100% in person make the least.  
  
Now we will make linear models for each country and see if we can improve our R^2 value.

### Linear Model for US data

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/57f3b003-3bac-4181-b4bd-c3e071bf7340">

Unfortunately our R^2 value actually decreased, so it would seem that country location is not a good predictor of salary for data scientists.

### Linear Model for Britain data

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/7e7a236a-bc39-47bb-8a43-798cff0e3265">

Our R^2 value for Britain is even lower than it was for the US data.

### Linear Model for Canada data

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/9848c313-1050-469e-b4df-3c6c75c1364e">

Our adjusted R^2 value for Canada is greater than it was for Britain, but it is still very low and lower than it was for the US.
Our team believes that our R^2 value decreased when looking at specific countries because there is more deviation in salaries inside a country
than between countries.

# Model Evaluation and Validation
### 10 Fold CV for US

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/18c8bd52-32de-41b2-9963-655da758b025">

From the 10 fold cross validation results of our US data it is clear that experience level is the most important predictor of salary for data scientists
in the US, as it has the lowest p-value. The next most important predictor is company size, and the least important predictor is employment type.

### 10 Fold CV for Britain

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/5a8bbe4a-7af9-4039-a480-48b451a68c57">

From the 10 fold cross validation results of our Britain data, experience level is the only significant variable in Britain.

### 10 Fold CV for Canada

<img width="750" alt="image" src="https://github.com/ssant096/Data-Science-Salary-Prediction/assets/102336530/e5865399-376c-453b-9f02-ae0a641b6f27">

From the 10 fold cross validation results of our Canada data, experience level is the only significant variable in Canada as well.
Looking at the MSE values, since the US has the lowest normalized MSE, out of Canada and Britain, it would seem that our model fits US data the
best.

# Conclusions
Surprisingly enough we found that country location alone is not a good predictor of salaries for data scientists. We also concluded that overall the
most important predictors for salary was overwhelmingly solely experience level. In the US, the next most important variables were company size
then employment type, but for the other countries we looked at, experience level was the only significant predictor. Finally, as for limitations to our
data, we were not able to take job title into consideration for our analysis, which may have been beneficial in increasing our models’ R^2 value.
Our R^2 values were not high for any of our models so there is likely important predictors we are missing from our data such as location within a
country. Lastly, we did not have much data for countries outside of the US, so our results may not be very accurate and representative of those
countries.

# Author Contributions
Shan Santhakumar - Helped make graphs for exploratory data analysis (heatmap) and worked on 10 fold CV for US data. 
Eden Fraczkiewicz - Worked on researching our data, helped make linear models for our top 3 countries with the most data and the 10 fold CV for
Canada data. 
Luis Colin Cornejo - Made graphs for exploratory data analysis (boxplots) and helped make linear models for our top 3 countries with the most
data. 
Lorenzo Gonzalez - Worked on the initial data linear model and helped make linear models for our top 3 countries with the most data. 
Vishal Gondi - Helped make graphs for exploratory data analysis (bargraphs), made graphs for summary statistics and created 10 fold CV for
Britain data. 
# Data Availability
https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023?select=ds_salaries.csv
