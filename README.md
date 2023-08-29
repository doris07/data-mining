# Foundations and Applications of Data Mining

### Situation:
Trying to build 3 different types of recommendation systems, which using yelp dataset to predict the ratings/stars for given user ids and business ids.

### Tasks: 
My goal is building Item-Based CF recommendation system with Pearson similarity, Model-based, and Hybrid recommendation system. Additionaly, I would use the error distribution of computing the RMSE(Root Mean Squared Error) to evaluate your recommendation systems.

### Requirement of this project:
1. Use Python to implement all tasks
2. Required to only use the Spark RDD
3. Programming Environment: Python 3.6, JDK 1.8, Scala 2.12, and Spark 3.1.2
4. using RMSE to evaluate the recommendation system, has to beat 0.9800( which mine work achieved 0.9646)

### Dataset:
I randomly took 60% of the data as the training dataset, 20% of the data as the validation dataset, and 20% of the data as the testing dataset.
1. yelp_train.csv: the training data, which only include the columns: user_id, business_id, and stars.
2. yelp_val.csv: the validation data, which are in the same format as training data.

### Recommendation System that I am going to implement:
1.  Item-based CF recommendation system with Pearson similarity
2.  Model-based recommendation system
3.  Hybrid recommendation system

### Process:
**Summary:** To accomplish this goal, I embarked on a multifaceted approach. Firstly, for the Item-Based Collaborative Filtering (CF) recommendation system, I delved deep into the Yelp dataset, conducting extensive data preprocessing to ensure data quality. I then implemented the Pearson similarity algorithm, which involved rigorous testing and fine-tuning to optimize its performance. This step was particularly challenging as it required handling a large volume of user-item interactions effectively.
  Next, I tackled the Model-Based recommendation system. Here, I employed advanced machine learning techniques, including matrix factorization and gradient descent, to build a predictive model. This model was designed to capture complex patterns within the data and make accurate recommendations based on user behavior.
  In parallel, I worked on developing a Hybrid recommendation system that integrated the strengths of both the Item-Based CF and Model-Based approaches. This involved creating a framework that seamlessly combined the outputs of these two recommendation engines. I implemented user-specific weighting to ensure that the hybrid system delivered personalized recommendations.
  1. **Item-based CF recommendation system with Pearson similarity:** 
  2. **Model-based recommendation system:** use XGBregressor (a regressor based on Decision Tree) to train a model and choose useful features from the provided extra datasets
  3. **Hybrid recommendation system:** the CF focuses on the neighbors of the item and the model-based RS focuses on the user and items themselves, combine them together as a weighted average.

