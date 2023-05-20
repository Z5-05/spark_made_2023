# spark_made_2023
### Homework №1 for "Spark in ML" course
#### Logistic regression with Apache Spark
Run tests with ```sbt test```

![Example of test successful work:](https://github.com/Z5-05/spark_made_2023/blob/main/pics/tests_success.png)

Some detailes:
1. Test data was created using python. There were two datasets: crear with blobs, which every Logistic regression can classifier on two groupes and noisy - dataset with linear trend and noise. WIth last only metric testing available, because weights can be different;
2. Ideal variants of metrics and weights got with using `sklearn.linear_model.LogisticRegression`;
3. Model has `transform` and `predict` methods with probabilities and predicted labels.