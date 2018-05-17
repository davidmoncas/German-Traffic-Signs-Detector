# German-Traffic-Signs-Detector
## Deep Learning Challenge for Kiwi Campus

![Kiwi car](https://d3ngl870p61wq1.cloudfront.net/5ade592c5c36ce000b7b98bf/img/challengekiwibot%20sin%20fondo%202.png)

Kiwi Campus Inc, the biggest delivery company with robots in Sillicon Valley presented this challenge of robotics and machine learning. This repository is my solution to the challenge of Machine Learning.

### Description of the challenge

The machine learning challenge consist in create, train and test three image classification models: 

- Model 1: A logistic regression model using only Scikit-learn.
- Model 2: A logistic regression model using only Tensorflow
- Model 3: A convolutional neural netwroks following this paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf Using only tensorflow

For this three task, we have to use a Dataset called German Traffic Signs Dataset ( http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset ) , downloading and preparing the data using code is also part of the challenge.

### My solution

I could only develop the first two models, both are functional, capable of infer new images. I obtained accuracies arround (75%-85%).
For the challenge, we must create an app.py file containing all the code that could be accesed directly from the command prompt, the commands for the aplication are:

#### train:
```
python app.py train -m [model] -d [directory]
```
where [model] can be either **model1** or **model2** and [directory] should be **images/train** , this command will train the specified model and will create a file inside ./models/[model]/saved/ containing the saved model for use it later.

#### test
```
python app.py test -m [model] -d [directory]
```
where [model] can be either **model1** or **model2** and [directory] should be **images/test**, this command should return the accuracy of the model, in model1 also creates a report file with some metrics in the folder ./reports.

#### infer
```
python app.py infer -m [model] -d [directory]
```
where [model] can be either **model1** or **model2** and [directory] should be **images/user**, this command should open windows with plots of the images in the folder as well as the predicted class and title by the specified model.

![window with predictions](https://preview.ibb.co/mQJyty/Sin_t_tulo.jpg)
