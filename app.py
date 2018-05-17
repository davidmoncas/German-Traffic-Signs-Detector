#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NAME OF THE PROYECT:  German-Traffic-Signs-Detector
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PURPOSE OF THIS MODULE : 
#     		
#  
#  
# AUTHOR:   
#           David Montoya Castaño
# DATE:     
# 			17/05/2018
# UPDATE:  
#                     
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# /////////////////////////Packages\\\\\\\\\\\\\\\\\\\\\\\\\\\

import click
import numpy as np
import os , shutil
import zipfile
import urllib.request
import urllib
from random import shuffle
import pickle

from skimage import io,color
from skimage.transform import resize
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

import tensorflow as tf


# /////////////////////////CLI commands\\\\\\\\\\\\\\\\\\\\\\\\\\\

@click.group()
def cli():
    pass

#-----------------Download command---------------------
@cli.command()
def download():
	"""Download the data from the web, and split it into train and test"""
	downl()

#-----------------Train comand ------------------------		
@cli.command()
@click.option('-m',type=click.STRING,help='model you want to train')
@click.option('-d',type=click.STRING,help='folder path of the train data')
def train(m,d):
	""" Train a model using training data in specified folder"""
	if m=='model1':
		Xtrain,Ytrain=aquireData(d)
		train_model1(Xtrain,Ytrain)
	elif m=='model2':
		Xtrain,Ytrain=aquireData(d)
		train_model2(Xtrain,Ytrain)
	else:
		print(str(m)+' is not a valid model, try model1 or model2' )


#-------------------Test command--------------------------------
@cli.command()
@click.option('-m',type=click.STRING,help='model you want to test')
@click.option('-d',type=click.STRING,help='folder path of the test data')
def test(m,d):
	""" Test a model using test data in specified folder"""
	if m=='model1':
		Xtest,Ytest=aquireData(d)
		test_model1(Xtest,Ytest)
	elif m=='model2':
		Xtest,Ytest=aquireData(d)
		test_model2(Xtest,Ytest)
	else:
		print(str(m)+' is not a valid model, try model1 or model2' )


#----------------- infer command ------------------------------
@cli.command()
@click.option('-m',type=click.STRING,help='model you want to test')
@click.option('-d',type=click.STRING,help='folder path of the test data')
def infer(m,d):
	""" Predict new data in specified folder"""
	if m=='model1':
		Xinfer,originals=aquireImagesOnly(d)
		infer_model1(Xinfer,originals)
	elif m=='model2':
		Xinfer,originals=aquireImagesOnly(d)
		infer_model2(Xinfer,originals)
	else:
		print(str(m)+' is not a valid model, try model1 or model2' )



#/////////////////// Sub Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def aquireData(folder,ByW=False):
	"""	
		PURPOSE:		Gathers all the images of a folder and stores it in a vector
		DESCRIPTION: 	read images in ppm format, resize it to a fixed size, 
						convert it to greyscale (optional) and returns numpy array with all
						the images inside. Also read the first two characters of the title of 
						the file, which are the class of the image (00,01,02, etc.) and store
						the class in another numpy array
 		PARAMETERS
		folder:    		path of the folder with the images (string)
		ByW:			True if the images will be converted to grayscale (bool)

 		RESULT:
        images 		    numpy array with the images vectorized as well.
 		target			numpy array with the class of every image

	"""
	files=os.listdir(folder)
	target=[int(i[0:2]) for i in files]
	
	if ByW:
		images=[convertImageByW(io.imread(folder+'/'+i)) for i in files]

	else:
		images=[convertImageColor(io.imread(folder+'/'+i)) for i in files]

	print('successfully read '+ str(len(files)) + ' files from '+ folder)
	
	images=np.array(images)
	target=np.array(target)

	return images,target



def aquireImagesOnly(folder,ByW=False):
	"""	
		PURPOSE:		Gathers all the images of a folder and stores it in a vector
		DESCRIPTION: 	read images in ppm format, resize it to a fixed size, 
						convert it to greyscale (optional) and returns numpy array with all
						the images inside. the difference with aquireData() is that this function
						only reads the image without the class.
 		PARAMETERS
		folder:    		path of the folder with the images (string)
		ByW:			True if the images will be converted to grayscale (bool)

 		RESULT:
        images 		    numpy array with the images vectorized as well.
        originals		numpy array with the original images

	"""
	files=os.listdir(folder)
	
	if ByW:
		images=[convertImageByW(io.imread(folder+'/'+i)) for i in files]

	else:
		images=[convertImageColor(io.imread(folder+'/'+i)) for i in files]

	print('successfully read '+ str(len(files)) + ' files from '+ folder)
	
	originals=[io.imread(folder+'/'+i) for i in files]

	images=np.array(images)
	return images,originals




def convertImageByW(image,width=32,heigth=32):
	"""	
		PURPOSE:		Convert an image to a vector with a resized, greyscale, normalized image
		DESCRIPTION: 	took an image, resize it to a fixed size, convert it to black and white,
						then normalize all the elements using (X-mean(X))/std(X)

 		PARAMETERS
		image:    		Image that was read using io.imread 
		width:			width of the final image (int)
		heigth:			heigth of the final image (int)

 		RESULT:
        final_image	    numpy array with the image vectorized.
 
	"""
	img_byw=color.rgb2gray(image)

	# resize image to a normalized size:
	img_resized=resize(img_byw,(width,heigth),mode='constant')

	# Get the mean and standar deviation
	mean=np.mean(img_resized)
	std=np.std(img_resized)

	# normalize the image 
	img_normalized=(img_resized-mean)/std

	final_image=np.reshape(img_normalized,width*heigth)

	return final_image


def convertImageColor(image,width=32,heigth=32):
	"""	
		PURPOSE:		Convert an image to a vector with a resized, RGB, normalized image
		DESCRIPTION: 	took an image, resize it to a fixed size,
						then normalize all the elements using (X-mean(X))/std(X) for every channel

 		PARAMETERS
		image:    		Image that was read using io.imread 
		width:			width of the final image (int)
		heigth:			heigth of the final image (int)

 		RESULT:
        final_image	    numpy array with the image vectorized.
 
	"""
	img_resized=resize(image,(width,heigth),mode='constant')

	# Get the mean and standar deviation for every channel
	meanR=np.mean(img_resized[:,:,0])
	meanG=np.mean(img_resized[:,:,1])
	meanB=np.mean(img_resized[:,:,2])
	stdR=np.std(img_resized[:,:,0])
	stdG=np.std(img_resized[:,:,1])
	stdB=np.std(img_resized[:,:,2])
	# normalize the image 
	img_resized[:,:,0]=(img_resized[:,:,0]-meanR)/stdR
	img_resized[:,:,1]=(img_resized[:,:,1]-meanG)/stdG
	img_resized[:,:,2]=(img_resized[:,:,2]-meanB)/stdB

	final_image=np.reshape(img_resized,width*heigth*3)

	return final_image



def batching(X,Y,numBatch):
	"""	
		PURPOSE:		return a random sub-set of X and Y

 		PARAMETERS
		X:    			1D Array or list 
		Y:				1D array or list
		numBatch:		lenght of the final sub sets 
	

 		RESULT:
        XBatch	    	random sub-set of X
        yBatch 			random sub-set of Y
 
	"""
	indexes=np.array(range(len(X)))
	shuffle(indexes)
	xBatch=X[indexes[0:numBatch]]
	yBatch=Y[indexes[0:numBatch]]
	return xBatch,yBatch



def plot_predictions(originals,predictions):
	"""	
		PURPOSE:		Plot a series of images and its class (number and name)

 		PARAMETERS
		Originals:    	1D array with the images to plot 
		predictions:	1D array with the number of the class of each image

 		RESULT:
		none:			plots the images in matplotlib windows
 
	"""
	Nimages=len(predictions)
	number_couples=Nimages//2
	number_nons=Nimages%2

	for i in range(number_couples):
		f,axarr=plt.subplots(2)
		axarr[0].imshow(originals[2*i])
		axarr[1].imshow(originals[2*i+1])
		axarr[0].set_title(str(predictions[2*i])+' ' + classes[predictions[2*i]])
		axarr[1].set_title(str(predictions[2*i+1])+' ' + classes[predictions[2*i+1]])

		axarr[0].axis('off')
		axarr[1].axis('off')

		plt.show()
	if number_nons:
		f,axarr=plt.subplots(1)
		axarr.imshow(originals[-1])
		axarr.set_title(str(predictions[-1])+' ' + classes[predictions[-1]])
		axarr.axis('off')
		plt.show()



#------------------------------------Model 1 functions----------------------------

def train_model1(X,Y):
	"""	
		PURPOSE:		Train a logistic regression model using Scikit-learn
		DESCRIPTION: 	Train and save a model of logistic regression with the 
						input training data.

 		PARAMETERS
		X:    			1D Array with all the images  (numpy array)
		Y:				1D array with the classes (numpy array)
	

 		RESULT:
        NONE	    	Save the model into ./models/model1/saved/.
 
	"""
	logisticRegr = LogisticRegression()
	logisticRegr.fit(X, Y)
	
	#Save the model in the folder
	pickle.dump(logisticRegr, open('./models/model1/saved/LRmodel.sav', 'wb'))
	print("Logistic regression model successfully trained with scikit-learn")



def test_model1(X,Y):
	"""	
		PURPOSE:		Test a previously trained logistic regression model using scikit-learn
		DESCRIPTION: 	Use the specified testing data to get the accuracy of the model
						

 		PARAMETERS
		X:    			1D Array with all the images  (numpy array)
		Y:				1D array with the classes (numpy array)
	

 		RESULT:
        NONE	    	Prints the accuracy of the model
 
	"""
	#load the model
	model = pickle.load(open('./models/model1/saved/LRmodel.sav', 'rb'))
	#get the predictions and accuracy of the model
	predictions=model.predict(X)
	accuracy=accuracy_score(Y, predictions)

	file = open('reports/model1.txt','w') 
	file.write(classification_report(Y, predictions))
	file.close()
	print ('accuracy: '+ ("{0:.3f}".format(accuracy)))



def infer_model1(X,originals):
	"""	
		PURPOSE:		Predict the class of a user inserted image or set of images
		DESCRIPTION: 	Use the logistic regression model trained with Scikit-learn
						to predict the class of a set of images, it will plot the 
						original image with a title of its predicted class
						

 		PARAMETERS
		X:    			1D Array with all the images  (numpy array)
		Originals:		1D array with the original images to plot (numpy array)
	

 		RESULT:
        NONE	    	shows a series of windows with plots of couples of images
        				with its class as the image title.
 
	"""
	#load the model
	Nimages=len(X)
	#load the model
	model = pickle.load(open('./models/model1/saved/LRmodel.sav', 'rb'))
	#Get the predictions of the model
	predictions=model.predict(X)
	
	#ploting the predictions

	plot_predictions(originals,predictions)
	
		

#---------------------------Model 2 functions----------------------------------

def train_model2(X,Y,Nclasses=43,batch_size=200,num_steps=150):

	"""	
	PURPOSE:	Train a logistic regression model using Tensorflow
	DESCRIPTION: 	Train and save a model of logistic regression with the 
					input training data.

		PARAMETERS
	X:    			1D Array with all the images  (numpy array)
	Y:				1D array with the classes (numpy array)
	batch_size:		number of randon data examples to feed the model (int)
	num_steps:		number of batches the model will be feeded, epochs (int)

		RESULT:
    NONE	    	Save the model into ./models/model2/saved/.

	"""
	number_train=len(Y)
	number_features=len(X[0])
	

	#creating placeholders and variables
	x=tf.placeholder(tf.float32,[None,number_features],name="x")
	W=tf.Variable(tf.zeros([number_features,Nclasses]),name="W")
	b=tf.Variable(tf.zeros([Nclasses]),name="b")
	y_=tf.placeholder(tf.int64,[None])

	# Defining y=xW+b
	y=tf.matmul(x,W)+b

	# Defining Accuracy
	correct_prediction = tf.equal(tf.argmax(y, 1), y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Defining the loss function and the training step for gradient descent
	cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_,y)
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# Create and initialize a session
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		#Create a saver object which will save all the variables
		saver = tf.train.Saver()

		#Train the model
		for _ in range (num_steps):
			batch_xs,batch_ys=batching(X,Y,batch_size)
			sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

			#Showing the accuracy of the method every 10 epochs, using training data
			if _%10==0:
				print ('epoch: '+str(_)+' accuracy: '+str(sess.run(accuracy,feed_dict={x:X,y_:Y})) )
		# Save the model

		saver.save(sess,'./models/model2/saved/trained_variables',write_meta_graph=True)
		sess.close()
	print("Logistic regression model successfully trained with Tensorflow")
	


def test_model2(X,Y):
	"""	
	PURPOSE:		Test a previously trained logistic regression model using Tensorflow
	DESCRIPTION: 	Use the specified testing data to get the accuracy of the model
					

		PARAMETERS
	X:    			1D Array with all the images  (numpy array)
	Y:				1D array with the classes (numpy array)


		RESULT:
    NONE	    	Prints the accuracy of the model

	"""
	number_features=len(X[0])
	new_graph = tf.Graph()
	with tf.Session(graph=new_graph) as sess:  
		#oad meta graph and restore weights
		new_saver = tf.train.import_meta_graph('./models/model2/saved/trained_variables.meta')
		new_saver.restore(sess,tf.train.latest_checkpoint('./models/model2/saved/'))
		# Get the trained matrices W and b
		
		W = new_graph.get_tensor_by_name('W:0')
		b = new_graph.get_tensor_by_name('b:0')

		#create placeholders
		x=tf.placeholder(tf.float32,[None,number_features],name="x")
		y=tf.matmul(x,W)+b

		#get the prediction and accuracy of the method
		#correct_prediction = tf.equal(tf.argmax(y, 1), Y)
		
		predictions=sess.run(tf.argmax(y,1),feed_dict={x:X})
		correct_prediction=tf.equal(predictions,Y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#print the accuracy
		print ('accuracy: ' + "{0:.3f}".format (sess.run(accuracy,feed_dict={x:X})))		
	


def infer_model2(X,originals):
	"""	
		PURPOSE:		Predict the class of a user inserted image or set of images
		DESCRIPTION: 	Use the logistic regression model trained with Tensorflow
						to predict the class of a set of images, it will plot the 
						original image with a title of its predicted class
						

 		PARAMETERS
		X:    			1D Array with all the images  (numpy array)
		Originals:		1D array with the original images to plot (numpy array)
	

 		RESULT:
        NONE	    	shows a series of windows with plots of couples of images
        				with its class as the image title.
 
	"""
	number_features=len(X[0])
	new_graph = tf.Graph()
	with tf.Session(graph=new_graph) as sess:  
		#Load meta graph and restore weights
		new_saver = tf.train.import_meta_graph('./models/model2/saved/trained_variables.meta')
		new_saver.restore(sess,tf.train.latest_checkpoint('./models/model2/saved/'))
		# Get the trained matrices W and b
		
		W = new_graph.get_tensor_by_name('W:0')
		b = new_graph.get_tensor_by_name('b:0')

		#create placeholders
		x=tf.placeholder(tf.float32,[None,number_features],name="x")
		y_=tf.placeholder(tf.int64,[None])
		y=tf.matmul(x,W)+b

		#get the predictions
		predictions=sess.run(tf.argmax(y,1),feed_dict={x:X})

		plot_predictions(originals,predictions)
		
		#print ('accuracy: ' + "{0:.3f}".format (sess.run(accuracy,feed_dict={x:X,y_:Y})))		
	




#-------------------------------Download function---------------------


def downl():
	"""	
		PURPOSE:		Download and organize the data images separated into Test and Train sets
		DESCRIPTION: 	Downloads the image database of the German Traffic Signs Dataset,
						unzip it and copy the images of the classification set into another folder
						to then split the data in train a test data, using a proportion of 80-20
						

 		PARAMETERS
		(none)

 		RESULT:
        NONE	    	images/train and images/test are now full of images from the dataset
 
	"""
	#create the folder to save the file
	os.makedirs('images/download')
	click.echo('folder created')
	#download the data from the web
	click.echo('downloading the file')
	url='http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
	urllib.request.urlretrieve(url, "images/download/data.zip")
	click.echo('download complete')
	#unzip the folder
	zip_ref = zipfile.ZipFile("images/download/data.zip", 'r')
	zip_ref.extractall("images/download")
	zip_ref.close()
	click.echo('unzip complete')
	
	#cut all the files to a temporal folder and give them the name of their respective class
	os.makedirs('images/download/temp')
	folders=[str("{:.1f}".format(float(i/10))).replace(".","") for i in list(range(0,43))]
		
	for folder in folders:
		_files=os.listdir('images/download/FullIJCNN2013/'+folder)
		for file in _files:
			os.rename('images/download/FullIJCNN2013/'+folder+'/'+file, 'images/download/temp/'+ folder+'-'+file)
		
	#split the images in test and train, put in their respective folders
	files=os.listdir('images/download/temp')
	shuffle(files)
	Ntrain=int(len(files)*0.8)
	for i in range(0,Ntrain):
		os.rename('images/download/temp/'+files[i], 'images/train/'+ files[i])
	for i in range(Ntrain,len(files)):
		os.rename('images/download/temp/'+files[i], 'images/test/'+ files[i])
	#delete the temporal folder
	shutil.rmtree('images/download')
	click.echo('splitting data completed')


#----------------------------------Dictionary with all the classes--------------
classes={0 : 'speed limit 20 (prohibitory)',
1 : 'speed limit 30 (prohibitory)',
2 : 'speed limit 50 (prohibitory)',
3 : 'speed limit 60 (prohibitory)',
4 : 'speed limit 70 (prohibitory)',
5 : 'speed limit 80 (prohibitory)',
6 : 'restriction ends 80 (other)',
7 : 'speed limit 100 (prohibitory)',
8 : 'speed limit 120 (prohibitory)',
9 : 'no overtaking (prohibitory)',
10 : 'no overtaking (trucks) (prohibitory)',
11 : 'priority at next intersection (danger)',
12 : 'priority road (other)',
13 : 'give way (other)',
14 : 'stop (other)',
15 : 'no traffic both ways (prohibitory)',
16 : 'no trucks (prohibitory)',
17 : 'no entry (other)',
18 : 'danger (danger)',
19 : 'bend left (danger)',
20 : 'bend right (danger)',
21 : 'bend (danger)',
22 : 'uneven road (danger)',
23 : 'slippery road (danger)',
24 : 'road narrows (danger)',
25 : 'construction (danger)',
26 : 'traffic signal (danger)',
27 : 'pedestrian crossing (danger)',
28 : 'school crossing (danger)',
29 : 'cycles crossing (danger)',
30 : 'snow (danger)',
31 : 'animals (danger)',
32 : 'restriction ends (other)',
33 : 'go right (mandatory)',
34 : 'go left (mandatory)',
35 : 'go straight (mandatory)',
36 : 'go right or straight (mandatory)',
37 : 'go left or straight (mandatory)',
38 : 'keep right (mandatory)',
39 : 'keep left (mandatory)',
40 : 'roundabout (mandatory)',
41 : 'restriction ends (overtaking) (other)',
42 : 'restriction ends (overtaking (trucks)) (other)'}


if __name__ == '__main__':
    cli()