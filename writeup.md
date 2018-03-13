
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier
---

## Dataset Summary & Exploration

* The size of training set is ?   
   Number of training examples = 34799  
* The size of the validation set is ?   
  Number of validating examples = 4410   
* The size of test set is ?   
  Number of validating examples = 4410   
* The shape of a traffic sign image is ?   
  Image data shape = (32, 32, 3)   
* The number of unique classes/labels in the data set is ?   
  Number of classes = 43  


## Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 



##  Randomly show special class iamges`

    Show Class = [8],Name=[Speed limit (120km/h)] from data set,Show length is [10],Total length is [1260]
    


![png](./writeup_img/output_10_1.png)


##  Randomly show all classes iamges

    Random show each class form data set:
    


![png](./writeup_img/output_12_1.png)



```python
# show_class_by_class_images(X_train,y_train)
```

# Show class images histogram


#### Show per class images frequency

![png](./writeup_img/output_18_0.png)


#### Show per class on train and test set


![png](./writeup_img/output_20_0.png)


## Extend the images set

![png](./writeup_img/output_24_1.png)


![png](./writeup_img/output_25_1.png)


#### images generator

```python
gen_class_images_show(X_train,y_train,24)
```


![png](./writeup_img/output_28_0.png)


    Generating images... Class = 24
    Show Class = [24],Name=[Road narrows on the right] from data set,Show length is [20],Total length is [30]
    


![png](./writeup_img/output_28_2.png)


#### images data merge


```python
X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)
```

### Generator All class images to balace data

```python
X_train,y_train = gen_class_images(X_train,y_train)
```

    Class  0  :  210, Generated samples numbers = 6008
    Class  1  : 2220, Generated samples numbers = 6104
    Class  2  : 2250, Generated samples numbers = 6036
    Class  3  : 1410, Generated samples numbers = 6024
    Class  4  : 1980, Generated samples numbers = 6068
    Class  5  : 1860, Generated samples numbers = 6092
    Class  6  :  420, Generated samples numbers = 6008
    Class  7  : 1440, Generated samples numbers = 6016
    Class  8  : 1410, Generated samples numbers = 6024
    Class  9  : 1470, Generated samples numbers = 6008
    Class 10  : 2010, Generated samples numbers = 6030
    Class 11  : 1320, Generated samples numbers = 6048
    Class 12  : 2100, Generated samples numbers = 6120
    Class 13  : 2160, Generated samples numbers = 6112
    Class 14  :  780, Generated samples numbers = 6088
    Class 15  :  630, Generated samples numbers = 6054
    Class 16  :  420, Generated samples numbers = 6008
    Class 17  : 1110, Generated samples numbers = 6062
    Class 18  : 1200, Generated samples numbers = 6080
    Class 19  :  210, Generated samples numbers = 6008
    Class 20  :  360, Generated samples numbers = 6016
    Class 21  :  330, Generated samples numbers = 6068
    Class 22  :  390, Generated samples numbers = 6100
    Class 23  :  510, Generated samples numbers = 6122
    Class 24  :  270, Generated samples numbers = 6054
    Class 25  : 1500, Generated samples numbers = 6128
    Class 26  :  600, Generated samples numbers = 6040
    Class 27  :  240, Generated samples numbers = 6128
    Class 28  :  540, Generated samples numbers = 6040
    Class 29  :  270, Generated samples numbers = 6054
    Class 30  :  450, Generated samples numbers = 6106
    Class 31  :  780, Generated samples numbers = 6088
    Class 32  :  240, Generated samples numbers = 6128
    Class 33  :  689, Generated samples numbers = 6024
    Class 34  :  420, Generated samples numbers = 6008
    Class 35  : 1200, Generated samples numbers = 6080
    Class 36  :  390, Generated samples numbers = 6100
    Class 37  :  210, Generated samples numbers = 6008
    Class 38  : 2070, Generated samples numbers = 6060
    Class 39  :  300, Generated samples numbers = 6084
    Class 40  :  360, Generated samples numbers = 6016
    Class 41  :  240, Generated samples numbers = 6128
    Class 42  :  240, Generated samples numbers = 6128
    Generate images data has completed!
    

#### Save generated image data


```python
import pickle
gen_data_file = "traffic-signs-data/gen_data.p"
print("Generated iamges numbers = {}".format(len(X_train)))
pickle.dump({"images":X_train,"labels":y_train},open(gen_data_file,"wb"),protocol=4)
print("Generated images data has saved completly!")
```

    Generated iamges numbers = 43619
    Generated images data has saved completly!
    

#### Restore generated image data


```python
with open("traffic-signs-data/gen_data.p","rb") as f:
    image_data = pickle.load(f)

X_train,y_train = image_data["images"],image_data["labels"]
```

#### Split generated image data into train and valid set


```python
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
show_compared_histogram(y_train,y_valid)
```


![png](./writeup_img/output_39_0.png)   
![png](./writeup_img/output_39_0.png)


----

## Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128.)/ 128.` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

#### Normalize image data & shuffle train data


```python
X_train_normalized = normalize_img(X_train)
X_valid_normalized = normalize_img(X_valid)
X_test_normalized = normalize_img(X_test)
```

## Model Architecture

My final model like VGG consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| DropOut   	      	| keep Prob = 0.9 				                |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 			     	|
| DropOut   	      	| keep Prob = 0.9 				                |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 		     		|
| DropOut   	      	| keep Prob = 0.9 				                |
| Fully connected		| outputs 2048       				    		|
| DropOut   	      	| keep Prob = 0.8 				                |
| Fully connected		| outputs 2048       				    		|
| DropOut   	      	| keep Prob = 0.5 				                |
| Fully connected		| outputs 43        				    		|
| Softmax				| etc.        									|

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# -*- coding=UTF-8 -*-

# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43) # 这里要用独热编码，因为如果真实的标签是20，预测是10，两者之差10，惩罚太重了

# Training Pipeline (Loss Fucntion)
logits,loss_regularization, keep_prob1, keep_prob2, keep_prob3 = model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

# loss_operation = tf.reduce_mean(cross_entropy + loss_rate*loss_regularization)
# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
# training_operation = optimizer.minimize(loss_operation)

# Model Evaluation
#这里纯计算预测值是否与真实值一致，一致为真，不一致为假；若需要输出预测概率需用tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1)) # tf.argmax(logits, 1) return max value index of per col
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        
        # NOTE: Here should ban dropout, becouse model has trained completely
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob1:1.0, keep_prob2:1.0, keep_prob3:1.0})
        
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

####  Define mode function


```python
from sklearn.utils import shuffle
saver = tf.train.Saver()

learning_rate = 0.0005
loss_rate = 0.0001
loss_operation = tf.reduce_mean(cross_entropy + loss_rate*loss_regularization)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

def trainmodel(X_train=X_train, y_train=y_train,X_valid=X_valid, y_valid=y_valid, epochs=80, batch_size=128, start_log_epoch=8,epoch_tolerance=10):
    """
    Train mode
    
    Arguments:
        X_train: source image data
        y_train: source iamge data corresponding labels
        X_valid: valid image data
        y_valid: valid iamge data corrdsponding labels
        epochs: the number of epoch
        batch_size: take image num
        start_log_epoch: 开始记录性能提升
        epoch_tolerance: 多少轮以后准确率没提升
    """
    best_validation_accuracy = 0.0
    last_improved_epoch = start_log_epoch
    
    if start_log_epoch > epochs:
        print("Epochs numbers {:>2} should greater than start_log_epoch {:>2}".format(epochs,start_log_epoch))
        return
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            
            # Training
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob1:0.9, keep_prob2:0.8, keep_prob3:0.5})

            # Evaluate on train set and valid set
            train_accuracy = evaluate(X_train, y_train, batch_size)
            print("EPOCH {:>2} ...".format(i+1), end=' ')
            print("Train Accuracy = {:.4f}".format(train_accuracy), end='  ')
            
            validation_accuracy = evaluate(X_valid, y_valid, batch_size)
            print("Validation Accuracy = {:.4f}".format(validation_accuracy))
            
            if i > start_log_epoch and validation_accuracy > best_validation_accuracy:
                # updata the best-known validation accuracy
                best_validation_accuracy = validation_accuracy
                print("Current Best Validation Accuracy = {:.4f} has saved completely!".format(best_validation_accuracy))
                
                # Record the last validation accuracy improve epoch
                last_improved_epoch = i
                
                # Save all parameters to file
                saver.save(sess, './model/')
            
            if i - last_improved_epoch > epoch_tolerance:
                print("{} epochs have no improvement after the best validation accuracy = {:.4f}".format(epoch_tolerance,best_validation_accuracy))
                
                # Jump out the loop
                break
                

        # saver.save(sess, './model/')
        # print("Model saved")
        print("Best Accuracy = {:.4f} Model has saved!".format(best_validation_accuracy))
```

#### train mode


```python
trainmodel(X_train_normalized, y_train,X_valid_normalized, y_valid)
```

    Training...
    
    EPOCH  1 ... Train Accuracy = 0.9706  Validation Accuracy = 0.9650
    EPOCH  2 ... Train Accuracy = 0.9903  Validation Accuracy = 0.9856
    EPOCH  3 ... Train Accuracy = 0.9912  Validation Accuracy = 0.9873
    EPOCH  4 ... Train Accuracy = 0.9907  Validation Accuracy = 0.9870
    EPOCH  5 ... Train Accuracy = 0.9924  Validation Accuracy = 0.9893
    EPOCH  6 ... Train Accuracy = 0.9963  Validation Accuracy = 0.9933
    EPOCH  7 ... Train Accuracy = 0.9971  Validation Accuracy = 0.9943
    EPOCH  8 ... Train Accuracy = 0.9979  Validation Accuracy = 0.9952
    EPOCH  9 ... Train Accuracy = 0.9975  Validation Accuracy = 0.9953
    EPOCH 10 ... Train Accuracy = 0.9956  Validation Accuracy = 0.9927
    Current Best Validation Accuracy = 0.9927 has saved completely!
    EPOCH 11 ... Train Accuracy = 0.9975  Validation Accuracy = 0.9955
    Current Best Validation Accuracy = 0.9955 has saved completely!
    EPOCH 12 ... Train Accuracy = 0.9969  Validation Accuracy = 0.9940
    EPOCH 13 ... Train Accuracy = 0.9988  Validation Accuracy = 0.9963
    Current Best Validation Accuracy = 0.9963 has saved completely!
    EPOCH 14 ... Train Accuracy = 0.9989  Validation Accuracy = 0.9972
    Current Best Validation Accuracy = 0.9972 has saved completely!
    EPOCH 15 ... Train Accuracy = 0.9986  Validation Accuracy = 0.9967
    EPOCH 16 ... Train Accuracy = 0.9982  Validation Accuracy = 0.9961
    EPOCH 17 ... Train Accuracy = 0.9987  Validation Accuracy = 0.9969
    EPOCH 18 ... Train Accuracy = 0.9978  Validation Accuracy = 0.9951
    EPOCH 19 ... Train Accuracy = 0.9988  Validation Accuracy = 0.9967
    EPOCH 20 ... Train Accuracy = 0.9989  Validation Accuracy = 0.9968
    EPOCH 21 ... Train Accuracy = 0.9993  Validation Accuracy = 0.9974
    Current Best Validation Accuracy = 0.9974 has saved completely!
    EPOCH 22 ... Train Accuracy = 0.9987  Validation Accuracy = 0.9966
    EPOCH 23 ... Train Accuracy = 0.9993  Validation Accuracy = 0.9974
    EPOCH 24 ... Train Accuracy = 0.9962  Validation Accuracy = 0.9938
    EPOCH 25 ... Train Accuracy = 0.9990  Validation Accuracy = 0.9974
    EPOCH 26 ... Train Accuracy = 0.9980  Validation Accuracy = 0.9961
    EPOCH 27 ... Train Accuracy = 0.9987  Validation Accuracy = 0.9969
    EPOCH 28 ... Train Accuracy = 0.9990  Validation Accuracy = 0.9968
    EPOCH 29 ... Train Accuracy = 0.9986  Validation Accuracy = 0.9965
    EPOCH 30 ... Train Accuracy = 0.9985  Validation Accuracy = 0.9962
    EPOCH 31 ... Train Accuracy = 0.9994  Validation Accuracy = 0.9980
    Current Best Validation Accuracy = 0.9980 has saved completely!
    EPOCH 32 ... Train Accuracy = 0.9995  Validation Accuracy = 0.9976
    EPOCH 33 ... Train Accuracy = 0.9993  Validation Accuracy = 0.9974
    EPOCH 34 ... Train Accuracy = 0.9982  Validation Accuracy = 0.9956
    EPOCH 35 ... Train Accuracy = 0.9991  Validation Accuracy = 0.9972
    EPOCH 36 ... Train Accuracy = 0.9995  Validation Accuracy = 0.9976
    EPOCH 37 ... Train Accuracy = 0.9996  Validation Accuracy = 0.9977
    EPOCH 38 ... Train Accuracy = 0.9994  Validation Accuracy = 0.9979
    EPOCH 39 ... Train Accuracy = 0.9997  Validation Accuracy = 0.9981
    Current Best Validation Accuracy = 0.9981 has saved completely!
    EPOCH 40 ... Train Accuracy = 0.9994  Validation Accuracy = 0.9977
    EPOCH 41 ... Train Accuracy = 0.9986  Validation Accuracy = 0.9965
    EPOCH 42 ... Train Accuracy = 0.9996  Validation Accuracy = 0.9977
    EPOCH 43 ... Train Accuracy = 0.9994  Validation Accuracy = 0.9972
    EPOCH 44 ... Train Accuracy = 0.9985  Validation Accuracy = 0.9961
    EPOCH 45 ... Train Accuracy = 0.9993  Validation Accuracy = 0.9978
    EPOCH 46 ... Train Accuracy = 0.9962  Validation Accuracy = 0.9941
    EPOCH 47 ... Train Accuracy = 0.9995  Validation Accuracy = 0.9982
    Current Best Validation Accuracy = 0.9982 has saved completely!
    EPOCH 48 ... Train Accuracy = 0.9998  Validation Accuracy = 0.9984
    Current Best Validation Accuracy = 0.9984 has saved completely!
    EPOCH 49 ... Train Accuracy = 0.9991  Validation Accuracy = 0.9965
    EPOCH 50 ... Train Accuracy = 0.9995  Validation Accuracy = 0.9978
    EPOCH 51 ... Train Accuracy = 0.9994  Validation Accuracy = 0.9977
    EPOCH 52 ... Train Accuracy = 0.9991  Validation Accuracy = 0.9972
    EPOCH 53 ... Train Accuracy = 0.9996  Validation Accuracy = 0.9980
    EPOCH 54 ... Train Accuracy = 0.9988  Validation Accuracy = 0.9970
    EPOCH 55 ... Train Accuracy = 0.9997  Validation Accuracy = 0.9984
    EPOCH 56 ... Train Accuracy = 0.9996  Validation Accuracy = 0.9979
    EPOCH 57 ... Train Accuracy = 0.9995  Validation Accuracy = 0.9980
    EPOCH 58 ... Train Accuracy = 0.9988  Validation Accuracy = 0.9967
    EPOCH 59 ... Train Accuracy = 0.9995  Validation Accuracy = 0.9978
    10 epochs have no improvement after the best validation accuracy = 0.9984
    Best Accuracy = 0.9984 Model has saved!
    

#### FineTune modle


```python
learning_rate = 0.0001
loss_rate = 0.0001
loss_operation = tf.reduce_mean(cross_entropy + loss_rate*loss_regularization)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

def finetune(X_train=X_train, y_train=y_train,X_valid=X_valid, y_valid=y_valid, epochs=80, batch_size=128, start_log_epoch=3,epoch_tolerance=10):
    """
    Fine tune the retrained model
    """
    best_validation_accuracy = 0.0
    last_improved_epoch = start_log_epoch
    
    if start_log_epoch > epochs:
        print("Epochs numbers {:>2} should greater than start_log_epoch {:>2}".format(epochs,start_log_epoch))
        return
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model/')
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            
            # Training
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob1:0.9, keep_prob2:0.75, keep_prob3:0.5})

            # Evaluate on train set and valid set
            train_accuracy = evaluate(X_train, y_train, batch_size)
            print("EPOCH {:>2} ...".format(i+1), end=' ')
            print("Train Accuracy = {:.4f}".format(train_accuracy), end='  ')
            
            validation_accuracy = evaluate(X_valid, y_valid, batch_size)
            print("Validation Accuracy = {:.4f}".format(validation_accuracy))
            
            if i > start_log_epoch and validation_accuracy > best_validation_accuracy:
                # updata the best-known validation accuracy
                best_validation_accuracy = validation_accuracy
                print("Current Best Validation Accuracy = {:.4f} has saved completely!".format(best_validation_accuracy))
                
                # Record the last validation accuracy improve epoch
                last_improved_epoch = i
                
                # Save all parameters to file
                saver.save(sess, './model/')
            
            if i - last_improved_epoch > epoch_tolerance:
                print("{:>2} epochs have no improvement after the best validation accuracy = {:.4f}".format(epoch_tolerance,best_validation_accuracy))
                
                # Jump out the loop
                break
                

        # saver.save(sess, './model/')
        # print("Model saved")
        print("Best Accuracy = {:.4f} Model has saved!".format(best_validation_accuracy))
```


```python
finetune(X_train_normalized, y_train,X_valid_normalized, y_valid)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Training...
    
    EPOCH  1 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9989
    EPOCH  2 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9987
    EPOCH  3 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9990
    EPOCH  4 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9988
    EPOCH  5 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9989
    Current Best Validation Accuracy = 0.9989 has saved completely!
    EPOCH  6 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9989
    EPOCH  7 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9990
    Current Best Validation Accuracy = 0.9990 has saved completely!
    EPOCH  8 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9988
    EPOCH  9 ... Train Accuracy = 0.9999  Validation Accuracy = 0.9987
    EPOCH 10 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9991
    Current Best Validation Accuracy = 0.9991 has saved completely!
    EPOCH 11 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9992
    Current Best Validation Accuracy = 0.9992 has saved completely!
    EPOCH 12 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9988
    EPOCH 13 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9991
    EPOCH 14 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9991
    EPOCH 15 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9989
    EPOCH 16 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9990
    EPOCH 17 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9987
    EPOCH 18 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9991
    EPOCH 19 ... Train Accuracy = 0.9999  Validation Accuracy = 0.9986
    EPOCH 20 ... Train Accuracy = 0.9999  Validation Accuracy = 0.9987
    EPOCH 21 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9989
    EPOCH 22 ... Train Accuracy = 1.0000  Validation Accuracy = 0.9990
    10 epochs have no improvement after the best validation accuracy = 0.9992
    Best Accuracy = 0.9992 Model has saved!
    

#### Test mode on test data set


```python
def testdata(X_data, y_labels, BATCH_SIZE=128):
    with tf.Session() as sess:
        saver.restore(sess, './model/')
        test_accuracy = evaluate(X_data, y_labels, BATCH_SIZE)
        print("Test Accuracy = {:.4f}".format(test_accuracy))
```


```python
testdata(X_test_normalized,y_test)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Test Accuracy = 0.9846
    

## Analysis Error Images


```python
def predict(X_data,y_data,logits=logits,BATCH_SIZE=128):
    """
    Predict Labels
    
    Arguments:
        X_data: source images
        y_data: source images corresponding lables
        logits: tensor
    """
    y_softmax = tf.nn.softmax(logits=logits)
    y_predict = tf.argmax(y_softmax,axis=1)
    num = len(X_data)
    labels = []
    with tf.Session() as sess:
        saver.restore(sess, './model/')
        for offset in range(0,num,BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            label = sess.run(y_predict,feed_dict={x:batch_x,y:batch_y,keep_prob1:1.0,keep_prob2:1.0,keep_prob3:1.0})
            labels.extend(label)
    return np.array(labels)
```


    The number of incorrectly predict labels is 194
    


![png](./writeup_img/output_65_1.png)


---

## Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images





![png](./writeup_img/output_68_0.png)

    

### Predict the Sign Type for Each Image



![png](./writeup_img/output_70_1.png)


### Analyze Performance

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| Speed limit (20km/h)  | Speed limit (20km/h)  			     		| 
| Stop       			| Stop  										|
| Speed limit (50km/h)	| Speed limit (50km/h)							|
| Turn left ahead  		| Turn left ahead		    	 				|
| Speed limit (20km/h)	| Speed limit (20km/h) 							|
    

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
for i in range(len(pro_top_5.values)):
    print("Image {:>2}:".format(i))
    pro_top_5.values[i].sort()
    sorted_p = pro_top_5.values[i][::-1]
    for j in range(len(sorted_p)):
        print(" * P{}: {} - {}".format(j, sorted_p[j], signnames[pro_top_5.indices[i][j]]))
```

    Image  0:
     * P0: 0.9999920129776001 - Speed limit (20km/h)
     * P1: 3.950679001718527e-06 - Bumpy road
     * P2: 1.2387681636027992e-06 - Go straight or left
     * P3: 7.85742145126278e-07 - Slippery road
     * P4: 3.9243425931090314e-07 - Dangerous curve to the right
    Image  1:
     * P0: 0.9999912977218628 - Stop
     * P1: 8.44904025143478e-06 - Priority road
     * P2: 1.296925660199122e-07 - Go straight or left
     * P3: 1.1709016689565033e-07 - Road work
     * P4: 5.583398987596411e-08 - Bicycles crossing
    Image  2:
     * P0: 1.0 - Speed limit (50km/h)
     * P1: 3.9274411076053626e-16 - Double curve
     * P2: 1.5519691692895464e-17 - Speed limit (30km/h)
     * P3: 1.501514129774017e-17 - Slippery road
     * P4: 1.2681259471227926e-17 - Roundabout mandatory
    Image  3:
     * P0: 1.0 - Turn left ahead
     * P1: 2.614573655981291e-16 - Priority road
     * P2: 8.40383209222762e-17 - Go straight or left
     * P3: 8.06500898715855e-17 - Ahead only
     * P4: 1.1742970304435551e-18 - Bumpy road
    Image  4:
     * P0: 0.6406804919242859 - Speed limit (20km/h)
     * P1: 0.2095475047826767 - Speed limit (120km/h)
     * P2: 0.1461586058139801 - Dangerous curve to the right
     * P3: 0.003080248599871993 - Speed limit (80km/h)
     * P4: 0.00015177369641605765 - No passing for vehicles over 3.5 metric tons
    


---

##  Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.


```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(sess, image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input,keep_prob1: 1.0,keep_prob2: 1.0,keep_prob3: 1.0})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    if featuremaps > 48:
        featuremaps = 48
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```


```python
def output_Layer_FeatureMap(img_input,layer):
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model/')
        layer = tf.get_default_graph().get_tensor_by_name("conv{}_activation:0".format(layer))
        print(layer)
        outputFeatureMap(sess, image_input, layer)
```


```python
img_src = X_test[1]
img_src_normalized = X_test_normalized[0]
image_input = []
image_input.append(img_src_normalized)
# image_input = np.reshape(img_src_normalized,(1,)+ img_src_normalized.shape)
image_input = np.array(image_input)
print("Source image shape = {}".format(image_input.shape))

plt.figure(figsize=(2, 2))
plt.imshow(img_src) 
plt.xticks([])
plt.yticks([])
plt.show()
```

    Source image shape = (1, 32, 32, 3)
    


![png](./writeup_img/output_81_1.png)


### Conv Layer 1


```python
output_Layer_FeatureMap(image_input,1)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv1_activation:0", shape=(?, 32, 32, 32), dtype=float32)
    


![png](./writeup_img/output_83_1.png)


### Conv Layer 2


```python
output_Layer_FeatureMap(image_input,2)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv2_activation:0", shape=(?, 32, 32, 32), dtype=float32)
    


![png](./writeup_img/output_85_1.png)


### Conv Layer 3


```python
output_Layer_FeatureMap(image_input,3)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv3_activation:0", shape=(?, 16, 16, 64), dtype=float32)
    


![png](./writeup_img/output_87_1.png)



```python
output_Layer_FeatureMap(image_input,4)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv4_activation:0", shape=(?, 16, 16, 64), dtype=float32)
    


![png](./writeup_img/output_88_1.png)



```python
output_Layer_FeatureMap(image_input,4)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv4_activation:0", shape=(?, 16, 16, 64), dtype=float32)
    


![png](./writeup_img/output_89_1.png)



```python
output_Layer_FeatureMap(image_input,5)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv5_activation:0", shape=(?, 8, 8, 128), dtype=float32)
    


![png](./writeup_img/output_90_1.png)



```python
output_Layer_FeatureMap(image_input,6)
```

    INFO:tensorflow:Restoring parameters from ./model/
    Tensor("conv6_activation:0", shape=(?, 8, 8, 128), dtype=float32)
    


![png](./writeup_img/output_91_1.png)

