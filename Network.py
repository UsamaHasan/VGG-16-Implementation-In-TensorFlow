
import tensorflow as tf 
import numpy as np
import time
import os

'''VGG-16 in TensorFlow'''
class VGG16():
    __doc__= 'VGG-16 Implementation in Tensorflow'


    x = tf.placeholder(tf.float32 , shape = [None,28*28], name='x_axis')
                    #Declare the placeholder For Y i.e training labels
    y = tf.placeholder(tf.float32 ,shape= [None,10] , name='labels' )

    x_image = tf.reshape(x , [-1,28,28,1])
    
    y_true = tf.argmax(y,dimension=1)

        

    def __init__(self):
        pass
    
    def conv_layer(self , input , no_input_channels,no_filter, filter_size ,name):
        ''''''
        with tf.variable_scope(name):
            shape = [filter_size , filter_size , no_input_channels, no_filter ]
            print(f'Conv Layer Shape:{shape}')
            #Declare Weights w.r.t to the scope of layer
            weights = tf.Variable(tf.truncated_normal(shape,stddev=0.05))
            #Declare Bias
            biases = tf.Variable(tf.constant(0.05,shape=[no_filter]))
            #Conv Layer
            conv = tf.nn.conv2d(input=input , filter= weights , strides= [1,1,1,1],padding='SAME')
            #Add Bias in convoluted map
            conv += biases
            
            return conv , weights
    ''' Relu layer
    Args:
    Input(Tensor): Takes 
    
    '''   
    def relu_layer(self,input , name):
        with tf.variable_scope(name):
            layer = tf.nn.relu(input)
        return layer
        
    '''Fully Connected Layer'''
    def fc_layer(self, input ,num_input , num_output ,name):
        with tf.variable_scope(name):
            #Declare Weights
            weights = tf.Variable(tf.truncated_normal([num_input, num_output],stddev= 0.05))
            #Declare Bias
            bias =  tf.Variable(tf.constant(0.05,shape=[num_output]))

            layer = tf.matmul(input , weights) + bias

            return layer
    '''Pooling Layer'''
    def pooling_layer(self,layer,name):

        with tf.variable_scope(name):
            output_layer = tf.nn.max_pool(layer, ksize=[1,2,2,1] , strides=[1,2,2,1] ,padding='SAME' )
        return output_layer
    '''
    Train:
    Args:
    Epochs(Int) = Number of training iterations
    Batch_size(Int) = 
    '''
    def train(self , data , epochs , batch_size) :
        
        # Model
        self.conv1 , self.weight_conv1 = self.conv_layer(input= self.x_image , no_input_channels= 1 ,no_filter= 16 , filter_size= 3 , name = 'conv1')

        self.relu_1  = self.relu_layer(self.conv1,'relu_1')
        
        self.conv2 , self.weight_conv2 = self.conv_layer(self.relu_1 , no_input_channels= 16 , filter_size=3 , no_filter=16 , name = 'conv2' )

        self.relu_2 = self.relu_layer(self.conv2 , 'relu_2')

        self.conv_3  , self.weight_conv3= self.conv_layer(self.relu_2, no_input_channels = 16 , no_filter= 16, filter_size = 3, name = 'conv3' )

        self.relu_3 = self.relu_layer(self.conv_3 , 'relu_3')

        self.pool_1 = self.pooling_layer(self.relu_3 ,name ='Pooling_1')

#        self.conv4 , self.weight_conv4 = self.conv_layer(input=self.pool_1, no_input_channels= 16 ,no_filter= 32 , filter_size= 3 , name = 'conv4')

 #       self.relu_4  = self.relu_layer(self.conv4,'relu_4')
        
  #      self.conv5 , self.weight_conv5 = self.conv_layer(self.relu_4 , no_input_channels= 32 , filter_size=3 , no_filter=32 , name = 'conv5' )

   #     self.relu_5 = self.relu_layer(self.conv5 , 'relu_5')

    #    self.conv_6  , self.weight_conv6= self.conv_layer(self.relu_5, no_input_channels = 32 , no_filter= 32, filter_size = 3, name = 'conv6' )

     #   self.relu_6 = self.pooling_layer(self.conv_6 , 'relu_6')
        
      #  self.pool_2 = self.pooling_layer(self.relu_6 ,name ='Pooling_2')


        #tf.Tensor.get_shape()
        number_features = self.pool_1.get_shape()[1:4].num_elements()
        
        #flatten
        self.flatten_layer = tf.reshape(self.pool_1 ,[-1 , number_features])
        #Fully Connected Layer
        self.fc_1 = self.fc_layer(self.flatten_layer , num_input=number_features ,num_output = 128 , name ='Fc_1' )

        self.relu_7 = self.relu_layer(self.fc_1 , name = 'relu_4')        
        #Output Layer
        self.fc_2 =  self.fc_layer(self.relu_7 , num_input = 128 , num_output = 10 , name ='output_layer')

        #Softmax For Output Layer
        with tf.variable_scope('Softmax'):
            y_pred = tf.nn.softmax(self.fc_2)
            #Returns the index with the largest value across axes of a tensor.
            y_pred_cls = tf.arg_max(y_pred,dimension=1)
        #Cross Entropy to calculate loss funtion
        with tf.variable_scope('Cross_Entropy'):
            # This will return a 1-D vector that contains the error/cost w.r.t each predicted class
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits =  self.fc_2,labels = self.y   )
            #Calculate the average error
            self.mean_error = tf.reduce_mean(cross_entropy)
            self.mean_error = tf.Print(self.mean_error , [self.mean_error])
        #Optimizer function which will minimize loss and back propagation
        with tf.name_scope('Optimzier'):
            optimizer  = tf.train.AdamOptimizer(learning_rate=0.01)
            gradients = optimizer.compute_gradients(self.mean_error)
            optim = optimizer.minimize(self.mean_error)
            for v , g in  gradients:
                if 'output_layer' in v.name and 'weights' in v.name:
                    with tf.variable_scope('gradient'):
                        tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                        tf_grad_norm_summary = tf.summary.scalar('gradients' , tf_last_grad_norm)
                        break
        #Accuracy Calculation
        with tf.variable_scope('Accuracy'):
            #Returns the truth value of (x == y) element-wise.
            #tf.math.equal(x, y) ==> array([True, False]) or array([1,0,0,1,0,..])
            correct_prediction = tf.equal(y_pred_cls,self.y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #session = tf.InteractiveSession()
        #Creating a tf.Summary.FileWriter object to write the current scalar variables and learning parameters.
        with tf.name_scope('Performance'):
            loss_placeholder = tf.placeholder(tf.float32 , shape=None , name='loss_summary')
            loss_summary = tf.summary.scalar('loss',loss_placeholder)

            accuracy_placeholder = tf.placeholder(tf.float32 , shape = None , name = 'accuracy_summary')
            accuracy_summary = tf.summary.scalar('accuracy' , accuracy_placeholder)

        #Taking gradients of last layer from each epoch and writing them in tf.Summary file for tensorboard 
        
                

        with tf.Session() as sess:
            
            #Initializing all the variables
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(os.path.join('summary','first'),graph = sess.graph)
            #feed Dictionary for training 
                        
            for epoch in range(epochs):
                start_time = time.time()
                train_accuracy = 0
                #Loading Mini-batch from data pipeline
                for batch in range(0,int(len(data.train.labels)/batch_size)):
                    
                    x_train , y_train = data.train.next_batch(batch_size)
                    feed_dict_train = {self.x:x_train , self.y:y_train}
                    #Feed Dictionary for validation
                    feed_dict_validation = {self.x:data.validation.images , self.y:data.validation.labels}                    
            

                    if batch == 0:

                        
                        #Start Training in TensorFlow session, creating graph of our model.
                        sess.run(optim ,feed_dict=feed_dict_train)
                       # sess.run(tf_grad_norm_summary , feed_dict = feed_dict_train)
                        #Calculate Accuracy
                        train_accuracy +=sess.run(accuracy,feed_dict=feed_dict_train)
                    else:
                        sess.run(optim ,feed_dict=feed_dict_train)
                        train_accuracy +=sess.run(accuracy,feed_dict=feed_dict_train)
                        

                train_accuracy /= int(len(data.train.labels)/batch_size)
                #Calculate Validation Accuracy or 
                validation_accuracy = sess.run(accuracy,feed_dict = feed_dict_validation)
                end_time = time.time()
                #writer.add_summary(tf_grad_norm_summary)
                print(f'Training Accuracy:{train_accuracy}')
                print(f'Validation Accuracy:{validation_accuracy}')
                print(f'Epoch:{epoch} Time Taken:{int(end_time-start_time)}')
            #Close the current session
            sess.close()