from VGG import VGG16
from tensorflow.examples.tutorials.mnist import input_data
from VGG import VGG16

if __name__=='__main__':

    data = input_data.read_data_sets('dataset/mnist/',one_hot = 'True')
    
    print("Size of:")
    print("- Traiconing-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(data.test.labels)))
    print("- Validation-set:\t{}".format(len(data.validation.labels)))
    
    net = VGG16()
    net.train(data ,10 , 32)
