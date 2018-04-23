import model
from cnn_utils import *
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

_, _, parameters = model.model(X_train, Y_train, X_test, Y_test)

fname = "images/two.JPG"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64))
my_image=np.reshape(my_image,[64,64,3])
plt.imshow(my_image)
my_image=my_image[np.newaxis,:]
print(my_image.shape)
print(model.predict1(my_image,parameters))


