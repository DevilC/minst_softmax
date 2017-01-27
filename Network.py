import Load_Data as data
import theano
import numpy as np
from theano import tensor as T
import os
import cPickle as pickle

def weight_init(shape, choose):
    path = ''
    if choose == 0:
        path = r'./data/input_hidden_weight.txt'
    else:
        path = r'./data/hidden_output_weight.txt'
    if os.path.exists(path):
        w = np.loadtxt(path)
    else:
        print 'file do not exist'
        w = np.random.uniform(-0.1, 0.1, shape)
    return w

def save_weight():
    np.savetxt(r'./data/input_hidden_weight.txt',np.array(i_h_weight.get_value()))
    np.savetxt(r'./data/hidden_output_weight.txt', np.array(h_o_weight.get_value()))

input_size = 784
sample_num = 50000
step = 0.1
hidden_num = 30
output_num = 10

input = T.vector('hidden_input')
h_output = T.vector('hidden_output')
o_input = T.vector('outLayer_input')
o_output = T.vector('outLayer_output')
target = T.vector('target')

i_h_shape = (hidden_num, input_size)
i_h_weight = theano.shared(weight_init(i_h_shape,0), 'input2hidden_weight')

h_o_shape = (output_num, hidden_num)
h_o_weight = theano.shared(weight_init(h_o_shape,1), 'hidden2output_weight')

h_sum = T.dot(input, T.transpose(i_h_weight))
h_output = 1/(1+T.exp(-h_sum))#sigmod function

o_input = h_output
o_sum = T.dot(o_input, T.transpose(h_o_weight))
o_output = 1/(1+T.exp(-o_sum))

loss = T.sum((o_output-target)**2) / output_num

grad_h2o = T.grad(loss, h_o_weight)
grad_i2h = T.grad(loss, i_h_weight)

#f = theano.function([input], o_output )
train = theano.function([input, target], loss,
                    updates = [(h_o_weight, h_o_weight - step * grad_h2o),
                               (i_h_weight, i_h_weight - step * grad_i2h)],
                    on_unused_input='ignore')

network = theano.function([input], o_output, on_unused_input='ignore')

sum_loss =0
for i in range(0,5):
    print '%d train'%(i)

    right = 0
    for ip, t in zip(data.test_setX, data.test_setY):
        t = int(t)
        op = np.array(network(ip))
        result = np.argmax(op)
        if (result == t):
            right += 1
    print 'right rate %s'%(float(right)/100),'%'

    for ip, t in zip(data.train_setX, data.train_setY):
        y = [0 for x in range(0,10)]
        y[int(t)] = 1
        sum_loss += train(ip, y)
    print 'total loss: ', sum_loss
    sum_loss = 0
    save_weight()

