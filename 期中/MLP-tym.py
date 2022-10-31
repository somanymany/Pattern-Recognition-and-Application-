from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

mnist = input_data.read_data_sets('C:/Users/dell/.spyder-py3/MNIST_data',one_hot=True)
#每个批次大小
batch_size=60
learn_rate=0.01#学习率
round=300#迭代次数
#计算有多少批次
n_batch=mnist.train.num_examples //batch_size
#定义占位符
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#定义隐层，初始化w,b，使用ReLU做激活函数
#W=tf.Variable(tf.random.normal([784,10]))
#b=tf.Variable(tf.random.normal([10]))
#prediction=tf.matmul(x,W)+b
W_L1=tf.Variable(tf.zeros([784,100]))
b_L1=tf.Variable(tf.random.normal([100]))
WL1_plus_b=tf.matmul(x,W_L1)+b_L1
L1=tf.nn.relu(WL1_plus_b)

W_L2=tf.Variable(tf.random.normal([100,10]))
b_L2=tf.Variable(tf.random.normal([10]))
WL2_plus_b=tf.matmul(L1,W_L2)+b_L2

prediction=WL2_plus_b

#定义损失函数，用交叉熵
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

epoch_J={"epoch":[],"loss":[]}
#定义梯度下降法
train_step=tf.train.GradientDescentOptimizer(learn_rate).minimize(cost) 
#返回预测结果是否正确
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化变量
init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(round):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            loss=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys})
           
        epoch_J["epoch"].append(i+1)
        epoch_J["loss"].append(loss)
        #print("迭代次数： ",i+1,"loss值：",epoch_J["loss"][i],"当前W：",sess.run(W_L2),"当前b:",sess.run(b_L2))
    #画代价函数图
    plt.plot(epoch_J["epoch"],epoch_J["loss"],'ro',label="epoch_loss")
    plt.legend()
    plt.show()
    acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print("Iter "+str(i+1)+",testring acc"+str(acc)) 

    
        
        