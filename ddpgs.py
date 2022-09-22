"""
DDPGs
"""
import tensorflow.compat.v1 as tf
import numpy as np
import datetime
from environment import IesEnv_winter
tf.compat.v1.disable_eager_execution()
np.random.seed(2022)
tf.set_random_seed(2022)
is_train = tf.placeholder_with_default(False, (), 'is_train')
starttime = datetime.datetime.now()
OUTPUT_GRAPH = False
#定义超参数
gamma=0.9
hidden_width1=64
hidden_width2=64
hidden_width3=64
best=10
daylength=10
MAX_EPISODES = 20
MAX_EP_STEPS = 24
initial_learning_rate = 0.000005
MEMORY_CAPACITY = 1000
MEMORY_CAPACITY1 = 1000
batch_size = 1
batch_size1 = 32
var=[0.001,0.001,0.001,50]
varr=[0.001,0.001,0.001,50]
tao=0.01
env = IesEnv_winter(period=24,day=0,time_step=0)
env_main = IesEnv_winter(period=24,day=0,time_step=0)
N_S = 5
N_A = 4
max_action = env.action_space.high
max_action=max_action.reshape(1,-1)
low_action=env.action_space.low
low_action=low_action.reshape(1,-1)
a_bound=[env.action_space.low,env.action_space.high]
LARGE_NUM = 1e9
k=5
EPS=1e-4
episode1=100
episode2=100
episode3=100
length=100
b_bound_numb=[length,length,length,length,length]
current_class_state =[0.20,0.20,0.20,0.20,0.20]
current_class_state=np.mat(current_class_state)
LEARNING_RATE_STEP=150000

class policygradient(object):
    def __init__(self):
        with tf.name_scope(name='Pg_class_train'):
            self.weights1 = tf.Variable(tf.random_normal([k, hidden_width1], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width1, k], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width1, k], stddev=0.1))
    def forward(self, state):
        la = tf.nn.relu(tf.matmul(state, self.weights1))
        miu = tf.nn.tanh(tf.matmul(la,self.weights2))
        sigema = tf.nn.softplus(tf.matmul(la, self.weights3)) + EPS
        normal_dist = tf.distributions.Normal(miu, sigema)
        action =tf.nn.softmax(normal_dist.sample(1)[0])
        log_prob = normal_dist.log_prob(action)

        return action, log_prob

class actor_target(object):
    def __init__(self):
        with tf.name_scope(name='actor_target_net'):
            self.weights1 = tf.Variable(tf.random_normal([N_S, hidden_width1], stddev=0.1))
            self.weights11 = tf.Variable(tf.random_normal([hidden_width1, hidden_width1], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width1, N_A-1], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width1, 1], stddev=0.1))

    def forward(self, state):
        net1 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(state, self.weights1),training=is_train))*10
        net1=tf.layers.dropout(net1,0.5)
        net11 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(net1, self.weights11),training=is_train))*10
        net11=tf.layers.dropout(net11, 0.5)
        action1 = tf.nn.softmax(tf.sigmoid(tf.layers.batch_normalization(tf.matmul(net11, self.weights2),training=is_train)))
        action2 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(net11, self.weights3),training=is_train))*320
        action=tf.concat([action1,action2], 1)
        return action

class actor_eval(object):
    def __init__(self):
        with tf.name_scope(name='actor_eval_net'):
            self.weights1 = tf.Variable(tf.random_normal([N_S, hidden_width1], stddev=0.1))
            self.weights11 = tf.Variable(tf.random_normal([hidden_width1, hidden_width1], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width1, N_A - 1], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width1, 1], stddev=0.1))

    def forward(self, state):
        net1 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(state, self.weights1), training=is_train)) * 10
        net1 = tf.layers.dropout(net1, 0.5)
        net11 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(net1, self.weights11), training=is_train)) * 10
        net11 = tf.layers.dropout(net11, 0.5)
        action1 = tf.nn.softmax(
            tf.sigmoid(tf.layers.batch_normalization(tf.matmul(net11, self.weights2), training=is_train)))
        action2 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(net11, self.weights3), training=is_train)) * 320
        action = tf.concat([action1, action2], 1)
        return action

class critic_target(object):
    def __init__(self):
        with tf.name_scope(name='critic_target_net'):
            self.weights1 = tf.Variable(tf.random_normal([N_S + N_A, hidden_width2], stddev=1))
            self.bias1 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width2, hidden_width2], stddev=1))
            self.bias2 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width2, 1], stddev=1))
            self.bias3 = tf.Variable(tf.random_normal([batch_size, 1], stddev=0.1))

    def forward(self, state, action):
        lc1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(tf.concat((state, action), 1), self.weights1) + self.bias1,
                                          training=is_train))
        lc1 = tf.layers.dropout(lc1, 0.5)
        value_q1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(lc1, self.weights2) + self.bias2, training=is_train))
        value_q1 = tf.layers.dropout(value_q1, 0.5)
        value_q2 = tf.matmul(value_q1, self.weights3) + self.bias3
        return value_q2

class critic_eval(object):
    def __init__(self):
        with tf.name_scope(name='critic_eval_net'):
            self.weights1 = tf.Variable(tf.random_normal([N_S + N_A, hidden_width2], stddev=1))
            self.bias1 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width2, hidden_width2], stddev=1))
            self.bias2 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width2, 1], stddev=1))
            self.bias3 = tf.Variable(tf.random_normal([batch_size, 1], stddev=0.1))

    def forward(self, state, action):
        lc1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(tf.concat((state, action), 1), self.weights1) + self.bias1,
                                          training=is_train))
        lc1 = tf.layers.dropout(lc1, 0.5)
        value_q1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(lc1, self.weights2) + self.bias2, training=is_train))
        value_q1 = tf.layers.dropout(value_q1, 0.5)
        value_q2 = tf.matmul(value_q1, self.weights3) + self.bias3

        return value_q2

class lstm(object):
    def __init__(self, input_width, state_width, hidden_width):
        self.input_width = input_width
        self.state_width = state_width
        self.hidden_width = hidden_width
        self.times = 0
        self.Mfh, self.Mfg = self.init_weight_mat()
        self.Mih, self.Mig = self.init_weight_mat()
        self.Moh, self.Mog = self.init_weight_mat()
        self.Mch, self.Mcg = self.init_weight_mat()
        self.c_list = self.init_state_vec()
        self.h_list = self.init_state_vec()
        self.ft_list = self.init_state_vec()
        self.it_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ut_list = self.init_state_vec()
    def init_weight_mat(self):
        with tf.name_scope(name='contrast_lstm'):
            Mh = tf.Variable(tf.random_normal([self.state_width, self.state_width],stddev=0.1))
            Mg = tf.Variable(tf.random_normal([N_S, self.state_width],stddev=0.1))
        return Mh, Mg
    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(tf.zeros([batch_size,self.state_width]))
        return state_vec_list
    def calc_gate(self, x, Mh, Mg):
        h = self.h_list[self.times - 1]
        net = tf.matmul(h,Mh) + tf.matmul(x,Mg)
        gate = tf.sigmoid(net)
        return gate
    def calc_gate1(self, x, Mh, Mg):
        h = self.h_list[self.times - 1]
        net = tf.matmul(h,Mh) + tf.matmul(x,Mg)
        gate = tf.tanh(net)
        return gate
    def forward(self, x):
        self.times += 1
        ft = self.calc_gate(x, self.Mfh, self.Mfg)
        self.ft_list.append(ft)
        it = self.calc_gate(x, self.Mih, self.Mig)
        self.it_list.append(it)
        ot = self.calc_gate(x, self.Moh, self.Mog)
        self.ot_list.append(ot)
        ut = self.calc_gate1(x, self.Mch, self.Mcg)
        self.ut_list.append(ut)
        c = ft * self.c_list[self.times - 1] + it * ut
        self.c_list.append(c)
        h = ot * tf.tanh(c)
        self.h_list.append(h)
        return h

class mlp(object):
    def __init__(self):
        with tf.name_scope(name='contrast_mlp'):
            self.weights1 = tf.Variable(tf.random_normal([hidden_width3,hidden_width3],stddev=1))
            self.bias1 = tf.Variable(tf.random_normal([batch_size,hidden_width3],stddev=1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width3,hidden_width3],stddev=1))
            self.bias2 = tf.Variable(tf.random_normal([batch_size,hidden_width3],stddev=1))
    def forward(self,x):
        result1 = tf.nn.relu(tf.matmul(x,self.weights1)+self.bias1)
        result2 = tf.matmul(result1,self.weights2)+self.bias2
        return result2

class actor_target_contrast(object):
    def __init__(self):
        with tf.name_scope(name='actor_target_contrast_net'):
            self.weights1 = tf.Variable(tf.random_normal([N_S, hidden_width2], stddev=0.1))
            self.weights11 = tf.Variable(tf.random_normal([hidden_width2, hidden_width2], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width2, N_A - 1], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width2, 1], stddev=0.1))
    def forward(self, state):
        net1 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(state, self.weights1), training=is_train)) * 10
        net1 = tf.layers.dropout(net1, 0.5)
        net11 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(net1, self.weights11), training=is_train)) * 10
        net11 = tf.layers.dropout(net11, 0.5)
        action1 = tf.nn.softmax(
            tf.sigmoid(tf.layers.batch_normalization(tf.matmul(net11, self.weights2), training=is_train)))
        action2 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(net11, self.weights3), training=is_train)) * 320
        action = tf.concat([action1, action2], 1)
        return action

class actor_eval_contrast(object):
    def __init__(self):
        with tf.name_scope(name='actor_eval_contrast_net'):
            self.weights1 = tf.Variable(tf.random_normal([N_S, hidden_width2], stddev=0.1))
            self.weights11 = tf.Variable(tf.random_normal([hidden_width2, hidden_width2], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width2, N_A - 1], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width2, 1], stddev=0.1))
    def forward(self, state):
        net1 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(state, self.weights1), training=is_train)) * 10
        net1 = tf.layers.dropout(net1, 0.5)
        net11 = tf.nn.tanh(tf.layers.batch_normalization(tf.matmul(net1, self.weights11), training=is_train)) * 10
        net11 = tf.layers.dropout(net11, 0.5)
        action1 = tf.nn.softmax(
            tf.sigmoid(tf.layers.batch_normalization(tf.matmul(net11, self.weights2), training=is_train)))
        action2 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(net11, self.weights3), training=is_train)) * 320
        action = tf.concat([action1, action2], 1)
        return action

class critic_eval_contrast(object):
    def __init__(self):
        with tf.name_scope(name='critic_eval_contrast_net'):
            self.weights1 = tf.Variable(tf.random_normal([hidden_width3 + N_A, hidden_width2], stddev=1))
            self.bias1 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width2, hidden_width2], stddev=1))
            self.bias2 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width2, 1], stddev=1))
            self.bias3 = tf.Variable(tf.random_normal([batch_size, 1], stddev=0.1))
    def forward(self, state, action):
        lc1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(tf.concat((state, action), 1), self.weights1) + self.bias1,
                                          training=is_train))
        lc1 = tf.layers.dropout(lc1, 0.5)
        value_q1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(lc1, self.weights2) + self.bias2, training=is_train))
        value_q1 = tf.layers.dropout(value_q1, 0.5)
        value_q2 = tf.matmul(value_q1, self.weights3) + self.bias3
        return value_q2

class critic_target_contrast(object):
    def __init__(self):
        with tf.name_scope(name='critic_target_contrast_net'):
            self.weights1 = tf.Variable(tf.random_normal([hidden_width3 + N_A, hidden_width2], stddev=1))
            self.bias1 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights2 = tf.Variable(tf.random_normal([hidden_width2, hidden_width2], stddev=1))
            self.bias2 = tf.Variable(tf.random_normal([batch_size, hidden_width2], stddev=0.1))
            self.weights3 = tf.Variable(tf.random_normal([hidden_width2, 1], stddev=1))
            self.bias3 = tf.Variable(tf.random_normal([batch_size, 1], stddev=0.1))
    def forward(self, state, action):
        lc1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(tf.concat((state, action), 1), self.weights1) + self.bias1,
                                          training=is_train))
        lc1 = tf.layers.dropout(lc1, 0.5)
        value_q1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(lc1, self.weights2) + self.bias2, training=is_train))
        value_q1 = tf.layers.dropout(value_q1, 0.5)
        value_q2 = tf.matmul(value_q1, self.weights3) + self.bias3
        return value_q2

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,a,[[r]],s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1
    def sample(self, n):
        assert self.pointer >= self.capacity,'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
def takelast(elem):
    return elem[-1]
state = tf.placeholder(tf.float32,[None, N_S]) #[1, 5]
reward = tf.placeholder(tf.float32,[None, 1])
state_ = tf.placeholder(tf.float32,[None, N_S])
a = tf.placeholder(tf.float32,[None, N_A])
a_ = tf.placeholder(tf.float32,[None, N_A])
reward_contrast= tf.placeholder(tf.float32,[None, 1])
state_pg = tf.placeholder(tf.float32,[1, k])
reward_pg = tf.placeholder(tf.float32,[None, 1])
actortarget=actor_target()
actoreval=actor_eval()
critictarget=critic_target()
criticeval=critic_eval()
Replay_buffer=Memory(MEMORY_CAPACITY, dims=2 * N_S + N_A + 1)
Replay_buffer_main=Memory(MEMORY_CAPACITY, dims=2 * N_S + N_A + 1)
contrast_lstm=lstm(N_S,hidden_width3,hidden_width3)
contrast_mlp=mlp()
actortargetcontrast=actor_target_contrast()
actorevalcontrast=actor_eval_contrast()
critictargetcontrast=critic_target_contrast()
criticevalcontrast=critic_eval_contrast()
policygradient_contrast=policygradient()
a_current=actoreval.forward(state)
a_next=actortarget.forward(state_)
value_q_current=criticeval.forward(state,a_current)
value_q_next=critictarget.forward(state_,a_next)
target_q=reward+gamma*value_q_next
loss_critic=tf.reduce_mean(tf.squared_difference(target_q,value_q_current))
loss_actor=-tf.reduce_mean(value_q_current)
l1=contrast_lstm.forward(state)
l2=contrast_lstm.forward(state_)
m1=contrast_mlp.forward(l1)
m2=contrast_mlp.forward(l2)
labels = tf.one_hot(tf.range(24), 24 * 2)
masks = tf.one_hot(tf.range(24), 24)
logits_aa = tf.matmul(m1, m1, transpose_b=True)
logits_aa = logits_aa - masks * LARGE_NUM
logits_bb = tf.matmul(m2, m2, transpose_b=True)
logits_bb = logits_bb - masks * LARGE_NUM
logits_ab = tf.matmul(m1, m2, transpose_b=True)
logits_ba = tf.matmul(m2, m1, transpose_b=True)
loss_a = tf.losses.softmax_cross_entropy(
    labels, tf.concat([logits_ab, logits_aa], 1), weights=1.0)
loss_b = tf.losses.softmax_cross_entropy(
    labels, tf.concat([logits_ba, logits_bb], 1), weights=1.0)
loss_contrast = loss_a + loss_b
a_current_contrast=actorevalcontrast.forward(state)
a_next_contrast=actortargetcontrast.forward(state_)
statemid=contrast_lstm.forward(state)
state_mid=contrast_mlp.forward(statemid)
statemid_=contrast_lstm.forward(state_)
state_mid_=contrast_mlp.forward(statemid_)
value_q_current_contrast=criticevalcontrast.forward(state_mid,a_current_contrast)
value_q_next_contrast=critictargetcontrast.forward(state_mid_,a_next_contrast)
target_q_contrast=reward_contrast+gamma*value_q_next_contrast
loss_critic_contrast=tf.reduce_mean(tf.squared_difference(target_q_contrast,value_q_current_contrast))
loss_actor_contrast=-tf.reduce_mean(value_q_current_contrast)
action_class,log_prob=policygradient_contrast.forward(state_pg)
loss_policy_class = -tf.reduce_mean(log_prob*reward_pg)
train_actor_target=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target_net')
train_actor_eval=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_eval_net')
train_critic_target=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target_net')
train_critic_eval=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_eval_net')
train_contrast_lstm=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='contrast_lstm')
train_contrast_mlp=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='contrast_mlp')
train_actor_target_contrast=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target_contrast_net')
train_actor_eval_contrast=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_eval_contrast_net')
train_critic_target_contrast=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target_contrast_net')
train_critic_eval_contrast=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_eval_contrast_net')
train_pg=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Pg_class_train')
soft_replace1 = [tf.assign(t, (1 - tao) * t + tao * e)
        for t, e in zip(train_critic_target, train_critic_eval)]
soft_replace2 = [tf.assign(t, (1 - tao) * t + tao * e)
        for t, e in zip(train_actor_target, train_actor_eval)]
soft_replace3 = [tf.assign(t, (1 - tao) * t + tao * e)
        for t, e in zip(train_critic_target_contrast, train_critic_eval_contrast)]
soft_replace4 = [tf.assign(t, (1 - tao) * t + tao * e)
        for t, e in zip(train_actor_target_contrast, train_actor_eval_contrast)]
global_step =tf.Variable(0, trainable=False)
LR_A = tf.train.exponential_decay(initial_learning_rate,global_step,LEARNING_RATE_STEP,1,staircase=True)
LR_AA = tf.train.exponential_decay(initial_learning_rate,global_step,LEARNING_RATE_STEP,1,staircase=True)
LR_C = tf.train.exponential_decay(initial_learning_rate,global_step,LEARNING_RATE_STEP,1,staircase=True)
LR_CC = tf.train.exponential_decay(initial_learning_rate,global_step,LEARNING_RATE_STEP,1,staircase=True)
LR_M = tf.train.exponential_decay(initial_learning_rate,global_step,LEARNING_RATE_STEP,1,staircase=True)
LR_P = tf.train.exponential_decay(initial_learning_rate,global_step,LEARNING_RATE_STEP,1,staircase=True)
train_op_critic = tf.train.AdamOptimizer(LR_C).minimize(loss_critic,global_step=global_step,var_list=train_critic_eval)
train_op_actor = tf.train.AdamOptimizer(LR_A).minimize(loss_actor,global_step=global_step,var_list=train_actor_eval)
train_op_lstm_mlp = tf.train.AdamOptimizer(LR_M).minimize(loss_contrast,global_step=global_step,var_list=train_contrast_lstm+train_contrast_mlp)
train_op_actor_contrast = tf.train.AdamOptimizer(LR_AA).minimize(loss_actor_contrast,global_step=global_step,var_list=train_actor_eval_contrast)
train_op_critic_contrast = tf.train.AdamOptimizer(LR_CC).minimize(loss_critic_contrast,global_step=global_step,var_list=train_critic_eval_contrast)
train_op_policygradiant = tf.train.AdamOptimizer(LR_P).minimize(loss_policy_class,global_step=global_step,var_list=train_pg)
init_op = tf.global_variables_initializer()
###############################  training  ####################################
with tf.Session() as sess:
    sess.run(init_op)
    saver = tf.train.Saver()
    data_list = []
    for day in range(length):
        day_init_reward = 0
        for i in range(20):
            s = env.reset(day)
            ep_reward = 0
            ep_state = s
            for j in range(MAX_EP_STEPS):
                a = sess.run(a_current, feed_dict={state: s})
                a = np.random.normal(a, var)
                s_, r,_ = env.forward(a, s, day)
                Replay_buffer.store_transition(s, a, r, s_)
                if Replay_buffer.pointer > MEMORY_CAPACITY:
                    var[0] *= 0.9
                    var[1] *= 0.9
                    var[2] *= 0.9
                    var[3] *= 0.9
                    b_M = Replay_buffer.sample(batch_size1)
                    b_s = b_M[:, :N_S]
                    b_a = b_M[:, N_S: N_S + N_A]
                    b_r = b_M[:, -N_S - 1: -N_S]
                    b_s_ = b_M[:, -N_S:]
                    sess.run(train_op_actor, feed_dict={state: b_s, is_train: True})
                    sess.run(train_op_critic, feed_dict={state: b_s, state_: b_s_, reward: b_r, is_train: True})
                    sess.run(soft_replace2)
                    sess.run(soft_replace1)
                s = s_
                if j < MAX_EP_STEPS - 1:
                    ep_state = np.vstack((ep_state, s))
                else:
                    ep_state = ep_state
                ep_reward += r
        day_init_reward = ep_reward
        data_list.append([ep_state, ep_reward])
        print('day_init_reward', day_init_reward, 'day=', day)
    init_reward = 0
    for i in range(length):
        init_reward += data_list[i][1]
    data_list.sort(key=takelast, reverse=True)
    EP_reward_list = []
    EP_reward_train_list = []
    EP_reward_test_list = []
    EP_current_cost_test_list = []
    days_rewards_best=-100000000
    numb_class =[[20,20,20,20,20]]
    assemble_state = np.zeros([k, length, 24, 5])
    for i in range(int(numb_class[0][0])):
        assemble_state[0, i] = data_list[i][0]
    for i in range(int(numb_class[0][1])):
        assemble_state[1, i] = data_list[i + int(numb_class[0][0])][0]
    for i in range(int(numb_class[0][2])):
        assemble_state[2, i] = data_list[i + int(numb_class[0][0]) + int(numb_class[0][1])][0]
    for i in range(int(numb_class[0][3])):
        assemble_state[3, i] = data_list[i + int(numb_class[0][0]) + int(numb_class[0][1]) + int(numb_class[0][2])][0]
    for i in range(int(numb_class[0][4])):
        assemble_state[4, i] = \
        data_list[i + int(numb_class[0][0]) + int(numb_class[0][1]) + int(numb_class[0][2]) + int(numb_class[0][3])][0]
    for t in range(100):
        for i in range(k):
            for j in range(k):
                train_length = min(int(numb_class[0][i]), int(numb_class[0][j]))
                for p in range(train_length):
                    sess.run(train_op_lstm_mlp, feed_dict={state: assemble_state[i, p].reshape(-1, N_S),
                                                           state_: assemble_state[j, p].reshape(-1, N_S)})
    for i in range(500):
        EP_reward = 0
        for day in range(220):
            s = env_main.reset(day)
            ep_reward_main = 0
            for j in range(MAX_EP_STEPS):
                a = sess.run(a_current_contrast, feed_dict={state: s})  # a[1,3]
                a = np.random.normal(a, varr)
                s_, r,_ = env_main.forward(a, s, day)
                Replay_buffer_main.store_transition(s, a, r, s_)
                if Replay_buffer_main.pointer > MEMORY_CAPACITY1:
                    var[0] *= 0.9  # decay the action randomness
                    var[1] *= 0.9
                    var[2] *= 0.9
                    var[3] *= 0.9
                    b_M = Replay_buffer_main.sample(batch_size1)
                    b_s = b_M[:, :N_S]
                    b_a = b_M[:, N_S: N_S + N_A]
                    b_r = b_M[:, -N_S - 1: -N_S]
                    b_s_ = b_M[:, -N_S:]
                    sess.run(train_op_actor_contrast, feed_dict={state: b_s, state_: b_s_, is_train: True})
                    sess.run(train_op_critic_contrast,feed_dict={state: b_s, state_: b_s_, reward_contrast: b_r, is_train: True})
                    sess.run(soft_replace2)
                    sess.run(soft_replace3)
                s = s_
                ep_reward_main += r
            EP_reward += ep_reward_main
        EP_reward_train_list.append(EP_reward)
        EP_reward_list.append(EP_reward)



