import numpy as np

def softmax(z):
    a = np.exp(z)
    sigma = sum(a)
    return a / sigma


class RNN(object):
    def __init__(self, word_dim, n_hidden):
        self.n_hidden = n_hidden
        self.word_dim = word_dim

    def init_params(self):
        self.U = np.random.uniform(-np.sqrt(1./self.word_dim),
                                   np.sqrt(1./self.word_dim), (self.n_hidden, self.word_dim))
        self.W = np.random.uniform(-np.sqrt(1./self.word_dim),
                                   np.sqrt(1./self.word_dim), (self.n_hidden, self.n_hidden))
        self.V = np.random.uniform(-np.sqrt(1./self.word_dim),
                                   np.sqrt(1./self.word_dim), (self.word_dim, self.n_hidden))
        print "parameters initialized"

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T+1, self.n_hidden))
        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]]+self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return o, s

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, X_train, Y_train):
        J = 0
        for i in np.arange(len(Y_train)):
            o, s = self.forward_propagation(X_train[i])
            word_probs = o[np.arange(len(Y_train[i])), Y_train[i]]
            J += np.sum(np.log(word_probs))
        J = -1 * J / np.sum(len(y) for y in Y_train)
        return J

    def bptt(self, x, y):
        # Suppose at unit t
        # forward propagation:
        #   s_t = tanh(U*x_t + W*s_(t-1))
        #   o_t = softmax(V*s_t)
        #
        # back propagation:
        #   z1 = V*s_t
        #   z2 = U*x_t + W*s_(t-1)
        #   delta_o[t] = dL/dz1 = o_t-y_t
        #   grad_V = dL/dV = dL/dz1 * dz1/dV
        #   for j in range(0, t+1):
        #       delta_s_j = dL/dz2_j = dL/dz1 * dz1/ds_t * ds_t/ds_j * ds_j/dz2_j
        #   grad_W = sum(delta_s_j * dz2_j/ds_j)
        #

        T = len(y)
        U = self.U
        V = self.V
        W = self.W
        o, s = self.forward_propagation(x)

        grad_U = np.zeros(U.shape)
        grad_V = np.zeros(V.shape)
        grad_W = np.zeros(W.shape)

        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(T)[::-1]:
            grad_V += np.outer(delta_o[t], s[t])
            delta_s = delta_o[t].dot(V) * (1-s[t]**2)
            for j in np.arange(0, t+1)[::-1]:
                grad_W += np.outer(delta_s, s[j-1])
                grad_U[:, x[j]] += delta_s
                delta_s = delta_s.dot(W) * (1-s[j-1]**2)

        return grad_V, grad_W, grad_U

    def train(self, X_train, Y_train, learning_rate, n_epoch):
        print "Start training ..."
        for epoch in np.arange(n_epoch):
            loss = self.calculate_loss(X_train, Y_train)
            print "epoch", epoch, "loss =", loss
            for i in np.arange(len(Y_train)):
                grad_V, grad_W, grad_U = self.bptt(X_train[i], Y_train[i])
                self.U -= learning_rate * grad_U
                self.W -= learning_rate * grad_W
                self.V -= learning_rate * grad_V
