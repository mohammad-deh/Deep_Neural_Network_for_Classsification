#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#################################################################

# Creates the structure of a Multi Layer Perceptron in which we have 2 hidden layers
# with relu functions on them as the activatuion function.
# The input layer size is our input vector that has already been created by MLP_INPUT function.
# The output layer size is our output vector that has already been created by MLP_OUTPUT function.
# The size the the first hidden layer will be identified later on and the size of the second 
# one is 128.
###################################################################

class Net(nn.Module):
    def __init__(self, input_size, hidden_size,  output5_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128) 
        #self.relu = nn.ReLU()
        
        self.fc7 = nn.Linear(128, output5_size)
        
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        
        out5 = self.fc7(out)
        return  out5
    
    



###################################################################

# Builds a model based on the NET class (MLP).

# Args:
#      X (A numpy array): The input vector of the MLP which is the BERT embedding vector 
#                         concatenated with some categorical variables.

#      Y5 (A numpy array): The output vector of the MLP which is the target variable (NAICS_RANK)
#                          for our classification task.
# return: 
#         model (an MLP): A deep neural network
#######################################################################

def BUILD_MODEL(X,Y5):
    #fitting parameters
    nb_features = 256
    Q = len(X[1])
    D5=len(np.unique(Y5))

    model = Net(Q, nb_features, D5)
    
    return model





#################################################################

# Evaluate the loss function in the last layer of MLP which is the cross entropy loss function.
# 
# Args: 
#   y_pred (A numpy array): The predicted target variable.
#   y_true (A numpy Array): The actual target variable.
#   log_vars (An scaler): An embedded scaler to the loss function to avoid overfitting

# Return:
#        loss (A number): The value of a loss function 
###################################################################

def criterion(y_pred, y_true, log_vars):
  
    precision = torch.exp(-log_vars)
        #diff = -logsoft_fun(y_pred[i]-y_true[i])
    
    loss_function = nn.CrossEntropyLoss()
    diff = loss_function(y_pred,  y_true.long())
        #diff = (y_pred[i]-y_true[i])**2.
        
    loss = precision * diff + log_vars
    
        
    return loss




#############################################################

# Shuffles the dataset
# Args:
#     X (a numpy array): A vector
#     Y5 (a numpy array): A vector
# Returns:
#     x[s] (a numpy array): A vector
#     Y5[s] (a numpy array): A vector

# Note that the vector X and Y5 should have the same length.
###############################################################

def shuffle_data(X,Y5):
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    return X[s],  Y5[s]






###################################################################

# Evaluates the loss function as well as the predicted target variablue.

# Args:
#      X (A numpy array): The input vector of the MLP which is the BERT embedding vector 
#                         concatenated with some categorical variables.

#      Y5 (A numpy array): The output vector of the MLP which is the target variable (NAICS_RANK)
#                          for our classification task.
# return: 
#         loss_val_avg (a number): The value of the loss function
#         predictions5 (a numpy array): the predicted target variable
#         true_vals5 (a numpy array): The true target variable
#######################################################################

def evaluate(X, Y5):

    model.eval()
    
    loss_val_total = 0
    predictions5,  true_vals5  =  [], []
    
    for j in range(len(X)//batch_size):
        
        
        
        inp = torch.from_numpy(X[(j*batch_size):((j+1)*batch_size)])
        
        target5 = torch.from_numpy(Y5[(j*batch_size):((j+1)*batch_size)])
        
        with torch.no_grad():        
            out = model(inp)
        
        
        
        loss = criterion(out, target5, log_var_e)
        
        loss_val_total += loss.item()
        
        
        predictions5.append(out.detach().numpy())
        
        true_vals5.append(target5.numpy())
        
    
    
    loss_val_avg = loss_val_total * batch_size /len(X)
    
    predictions5 = np.concatenate(predictions5, axis=0)
    
    
    true_vals5 = np.concatenate(true_vals5, axis=0)
    
            
    return loss_val_avg, predictions5,  true_vals5






##################################################################

# Evaluates the f1 score between two arrays.

# Args:
#      preds (A numpy array): the predicted target variable in which each element shows
#                           a vector that has as the same length as the number of classes
#                           and each element of the vector shows the probabilty of 
#                           an applicatio belonging to a class. 
#                           

#      labels (A numpy array): The true target variable which shows the groundtruth label
#                           (true class) of each application                     

# return: f1 score between two numpy arrays. 
#         
#######################################################################

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')





##################################################################

# Evaluates the Balance accuracy score between two arrays.

# Args:
#      preds (A numpy array): the predicted target variable in which each element shows
#                           a vector that has as the same length as the number of classes
#                           and each element of the vector shows the probabilty of 
#                           an applicatio belonging to a class. 
#                           

#      labels (A numpy array): The true target variable which shows the groundtruth label
#                           (true class) of each application                     

# return: Balance accuracy score between two numpy arrays. 
#         
#######################################################################

def BALANCED_ACCURACY(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return balanced_accuracy_score(labels_flat, preds_flat)





# Calculates the f1 score for each epoch for which the model is being trained.

# Args:
#   X_TRAIN (a numpy array): The training part of the input vector
#   Y5_TRAIN (a numpy array): The training part of the output vector
#   X_TEST (a numpy array): The testing part of the input vector
#   Y5_TEST (a numpy array): The testing part of the output vector
#   X_VAL (a numpy array): The validation part of the input vector
#   Y5_VAL (a numpy array): The validation part of the output vector

# Returns:
#         test_f1_5_list (a list of numbers): The calculated f1 score for each epoch

##############################################################################

def inner_cv_test(X_TRAIN,Y5_TRAIN,X_TEST,Y5_TEST,X_VAL,Y5_VAL):
    N = len(X_TRAIN)
    loss_history = np.zeros(nb_epoch)
    loss_history_val = np.zeros(nb_epoch)
    loss_history_test = np.zeros(nb_epoch)

    val_f1_5_list = np.zeros(nb_epoch)
    test_f1_5_list = np.zeros(nb_epoch)
    for i in range(nb_epoch):

        epoch_loss = 0
    
        X,  Y5 = shuffle_data(X_TRAIN, Y5_TRAIN)
        #X, Y1, Y2 = X_TRAIN, Y1_TRAIN, Y2_TRAIN
        for j in range(N//batch_size):

            optimizer.zero_grad()
            #print(X.shape[1])
            inp = torch.from_numpy(X[(j*batch_size):((j+1)*batch_size)])
            #print(inp.shape[1])
            target5 = torch.from_numpy(Y5[(j*batch_size):((j+1)*batch_size)])

            out = model(inp)

            loss = criterion(out, target5, log_var_e)

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()


        loss_history[i] = epoch_loss * batch_size / N 

        loss_val_avg, predictions5, true_vals5 = evaluate(X_VAL, Y5_VAL)
        #print("Validation loss: ", loss_val_avg)
        loss_history_val[i] = loss_val_avg


        val_f1_5 = f1_score_func(predictions5, true_vals5)
        val_f1_5_list[i] = val_f1_5



        ##############
        loss_test_avg, predictions5_test, true_test5 = evaluate(X_TEST, Y5_TEST)
        #print("Validation loss: ", loss_val_avg)
        loss_history_test[i] = loss_test_avg


        test_f1_5 = f1_score_func(predictions5_test, true_test5)
        #print(predictions5_test)
        test_f1_5_list[i] = test_f1_5
        #################

        print("Epoch: ", i+1)

        print("F1 Score (Weighted) for NAICS in validation set: ", val_f1_5)
        print("F1 Score (Weighted) for NAICS in test set: ", test_f1_5)
        
    
    return test_f1_5_list[i]






######################################################################

# Oversamples the dataset in order to have an imbalanced dataset using SMOTE() function and
# splitting the data into train, validation and test parts and then call the inner_cv_test()
# function to train the model

# Args:
#      X (A numpy array): The input vector of the MLP which is the BERT embedding vector 
#                         concatenated with some categorical variables.

#      Y5 (A numpy array): The output vector of the MLP which is the target variable (NAICS_RANK)
#                          for our classification task.
# Returns: 
#         return_vec (a list of number): The calculated f1 score for each epoch.

############################################################################


def iteration_cv_run(X,Y5):
    

    X_TRAIN_VAL, X_TEST, Y5_TRAIN_VAL, Y5_TEST = train_test_split(X, Y5,
        test_size=0.1, shuffle = True)

    # transform the dataset
    oversample = SMOTE()
    X_TRAIN_VAL, Y5_TRAIN_VAL = oversample.fit_resample(X_TRAIN_VAL, Y5_TRAIN_VAL)


    X_TRAIN, X_VAL, Y5_TRAIN, Y5_VAL = train_test_split(X_TRAIN_VAL, Y5_TRAIN_VAL, 
        test_size=0.2)
    


    return_vec=inner_cv_test(X_TRAIN,Y5_TRAIN,X_TEST,Y5_TEST,X_VAL,Y5_VAL)
    return return_vec


# fits the model on the dataset.
# Args:
#      X (A numpy array): The input vector of the MLP which is the BERT embedding vector 
#                         concatenated with some categorical variables.

#      Y5 (A numpy array): The output vector of the MLP which is the target variable (NAICS_RANK)
#                          for our classification task.
# Returns: 
#         model (a deep neural network): A fitted model based on our dataset.

############################################################################

def FIT_MODEL(X, Y5):
    
    global model
    model = BUILD_MODEL(X,Y5)
    
    global nb_epoch
    nb_epoch = 30
    global batch_size 
    batch_size = 64
    global log_var_e 
    log_var_e = torch.zeros((1,), requires_grad=True)
    global params 
    params = ([p for p in model.parameters()] +  [log_var_e] )
    global optimizer 
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    
    seq_len = 11
    desired_runs_seq=range(0,seq_len)
    for k in desired_runs_seq:
        print(k)
        if k == seq_len-1:
            nb_epoch = 1
        iteration_cv_run(X,Y5)
            
    return model

