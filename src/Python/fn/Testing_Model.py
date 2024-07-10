#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Shows the f1 score, Matthews Correction Coefficient (MCC) and balance accuracy
# of the pretrained model on a test dataset

# Args:
#     model (a classifier model): A model which has been trained already to classify
#                               the NAICS code for the application
 #    X (A numpy array): The input vector of the MLP which is the BERT embedding vector 
#                         concatenated with some categorical variables.

#     Y5 (A numpy array): The output vector of the MLP which is the target variable (NAICS_RANK)
#                          for our classification task.

###################################################################

def TEST_MODEL(model,X, Y5):
    X_TRAIN_VAL, X_TEST, Y5_TRAIN_VAL, Y5_TEST = train_test_split(X, Y5,
        test_size=0.1, shuffle = True)

    X_VAR=X_TEST
    Y_VAR=Y5_TEST
    model.eval()

    loss_val_total = 0
    predictions5,  true_vals5  =  [], []
    last_results_prediction=[]   
    last_results_target5=[]

    obs_index=[]


    for j in range(len(X_VAR)//batch_size):
        inp = torch.from_numpy(X_VAR[(j*batch_size):((j+1)*batch_size)])
        #print(inp.shape)
        target5 = torch.from_numpy(Y_VAR[(j*batch_size):((j+1)*batch_size)])

        with torch.no_grad():        
                out = model(inp)

        #print(out.detach().numpy().shape)
        loss = criterion(out, target5, log_var_e)
        #print(loss)
        loss_val_total += loss.item()


        predictions5.append(out.detach().numpy())

        last_results_prediction=out.detach().numpy()
        last_results_target5=target5.numpy()
        true_vals5.append(target5.numpy())
        temp_obs_index=range(j*batch_size,(j+1)*batch_size)
        obs_index.append(temp_obs_index)


    loss_val_avg = loss_val_total * batch_size /len(X_VAR)
    predictions5 = np.concatenate(predictions5, axis=0)
    true_vals5 = np.concatenate(true_vals5, axis=0)

    print("F1_score is: ",f1_score_func(predictions5, true_vals5))
    print("Matthews Correction Coefficient (MCC) is: ", matthews_corrcoef(np.argmax(predictions5, axis=1).flatten(), true_vals5))
    print("Balanced Accuracy is: ",BALANCED_ACCURACY(predictions5, true_vals5))


