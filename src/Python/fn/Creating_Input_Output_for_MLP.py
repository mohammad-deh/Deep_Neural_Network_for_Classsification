#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################################
# 
# Preprocessing function
# Cleanses the NAICS column, extraction the year and the month of 
# the date column and filtering out the applications that have been file before 2017.

# Args:
#   DF (dataframe): The main dataset with variables such as file number, firm size, file dating,
#                     and NAICS code

# Returns: 
#  MAIN_DF (dataframe): A dataframe in which the NAICS column has been cleaned and date column 
#                       has been splitted into two columns YEAR and MONTH.
################################################################################

def PREPROCESSING(DF):
    import ipynb  
    from ipynb.fs.full.my_functions_01242023 import model
    from ipynb.fs.full.my_functions_01242023 import get_cls_embedding
    
    
    global model_type
    model_type = input('Please insert the type of BERT embedding model:\n base or large?')
    
    model_picked = model("bert-"+model_type+"-uncased")
    
    if model_type=="large":
        EMBED_LEN = 1024
    elif model_type=="base":
        EMBED_LEN = 768
    else:
        print("The model type you chose is incorrect.")

    
    col_x = []
    for i in range(EMBED_LEN):
        col_x.append("X" + str(i+1))
        
    
    TEXT = input("Insert the name of a text column: ")
    list_test = DF[TEXT].apply(lambda x: get_cls_embedding(model_picked,x))
    ddf = pd.DataFrame(list(list_test), columns = col_x)
    DF_CONCAT = pd.concat([DF, ddf], axis=1)
    
    global COL
    COL = input("Insert the name of the NAICS column to be cleaned: ")
    CLEANED_DF = DF_CONCAT[DF_CONCAT[COL] != "000000"]
    CLEANED_DF = CLEANED_DF[CLEANED_DF[COL] != "      "]
    CLEANED_DF=CLEANED_DF.dropna(subset=[COL])
    CLEANED_DF[COL] = CLEANED_DF[COL].astype(int)
    CLEANED_DF = CLEANED_DF[CLEANED_DF[COL] != 0]
    CLEANED_DF[COL] = CLEANED_DF[COL].astype(str)
    CLEANED_DF[COL]=CLEANED_DF[COL].str[:6]
    
    DATE = input("Insert the name of the date column: ")

    CLEANED_DF["YEAR"] = pd.DatetimeIndex(CLEANED_DF[DATE]).year
    CLEANED_DF["MONTH"] = pd.DatetimeIndex(CLEANED_DF[DATE]).month
    CLEANED_DF=CLEANED_DF[CLEANED_DF['YEAR']>2016]
    CLEANED_DF=CLEANED_DF.reset_index(drop=True)
    
    return CLEANED_DF



################################################################################
# 
# Aggregating function
# Sums the BERT embedding vectors came from the observations in each application.
# Here in our case, the large based BERT embedding has 1024 elements (X1,...,X1024).
# Args:
#   DF (dataframe): A cleaned dataset which is the output of the preprocessing function.
# Returns: 
#  DF_ (dataframe): An aggregated dataframe based on embedding vectors.
################################################################################

def AGGREGATE(DF):
    FILE_NUMBER = input("Insert the name of the unique file number column: ")
    
    if model_type=="large":
        EMBED_LEN = 1024
    elif model_type=="base":
        EMBED_LEN = 768
    else:
        print("The model type you've chosen is incorrect.")

    col_x = []
    for i in range(EMBED_LEN):
            col_x.append("X" + str(i+1))

            
    DF_GROUP = DF.groupby(FILE_NUMBER)[col_x].sum().reset_index()
    
    DF_NOT_X = DF.drop(col_x, axis=1)


    DF_GROUP_MERGE = DF_NOT_X.merge(DF_GROUP,
                                on=[FILE_NUMBER],
                                            how='left')
    DF_ = DF_GROUP_MERGE.drop_duplicates([FILE_NUMBER], keep='last')
    
    return DF_





################################################################################
# 
# Extract function
# Extracts the applications associated with NAICS codes which have more than 10 applications.
# This is because the smote() function for oversampling the data does not work 
# for low-frequent NAICS classes. It also ranks the NAICS code in a descending way and
# label them as NAICS_RANK.

# Args:
#   DF (dataframe): An aggregated dataframe, per NAICS per application

# Returns: 
#  CLEANED_DF_UPD_JOIN (dataframe): A dataframe in which low-frequent NAICS codes are excluded
#  and some main variables have been put on first columns of the dataframe.

################################################################################

def EXTRACT(DF):
    #COL = input("Insert the name of the NAICS column to be counted for ranking and filtering: ")
    NAICS_CODE_SUMMARY=DF.groupby(by=[COL]).count()

    NAICS_CODE_SUMMARY=DF.assign(
        NUM_FILINGS = 
        DF
        .groupby([COL])[COL].transform('count')
        
    ) 

    NAICS_CODE_SUMMARY=NAICS_CODE_SUMMARY[[COL,'NUM_FILINGS']]
    NAICS_CODE_SUMMARY=NAICS_CODE_SUMMARY.reset_index(drop=True)
    NAICS_CODE_SUMMARY=NAICS_CODE_SUMMARY.sort_values(by=['NUM_FILINGS'], ascending=False)
    NAICS_CODE_SUMMARY=NAICS_CODE_SUMMARY.drop_duplicates(subset=[COL])
    NAICS_CODE_SUMMARY=NAICS_CODE_SUMMARY[NAICS_CODE_SUMMARY['NUM_FILINGS']>10]
    NAICS_CODE_SUMMARY=NAICS_CODE_SUMMARY.reset_index(drop=True)
    NAICS_CODE_SUMMARY['NAICS_RANK']=NAICS_CODE_SUMMARY.index

    CLEANED_DF_UPD=DF.copy()
    CLEANED_DF_UPD=CLEANED_DF_UPD[CLEANED_DF_UPD[COL].isin(NAICS_CODE_SUMMARY[COL])]
    CLEANED_DF_UPD=CLEANED_DF_UPD.reset_index(drop=True)

    CLEANED_DF_UPD_JOIN=CLEANED_DF_UPD.merge(NAICS_CODE_SUMMARY,
                                on=[COL],
                                            how='left')
    # Rearranging some columns

    NAICS_RANK_TEMP = CLEANED_DF_UPD_JOIN['NAICS_RANK']
    CLEANED_DF_UPD_JOIN = CLEANED_DF_UPD_JOIN.drop(columns=['NAICS_RANK'])
    CLEANED_DF_UPD_JOIN.insert(loc=1, column='NAICS_RANK', value=NAICS_RANK_TEMP)

    NUM_FILINGS_TEMP = CLEANED_DF_UPD_JOIN['NUM_FILINGS']
    CLEANED_DF_UPD_JOIN = CLEANED_DF_UPD_JOIN.drop(columns=['NUM_FILINGS'])
    CLEANED_DF_UPD_JOIN.insert(loc=2, column='NUM_FILINGS', value=NUM_FILINGS_TEMP)

    YEAR_TEMP = CLEANED_DF_UPD_JOIN['YEAR']
    CLEANED_DF_UPD_JOIN = CLEANED_DF_UPD_JOIN.drop(columns=['YEAR'])
    CLEANED_DF_UPD_JOIN.insert(loc=3, column='YEAR', value=YEAR_TEMP)

    YEAR_TEMP = CLEANED_DF_UPD_JOIN['MONTH']
    CLEANED_DF_UPD_JOIN = CLEANED_DF_UPD_JOIN.drop(columns=['MONTH'])
    CLEANED_DF_UPD_JOIN.insert(loc=4, column='MONTH', value=YEAR_TEMP)
    
    return CLEANED_DF_UPD_JOIN
    

    

    
    
################################################################################
# 
# Making input variable function for Multi Layer Perceptron (MLP)
# Makes the input variable for MLP, based on the aggregated BERT embedding vector and some 
# categorical variables concatenated to the embedding vector.

# Args:
#   DF (dataframe): A dataframe with aggregated embedding vector and some 
# categorical variables such as te firm size, year, .... 

# Returns: 
#  X (A numpy array): A vector of the size of embedding vector plus some dummy variables
# concatenated to that. 
######################################################################################

def MLP_INPUT(DF):
    
    if model_type=="large":
        EMBED_LEN = 1024
    elif model_type=="base":
        EMBED_LEN = 768
    else:
        print("The model type you've chosen is incorrect.")

    col_x = []
    for i in range(EMBED_LEN):
        col_x.append("X" + str(i+1))
        
    SEQUENCES_NP = DF[col_x].values.tolist()
    SEQUENCES_NP = np.array(SEQUENCES_NP)
    SEQUENCES_NP = SEQUENCES_NP.astype(np.float32)
    SEQUENCES_NP = torch.tensor(SEQUENCES_NP)

    sequences_np_stack_to_use = np.row_stack(SEQUENCES_NP)
    X_base_0 = sequences_np_stack_to_use.astype('float32')
    X_base=pd.DataFrame(X_base_0)
    X_base['index']=X_base.index
    
    DUMMY_LIST = []
    proc = True
    while proc:
        var = input("Enter a variable to be concatenated to the BERT Embedding vector,\n otherwise type 'no': ")
        if var == "no":
            proc = False
        else:
            DUMMY_LIST.append(var)
  
    X = X_base.copy()
    for var in DUMMY_LIST:
        VAR_DUMMY_DF = pd.get_dummies(DF[var])
        VAR_DUMMY_DF['index']=VAR_DUMMY_DF.index
        
        X = VAR_DUMMY_DF.merge(X,on=['index'],how='left')
        
    X = X.drop(columns=['index'])  
    X=X.to_numpy()
    X=X.astype('float32')

    return X





################################################################################
# 
# Making output variable function for Multi Layer Perceptron (MLP)
# Makes the output variable for MLP, based on the target variable (NAICS_RANK)in our dataframe. 

# Args:
#   DF (dataframe): A dataframe with a target variable (NAICS_RANK in our case).  

# Returns: 
#  Y5 (A numpy array): A vector of the size of the number of applications in our dataframe.
######################################################################################

def MLP_OUTPUT(DF):
    
    TARGET = input("Enter the name of the target variable: ")
   
    Y5 = DF[TARGET].astype('int32')
    Y5 = np.array(list(Y5))
    
    return Y5


   

