import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
import yaml

# train and save classifier 
def main(embedding,
        data,
        model_name,
        n_estimators,
        max_depth,
        learning_rate,
        subsample,            
        colsample_bynode,  
        tree_method, 
        random_state):

    X_train, X_val, y_train, y_val = train_test_split(embedding,
                                                      data["label"],
                                                      test_size=0.2, 
                                                      random_state = 2024)
    
    xgb_gpu = XGBRFClassifier(
        n_estimators = n_estimators,          # Number of trees
        max_depth = max_depth,                # Maximum depth per tree
        learning_rate = learning_rate,        # Learning rate
        subsample = subsample,                # Fraction of samples used per tree
        colsample_bynode = colsample_bynode,  # Fraction of features used per node
        tree_method = tree_method,            # Use GPU for training
        random_state = random_state           # For reproducibility
    )

    xgb_gpu.fit(X_train, y_train)

    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"/mnt/c/Users/Johan/Documents/ITligence/data/models/XGBRFClassifier_{model_name}_{time_string}.json" 
    xgb_gpu.save_model(model_path)

if __name__ == "__main__": 
    opts = yaml.safe_load(open("/mnt/c/Users/Johan/Documents/ITligence/code/python/pipeline/opts.yaml")) 

    main()