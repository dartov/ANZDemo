from teradataml import *
from teradataml.analytics.valib import *
from teradataml import configure
configure.val_install_location = "VAL"

from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """
    
    create_context(host = os.environ["AOA_CONN_HOST"],
                   username = os.environ["AOA_CONN_USERNAME"],
                   password = os.environ["AOA_CONN_PASSWORD"],
                   database = "EP_SDS")
    
    score = valib.LogRegPredict(data = DataFrame(data_conf["data_table"]), 
                                        model = DataFrame(kwargs.get("model_table")), 
                                        estimate_column = "predicted_churn",
                                        accumulate = "is_churn")
    
    df = score.result
    df = df.assign(predicted_churn = df.predicted_churn.cast(type_=INTEGER))
    results = df.select(["is_churn","predicted_churn"]).to_pandas()
    
    y_pred = results[["predicted_churn"]]
    y_test = results[["is_churn"]]
    
    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # create confusion matrix plot
    cf = metrics.confusion_matrix(y_test, y_pred)

    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['0','1'])
    plt.yticks([0, 1], ['0','1'])

    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')

    fig = plt.gcf()
    fig.savefig('artifacts/output/confusion_matrix', dpi=500)
    plt.clf()

    # dump results as json file evaluation.json to models/ folder
    print("Evaluation complete...")
    
    remove_context()
