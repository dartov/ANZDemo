from teradataml import *
from teradataml.analytics.valib import *
from teradataml import configure
configure.val_install_location = "VAL"

def score(data_conf, model_conf, **kwargs):
    """Python score method called by AOA framework batch mode

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
    
    model_table = "AOA_MODELS_{}".format(kwargs.get("model_version").split("-", 1)[0])
    
    score = valib.LogRegPredict(data = DataFrame(data_conf["data_table"]), 
                                        model = DataFrame(model_table), 
                                        estimate_column = "predicted_churn")
    
    df = score.result
    df = df.assign(predicted_churn = df.predicted_churn.cast(type_=INTEGER))
    
    df.to_sql(table_name = data_conf["result_table"], if_exists = 'replace')
    
    remove_context()

