from teradataml import *
from teradataml.analytics.valib import *
from teradataml import configure
configure.val_install_location = "VAL"


def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]
    
    create_context(host = os.environ["AOA_CONN_HOST"],
                   username = os.environ["AOA_CONN_USERNAME"],
                   password = os.environ["AOA_CONN_PASSWORD"],
                   database = "EP_SDS")

    # load data & engineer
    training_df = DataFrame(data_conf["data_table"]).sample(frac = float(hyperparams["sampling"]))

    print("Starting training...")

    # fit model to training data
    
    model = valib.LogReg(data = training_df,
                        columns = "largest_jump_payment,avg_payment,time_since_last,tot_no_bolton,tenure,num_visits,tot_no_postpaid_svcs,tot_no_prod,tot_no_svcs,tot_no_fixed_svcs,tot_no_mbb_svcs",
                        exclude_columns = "PRTY_ID",
                        response_column = "is_churn")

    print("Finished training")

    # saving model dataframes in the database so it could be used for evaluation and scoring
    
    model.model.to_sql(table_name = kwargs.get("model_table"), if_exists = 'replace')
    model.statistical_measures.to_sql(table_name = kwargs.get("model_table") + "_rpt", if_exists = 'replace')


    print("Saved trained model")
    
    remove_context()