import multiprocessing

import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK, space_eval



# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)




hybrid = recommender_class(Helper().URM_train_validation, recommenders)



# Step 1 : defining the objective function
def objective(params):
    print("\n############## New iteration ##############\n", params)
    params = {"weights": params}
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'SLIMElasticNetRecommender': hp.hp.uniform('SLIMElasticNetRecommender', 0.75, 1),
    # 'item_cbf_weight': hp.hp.uniform('item_cbf_weight', 0, 0.2),
    'ItemCollaborativeFilter': hp.hp.uniform('ItemCollaborativeFilter', 0.01, 0.04),
    'RP3betaRecommender': hp.hp.uniform('RP3betaRecommender', 0.75, 1),
    # 'UserCBF': hp.hp.quniform('user_cbf_weight', 0, 0.3, 0.0001),
    'ItemCBF': hp.hp.uniform('ItemCollaborative', 0.008, 0.02),
    'AlternatingLeastSquare': hp.hp.uniform('AlternatingLeastSquare', 0.05, 0.2),
    #'SLIM_BPR_Recommender': hp.hp.uniform('SLIM_BPR_Recommender', 0,1)
}


# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 100


# opt = {'SSLIM_weight': 0.8950096358670148, 'item_cbf_weight': 0.034234727663263104, 'item_cf_weight': 0.011497379340447589, 'rp3_weight': 0.8894480634395567, 'user_cbf_weight': 0, 'user_cf_weight': 0}
#
# new_opt = {'SSLIM_weight': 0.8525330515257261, 'item_cbf_weight': 0.03013686377319209, 'item_cf_weight': 0.01129668459365759, 'rp3_weight': 0.9360587800999112, 'user_cbf_weight': 0, 'user_cf_weight': 0}
#
# last_opt = {'SSLIM_weight': 0.8737840927419455, 'item_cbf_weight': 0.037666643326618406, 'item_cf_weight': 0.014294955186782246, 'rp3_weight': 0.9314974601074552, 'user_cbf_weight': 0, 'user_cf_weight': 0}
#
opt = {'AlternatingLeastSquare': 0.07611985905191196, 'ItemCBF': 0.017561491230314447, 'ItemCollaborativeFilter': 0.0341817493248531, 'RP3betaRecommender': 0.9713719890744753, 'SLIMElasticNetRecommender': 0.9974897962716185, 'SLIM_BPR_Recommender': 0.8633266021278376}

# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[opt])

best = space_eval(search_space, best)

# best will the return the the best hyperparameter set

print("\n############## Best Parameters ##############\n")
print(best, "\n\nEvaluating on test set now...")

RunRecommender.evaluate_on_test_set(Hybrid, {"weights": best}, Kfold=N_KFOLD,
                                    init_params={"recommenders": [MultiThreadSLIM_ElasticNet, RP3betaRecommender, ItemCBF, AlternatingLeastSquare, SLIM_BPR_Cython]},
                                    parallelize_evaluation=False,
                                    parallel_fit=False)

computer_sleep(verbose=False)
