import logging
from future_sales_prediction_2024 import *

#Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')


def main(preparation_required: bool = False):

    """
    Main function for executing the pipeline.
    
    Parameters:
    - preparation_required (bool): Whether data preparation steps are needed (e.g., pulling data, feature extraction)
    - model: Model to use for training. Defaults to XGBRegressor
    """

    #Step 1. Pull data using DVC
    print('Pulling data from remote storage')
    dvc_manager = DVCDataManager()
    dvc_manager.pull_data()

    if preparation_required:
        #Data Handling using pulled raw data
        preparer = DataPreparer(MemoryReducer)

        items = pd.read_csv('./data_pulled/items.csv')
        categories = pd.read_csv('./data_pulled/item_categories.csv')
        train = pd.read_csv('./data_pulled/train.csv')
        shops = pd.read_csv('./data_pulled/shops.csv')
        test = pd.read_csv('./data_pulled/test.csv')

        full_data, train = prepare_full_data(items = items, categories = categories, train = train, shops = shops, test = test)

        #Feature Exctration
        feature_extractor = FeatureExtractor(full_data, train, MemoryReducer())
        full_featured_data = feature_extractor.process()

    else:
        data_path = './data_pulled/full_featured_data.csv'
        if os.path.exists(data_path):
            print(f'Using existing data: {data_path}')
            full_featured_data = pd.read_csv(data_path)
        else:
            raise Exception(f'Preprocessed data in not found in the directory {data_path}')
    logging.info("Data extraction completed successfully")

    #Step 2. Data Validation
    print('Data Validation')
    column_types = {'date_block_num': 'int8', 'shop_id': 'int8', 'city_id': 'int8', 'item_id': 'int16', 'item_cnt_month': 'float16',
    'item_category_id': 'int8', 'main_category_id': 'int8', 'minor_category_id': 'int8'}

    values_ranges = {'date_block_num': (0, 34), 'shop_id': (0, 59), 'item_id': (0, 22169), 'item_cnt_month': (0, 669), 'city_id':(0,31),
                'item_category_id': (0,83), 'main_category_id': (0,11), 'minor_category_id': (0, 66)}

    validator = Validator(column_types, values_ranges, check_duplicates = True, check_missing = True)
    validator.transform(full_featured_data)
    logging.info("Data validation completed successfully")



    #Step 3. Data Split
    print('Data Split')
    trainer = Trainer()
    X_train, y_train, X_test = trainer.split_data(full_featured_data)
    logging.info("Data split completed successfully")

    #Step 4. Feature Importance layer of Baseline model with RandomForestRegressor.
    print('Feature importance calculations with Baseline Model')
    importance_layer = FeatureImportanceLayer(X_train, y_train)
    importance_layer.fit_baseline_model().plot_baseline_importance()
    logging.info("Feature importance calculations with Baseline Model completed successfully")

    #Step 5. Hyperparameter tunning
    print('Hyperparameter tunning')
    tuner = HyperparameterTuner(config_path="config.yaml")
    best_params, model_name = tuner.tune(X_train, y_train, model_name="XGBRegressor")
    logging.info("Hyperparameter tunning completed successfully")

    #Step 6. Model Training
    print('Model training')
    y_pred, model, rmse = trainer.train_predict(X_train, y_train, X_test, model_name, best_params)
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.4f}")
    logging.info("Model training completed successfully")

    #Step 7. Feature Importance layer with Final Model
    print(f'Feature importance calculations with {model_name}')
    feature_layer.fit_final_model(params = best_params)
    feature_layer.plot_final_importance()
    logging.info(f"Feature importance calculations with {model_name} completed successfully")

    #Step 8. Explainability layer
    print('Explainability layer of trained model')
    expl_layer = Explainability(model = model, X = X_test)
    expl_layer.explaine_instance()
    expl_layer.global_feature_importance()
    for feature in X_train.columns:
        expl_layer.feature_dependence(feature)
    logging.info("Shap values calculations completed successfully")
    logging.info("Explainability plots for global data, one random instance and all features are saved")


    #Step 9. Error Analysis layer
    print('Error Analysis layer')
    error_layer = ErrorAnalysis(X_train, y_train)
    error_layer.train_predict()
    error_layer.model_drawbacks()
    error_layer.large_target_error()
    error_layer.influence_on_error_rate()
    logging.info("Error analysis plots are saved")


if __name__ == 'main':
    main()




    



    



    

