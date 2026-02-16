def load_dataset(cfg):
    """
    Must return:
      X_train, X_test, y_train, y_test

    Requirements:
    - Load CICIDS2018 CSV(s) from cfg.data.root and cfg.data.files
    - Binary labels: Benign -> 0, any attack -> 1
    - Numeric features only
    - Handle NaN/Inf (drop or impute, but must document)
    - Stratified split using cfg.data.test_size and cfg.run.seed
    - Standard scaling (fit on train, transform test)
    """
    raise NotImplementedError("Implement dataset loading + preprocessing.")