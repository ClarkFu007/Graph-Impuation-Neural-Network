import numpy as np
import pandas as pd
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('colheader_justify', 'center')


# MIDA
Mean_values = pd.Series({'Time Step (FE) = 1': 4.0234, 'Time Step (TC) = 1': 4.0234,
                         'Time Step (FE) = 4': 4.0234, 'Time Step (TC) = 4': 4.0234,
                         'Time Step (FE) = 16': 4.0234, 'Time Step (TC) = 16': 4.0234,
                         'Time Step (FE) = 32': 4.0234, 'Time Step (TC) = 32': 4.0234,
                         'Time Step (FE) = 96': 4.0234, 'Time Step (TC) = 96': 4.0234})

MAE1 = pd.Series({'Time Step (FE) = 1': 0.0380, 'Time Step (TC) = 1': 0.0380,
                  'Time Step (FE) = 4': 2.0170, 'Time Step (TC) = 4': 0.0028,
                  'Time Step (FE) = 16': 1.8500, 'Time Step (TC) = 16': 0.0023,
                  'Time Step (FE) = 32': 1.7841, 'Time Step (TC) = 32': 0.0021,
                  'Time Step (FE) = 96': 2.4233, 'Time Step (TC) = 96': 0.0019})

RMSE1 = pd.Series({'Time Step (FE) = 1': 0.0719, 'Time Step (TC) = 1': 0.0719,
                   'Time Step (FE) = 4': 3.9240, 'Time Step (TC) = 4': 0.0072,
                   'Time Step (FE) = 16': 4.0118, 'Time Step (TC) = 16': 0.0058,
                   'Time Step (FE) = 32': 3.9960, 'Time Step (TC) = 32': 0.0049,
                   'Time Step (FE) = 96': 4.7800, 'Time Step (TC) = 96': 0.0049})

Mean_train_time1 = pd.Series({'Time Step (FE) = 1': 4.6757, 'Time Step (TC) = 1': 4.6757,
                              'Time Step (FE) = 4': 4.8271, 'Time Step (TC) = 4': 7.5820,
                              'Time Step (FE) = 16': 6.8588, 'Time Step (TC) = 16': 15.0926,
                              'Time Step (FE) = 32': 12.2331, 'Time Step (TC) = 32': 30.9461,
                              'Time Step (FE) = 96': 43.2654, 'Time Step (TC) = 96': 93.9695})
Mean_impute_time1 = pd.Series({'Time Step (FE) = 1': 0.0005, 'Time Step (TC) = 1': 0.0005,
                               'Time Step (FE) = 4': 0.0005, 'Time Step (TC) = 4': 0.0001,
                               'Time Step (FE) = 16': 0.0007, 'Time Step (TC) = 16': 0.0001,
                               'Time Step (FE) = 32': 0.0010, 'Time Step (TC) = 32': 0.0001,
                               'Time Step (FE) = 96': 0.0038, 'Time Step (TC) = 96': 0.0001})
data1 = pd.DataFrame({'Mean value': Mean_values,
                      'Mean Absolute Error': MAE1,
                      'Root Mean Squared Error': RMSE1,
                      'Mean train time/seconds': Mean_train_time1,
                      'Mean impute time/seconds': Mean_impute_time1})
print('\n')
print("When the noise percentage is 10%, the 'MIDA' experiment results are \n", data1)

MAE2 = pd.Series({'Time Step (FE) = 1': 0.0399, 'Time Step (TC) = 1': 0.0399,
                  'Time Step (FE) = 4': 1.7384, 'Time Step (TC) = 4': 0.0028,
                  'Time Step (FE) = 16': 1.7237, 'Time Step (TC) = 16': 0.0023,
                  'Time Step (FE) = 32': 1.9459, 'Time Step (TC) = 32': 0.0021,
                  'Time Step (FE) = 96': 2.4356, 'Time Step (TC) = 96': 0.0019})

RMSE2 = pd.Series({'Time Step (FE) = 1': 0.0712, 'Time Step (TC) = 1': 0.0712,
                   'Time Step (FE) = 4': 4.3432, 'Time Step (TC) = 4': 0.0072,
                   'Time Step (FE) = 16': 4.1009, 'Time Step (TC) = 16': 0.0058,
                   'Time Step (FE) = 32': 4.2501, 'Time Step (TC) = 32': 0.0048,
                   'Time Step (FE) = 96': 4.8675, 'Time Step (TC) = 96': 0.0049})
Mean_train_time2 = pd.Series({'Time Step (FE) = 1': 4.4128, 'Time Step (TC) = 1': 4.4128,
                              'Time Step (FE) = 4': 5.2318, 'Time Step (TC) = 4': 7.4807,
                              'Time Step (FE) = 16': 6.8805, 'Time Step (TC) = 16': 16.6315,
                              'Time Step (FE) = 32': 11.7829, 'Time Step (TC) = 32': 29.3210,
                              'Time Step (FE) = 96': 45.2907, 'Time Step (TC) = 96': 94.3529})
Mean_impute_time2 = pd.Series({'Time Step (FE) = 1': 0.0005, 'Time Step (TC) = 1': 0.0005,
                               'Time Step (FE) = 4': 0.0005, 'Time Step (TC) = 4': 0.0001,
                               'Time Step (FE) = 16': 0.0008, 'Time Step (TC) = 16': 0.0001,
                               'Time Step (FE) = 32': 0.0010, 'Time Step (TC) = 32': 0.0001,
                               'Time Step (FE) = 96': 0.0045, 'Time Step (TC) = 96': 0.0001})
data2 = pd.DataFrame({'Mean value': Mean_values,
                      'Mean Absolute Error': MAE2,
                      'Root Mean Squared Error': RMSE2,
                      'Mean train time/seconds': Mean_train_time2,
                      'Mean impute time/seconds': Mean_impute_time2})
print('\n')
print("When the noise percentage is 20%, the 'MIDA' experiment results are \n", data2)