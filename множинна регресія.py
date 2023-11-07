from colorama import Fore
import numpy as np
import statsmodels.api as sm

def adv_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print(Fore.LIGHTCYAN_EX, f"\n{results.summary()}")

    r_sq = results.rsquared
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept b0: {results.params[0]}")
    print(f"coefficients: {results.params[1:]}")

    y_pred = results.predict(x)
    print(f"predicted response: \n{y_pred}")

x = [[65.7, 35.5], [71.7, 34.1], [74.4, 30.5],
    [91.1, 36.5], [92.1, 32.1], [93, 40.1],
    [99.3, 34.1], [105.3, 39.3], [110.2, 39.2],
    [115.6, 41.3], [120.2, 40.3], [125, 42.3], [126.9, 41.5],
    [130.8, 43.6], [142.3, 45], [150.2, 44.8]]

y = [6.6, 6.4, 7.4, 6.7, 7.3, 9.3, 9.8, 8.7, 8.4, 9.3, 9.5, 9.5, 9.9, 10.1, 11, 11.8]

adv_regression(x, y)
