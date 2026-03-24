from __future__ import annotations


class ModelSearchAgent:
    """Select ML models based on task type. No LLM call needed."""

    def process(self, task_type: str) -> tuple[str, list[str]]:
        if task_type.lower() == "classification":
            models = [
                "RandomForestClassifier",
                "LogisticRegression",
                "XGBClassifier",
                "GradientBoostingClassifier",
                "SVCClassifier",
                "KNeighborsClassifier",
                "AdaBoostClassifier",
                "ExtraTreesClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
                "RidgeClassifier",
                "MLPClassifier",
                "DecisionTreeClassifier",
                "BayesianRidgeClassifier",
            ]
            msg = (
                "Model Search Agent: For your classification task, I have selected fourteen "
                "complementary algorithms to compare:\n\n"
                "- **RandomForestClassifier** — ensemble, handles non-linearities well\n"
                "- **LogisticRegression** — fast linear baseline\n"
                "- **XGBClassifier** — gradient boosting, excellent overall performance\n"
                "- **GradientBoostingClassifier** — sklearn gradient boosting, strong on structured data\n"
                "- **SVCClassifier** — support vector machine, effective on small-to-medium datasets\n"
                "- **KNeighborsClassifier** — instance-based, captures local patterns\n"
                "- **AdaBoostClassifier** — adaptive boosting, different strategy from GB/XGB\n"
                "- **ExtraTreesClassifier** — extremely randomized trees, fast and diverse\n"
                "- **LGBMClassifier** — LightGBM, fast gradient boosting\n"
                "- **CatBoostClassifier** — handles categorical features natively\n"
                "- **RidgeClassifier** — regularized linear, robust to correlated features\n"
                "- **MLPClassifier** — neural network, captures complex interactions\n"
                "- **DecisionTreeClassifier** — single tree, highly interpretable\n"
                "- **BayesianRidgeClassifier** — Bayesian linear, provides uncertainty\n\n"
                "I will now train and evaluate all fourteen."
            )
        else:
            models = [
                "RandomForestRegressor",
                "LinearRegression",
                "XGBRegressor",
                "GradientBoostingRegressor",
                "SVRRegressor",
                "KNeighborsRegressor",
                "AdaBoostRegressor",
                "ExtraTreesRegressor",
                "LGBMRegressor",
                "CatBoostRegressor",
                "ElasticNetRegressor",
                "RidgeRegressor",
                "LassoRegressor",
                "BayesianRidgeRegressor",
                "MLPRegressor",
                "DecisionTreeRegressor",
            ]
            msg = (
                "Model Search Agent: For your regression task, I have selected sixteen "
                "complementary algorithms:\n\n"
                "- **RandomForestRegressor** — ensemble, robust to outliers\n"
                "- **LinearRegression** — interpretable baseline\n"
                "- **XGBRegressor** — gradient boosting, often the top performer\n"
                "- **GradientBoostingRegressor** — sklearn gradient boosting, strong on structured data\n"
                "- **SVRRegressor** — support vector regression, good on small-to-medium datasets\n"
                "- **KNeighborsRegressor** — instance-based, captures local relationships\n"
                "- **AdaBoostRegressor** — adaptive boosting, complements other ensembles\n"
                "- **ExtraTreesRegressor** — extremely randomized trees, fast and diverse\n"
                "- **LGBMRegressor** — LightGBM, fast gradient boosting\n"
                "- **CatBoostRegressor** — handles categorical features natively\n"
                "- **ElasticNetRegressor** — L1+L2 regularization, great for correlated alloy compositions\n"
                "- **RidgeRegressor** — L2 regularization, robust to multicollinearity\n"
                "- **LassoRegressor** — L1 regularization, automatic feature selection\n"
                "- **BayesianRidgeRegressor** — Bayesian linear, provides uncertainty estimates\n"
                "- **MLPRegressor** — neural network, captures complex interactions\n"
                "- **DecisionTreeRegressor** — single tree, highly interpretable\n\n"
                "I will now train and evaluate all sixteen."
            )
        return msg, models
