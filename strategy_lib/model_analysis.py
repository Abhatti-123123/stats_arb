def analyze_model(model, X_train, feature_names, label="Model"):
    """
    Analyzes a fitted RandomForestClassifier inside a pipeline:
    - Prints feature importances
    - Plots bar chart
    - (Optional) SHAP explanations
    """
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import matplotlib.pyplot as plt

    if isinstance(model, Pipeline):
        clf = model.named_steps['clf']
    else:
        clf = model

    print(f"\n📊 Feature Importances for: {label}")
    importances = clf.feature_importances_
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print(feature_imp.head(10))
    feature_imp.plot(kind='bar', title=f'{label} - Top Features', figsize=(10, 4))
    plt.tight_layout()
    plt.show()

    # Optional: SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train)
        print(f"📉 SHAP Summary for: {label}")
        shap.summary_plot(shap_values[1], X_train)
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")