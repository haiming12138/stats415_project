import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import cross_validate, StratifiedKFold
from read_data_util import get_full_data, get_group_data


def plot_cv_auc(classifier, X, y, fname):
    """
    Generate CV AUC plot for given classifier & data
    """
    n_splits=5
    cv = StratifiedKFold(n_splits=n_splits)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train, :], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X.iloc[test, :],
            y[test],
            name=f"ROC fold {fold + 1}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability\n(Positive label 1)",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.savefig(f"./figures/{fname}_cv_roc.png", dpi=800)
    plt.close()


def save_cv_metric(classifier, X, y, fname):
    metrics = ['f1', 'balanced_accuracy', 'roc_auc']
    res = cross_validate(classifier, X, y, cv=5, scoring=metrics)

    output = ''
    for metric in metrics:
        perf = np.round(res[f'test_{metric}'].mean(), 3)
        output += f'cv_{metric}: {perf}\n'
    with open(f'./perf_log/{fname}.txt', '+w') as file:
        file.write(output)


def plot_shap_values(classifier, X, y, fname):
    # Create & calculate shap values
    explainer = shap.Explainer(classifier.predict_proba, X)
    shap_values = explainer(X)
    shap_values = shap_values[..., 1]

    # Bee swarm plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f'./figures/{fname}_beeswarm.png', dpi=800, bbox_inches='tight')
    plt.close()

    # Bar plot
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.savefig(f'./figures/{fname}_bar.png', dpi=800, bbox_inches='tight')
    plt.close()

    # Heat map
    shap.plots.heatmap(shap_values, max_display=12, show=False)
    plt.savefig(f'./figures/{fname}_heatmap.png', dpi=800, bbox_inches='tight')
    plt.close()



def run_all(classifier, X, y, fname):
    plot_cv_auc(classifier, X, y, fname)
    plot_shap_values(classifier, X, y, fname)


def run_all_visual(mode: str):
    """
    run all visualizations based on mode
    mode: one of ['svm_full', 'svm_group', 'xgb_full', 'xgb_group']
    """
    if mode.split('_')[1] == 'full':
        X, y = get_full_data()
        model = joblib.load(f'./models/{mode}.sav')
        run_all(model, X, y, mode)
    else:
        model_type = mode.split('_')[0]
        data = get_group_data()
        for group in ['young', 'mid', 'old']:
            fname = f'{model_type}_{group}'
            model = joblib.load(f'./models/{model_type}_{group}.sav')
            X, y = data[group]
            run_all(model, X, y, fname)
