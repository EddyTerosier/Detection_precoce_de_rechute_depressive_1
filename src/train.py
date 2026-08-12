"""Entrainement et validation du modele de diagnostic differentiel.

Tache : distinguer, a partir de la seule actigraphie d'une journee, un temoin
sain, un patient en episode depressif et un patient schizophrene.

Pourquoi trois classes
----------------------
Sur la tache binaire patient / temoin, le signal est unidimensionnel : les
patients sont moins actifs, et un simple seuil sur l'activite moyenne
journaliere egale un modele a dix-neuf variables. Un modele multivarie n'y
apporte rien de mesurable.

A trois classes, la situation change par construction : depression et
schizophrenie sont toutes deux hypoactives par rapport aux temoins, donc situees
du meme cote de n'importe quel seuil sur l'activite. Les distinguer exige de
mobiliser d'autres dimensions - fragmentation, duree des plages de repos,
variabilite du rythme. La regle a une seule variable est donc structurellement
incapable de resoudre cette tache, et l'ecart mesure entre elle et le modele
multivarie constitue la demonstration.

Protocole de validation
-----------------------
L'unite d'observation est la journee, mais les journees d'un meme sujet ne sont
pas independantes. Toutes les validations regroupent donc par sujet :
  - leave-one-subject-out : metriques principales, 77 plis, predictions mises en
    commun avant calcul,
  - StratifiedGroupKFold repete : dispersion, 5 plis x 10 repetitions.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import features as feat
import load

N_SPLITS = 5
N_REPEATS = 10

# Etiquettes de classe, alignees sur config.CLASS_NAMES. Nommees ici pour que le
# code de calcul du duel reste lisible sans consulter la configuration.
TEMOIN, DEPRESSION, SCHIZOPHRENIA = 0, 1, 2

FEATURE_SETS = {
    "baseline_1_variable": feat.BASELINE_FEATURE_COLUMNS,
    "reduit_7_variables": feat.REDUCED_FEATURE_COLUMNS,
    "complet_19_variables": feat.FEATURE_COLUMNS,
}


def build_logistic() -> Pipeline:
    """Regression logistique multinomiale, standardisation incluse.

    La standardisation est encapsulee dans le pipeline pour etre ajustee sur les
    seules donnees d'apprentissage de chaque pli : la calculer une fois pour
    toutes ferait fuiter la distribution du test.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000,
                                              random_state=config.RANDOM_SEED)),
        ]
    )


def build_forest() -> Pipeline:
    """Random Forest : capte interactions et effets non monotones entre classes.

    min_samples_leaf est volontairement eleve : avec 77 sujets, des feuilles trop
    fines memoriseraient des individus.
    """
    return Pipeline(
        [
            ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                  random_state=config.RANDOM_SEED,
                                                  n_jobs=-1)),
        ]
    )


MODEL_BUILDERS = {"logistique": build_logistic, "random_forest": build_forest}


def _one_vs_rest_auc(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """AUC one-versus-rest, classe par classe.

    Calculee explicitement plutot que via l'option multiclass de scikit-learn,
    dont les combinaisons d'arguments varient selon les versions.
    """
    per_class = {}
    for label, name in config.CLASS_NAMES.items():
        per_class[name] = roc_auc_score((y_true == label).astype(int), probabilities[:, label])
    per_class["macro"] = float(np.mean(list(per_class.values())))
    return per_class


def _duel_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """AUC du duel depression contre schizophrenie.

    Restreinte aux journees de patients, avec pour score la probabilite de
    schizophrenie conditionnee aux deux classes de patients. C'est la metrique
    centrale : elle mesure exactement ce qu'un seuil sur l'activite moyenne ne
    peut pas faire.
    """
    mask = np.isin(y_true, [DEPRESSION, SCHIZOPHRENIA])
    depression = probabilities[mask, DEPRESSION]
    schizophrenia = probabilities[mask, SCHIZOPHRENIA]
    conditional = schizophrenia / (depression + schizophrenia)
    return roc_auc_score((y_true[mask] == SCHIZOPHRENIA).astype(int), conditional)


def leave_one_subject_out(
    frame: pd.DataFrame, columns: list[str], builder
) -> tuple[dict, pd.DataFrame]:
    """Validation leave-one-subject-out avec mise en commun des predictions.

    Les metriques ne peuvent pas etre calculees pli par pli : un pli ne contient
    qu'un sujet, donc une seule classe. Les probabilites hors echantillon des 77
    plis sont donc rassemblees avant tout calcul.
    """
    X = frame[columns].to_numpy(dtype=float)
    y = frame["class_label"].to_numpy()
    groups = frame["subject_id"].to_numpy()

    probabilities = np.zeros((len(frame), len(config.CLASS_NAMES)))
    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        model = builder().fit(X[train_index], y[train_index])
        probabilities[test_index] = model.predict_proba(X[test_index])

    predicted = probabilities.argmax(axis=1)

    predictions = frame[["subject_id", "date", "day_index", "class_label", "class_name"]].copy()
    for label, name in config.CLASS_NAMES.items():
        predictions[f"p_{name}"] = probabilities[:, label]
    predictions["predicted"] = predicted

    # Niveau sujet : moyenne des probabilites journalieres, puis classe majoritaire.
    # C'est l'unite de decision cliniquement pertinente.
    columns_p = [f"p_{name}" for name in config.CLASS_NAMES.values()]
    per_subject = predictions.groupby("subject_id").agg(
        {**{column: "mean" for column in columns_p}, "class_label": "first"}
    )
    subject_probabilities = per_subject[columns_p].to_numpy()
    subject_truth = per_subject["class_label"].to_numpy()
    subject_predicted = subject_probabilities.argmax(axis=1)

    metrics = {
        "exactitude_jour": accuracy_score(y, predicted),
        "macro_f1_jour": f1_score(y, predicted, average="macro"),
        "auc_ovr": _one_vs_rest_auc(y, probabilities),
        "auc_duel_dep_vs_schizo": _duel_auc(y, probabilities),
        "exactitude_sujet": accuracy_score(subject_truth, subject_predicted),
        "matrice_jour": confusion_matrix(y, predicted).tolist(),
        "matrice_sujet": confusion_matrix(subject_truth, subject_predicted).tolist(),
    }
    return metrics, predictions


def repeated_group_kfold(frame: pd.DataFrame, columns: list[str], builder) -> dict[str, float]:
    """Macro-F1 et AUC du duel sur 5 plis x 10 repetitions, pour la dispersion."""
    X = frame[columns].to_numpy(dtype=float)
    y = frame["class_label"].to_numpy()
    groups = frame["subject_id"].to_numpy()
    macro_f1, duel = [], []

    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED + repeat
        )
        for train_index, test_index in splitter.split(X, y, groups):
            model = builder().fit(X[train_index], y[train_index])
            probabilities = model.predict_proba(X[test_index])
            macro_f1.append(
                f1_score(y[test_index], probabilities.argmax(axis=1), average="macro")
            )
            duel.append(_duel_auc(y[test_index], probabilities))

    macro_f1, duel = np.array(macro_f1), np.array(duel)
    return {
        "macro_f1_moyen": float(macro_f1.mean()),
        "macro_f1_ecart_type": float(macro_f1.std(ddof=1)),
        "duel_moyen": float(duel.mean()),
        "duel_ecart_type": float(duel.std(ddof=1)),
        "duel_p2_5": float(np.percentile(duel, 2.5)),
        "duel_p97_5": float(np.percentile(duel, 97.5)),
    }


def fitted_coefficients(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coefficients par classe du modele reajuste sur l'ensemble des donnees.

    Les variables etant standardisees, les coefficients sont comparables entre
    eux : ils expriment l'effet d'un ecart-type de la variable sur le logit de la
    classe consideree.
    """
    model = build_logistic().fit(
        frame[columns].to_numpy(dtype=float), frame["class_label"].to_numpy()
    )
    coefficients = model.named_steps["classifier"].coef_
    table = pd.DataFrame({"feature": columns})
    for label, name in config.CLASS_NAMES.items():
        table[name] = coefficients[label]
    return table


def _print_matrix(matrix: list[list[int]], title: str) -> None:
    names = [config.CLASS_NAMES[label] for label in sorted(config.CLASS_NAMES)]
    print(f"\n{title}")
    print(f"{'reel \\ predit':<16}" + "".join(f"{name:>16}" for name in names))
    for name, row in zip(names, matrix):
        print(f"{name:<16}" + "".join(f"{value:>16}" for value in row))


def print_results(results: dict[str, dict]) -> None:
    print("=" * 100)
    print("VALIDATION LEAVE-ONE-SUBJECT-OUT (77 plis, predictions mises en commun)")
    print("=" * 100)
    print(f"{'modele':<16}{'variables':<22}{'exact. jour':>12}{'macro F1':>10}"
          f"{'AUC macro':>11}{'DUEL dep/sch':>14}{'exact. sujet':>14}")
    print("-" * 100)
    for block in results.values():
        loso = block["loso"]
        print(
            f"{block['modele']:<16}{block['jeu']:<22}{loso['exactitude_jour']:>12.3f}"
            f"{loso['macro_f1_jour']:>10.3f}{loso['auc_ovr']['macro']:>11.3f}"
            f"{loso['auc_duel_dep_vs_schizo']:>14.3f}{loso['exactitude_sujet']:>14.3f}"
        )

    print()
    print("=" * 100)
    print(f"DISPERSION - StratifiedGroupKFold {N_SPLITS} plis x {N_REPEATS} repetitions")
    print("=" * 100)
    print(f"{'modele':<16}{'variables':<22}{'macro F1':>12}{'ecart-type':>12}"
          f"{'AUC duel':>11}{'intervalle 95%':>22}")
    print("-" * 100)
    for block in results.values():
        repeated = block["repeated"]
        interval = f"[{repeated['duel_p2_5']:.3f} ; {repeated['duel_p97_5']:.3f}]"
        print(
            f"{block['modele']:<16}{block['jeu']:<22}{repeated['macro_f1_moyen']:>12.3f}"
            f"{repeated['macro_f1_ecart_type']:>12.3f}{repeated['duel_moyen']:>11.3f}"
            f"{interval:>22}"
        )
    print("=" * 100)


def main() -> None:
    minutes, subjects, _ = load.load_dataset()
    frame = feat.build_features(minutes, subjects)
    feat.check_integrity(frame)

    results, predictions_by_run = {}, {}
    for model_name, builder in MODEL_BUILDERS.items():
        for set_name, columns in FEATURE_SETS.items():
            # La regle a une seule variable est une reference : lui appliquer une
            # foret n'aurait pas de sens comme point de comparaison.
            if set_name == "baseline_1_variable" and model_name != "logistique":
                continue

            run = f"{model_name} / {set_name}"
            loso_metrics, predictions = leave_one_subject_out(frame, columns, builder)
            results[run] = {
                "modele": model_name,
                "jeu": set_name,
                "n_variables": len(columns),
                "variables": columns,
                "loso": loso_metrics,
                "repeated": repeated_group_kfold(frame, columns, builder),
            }
            predictions_by_run[run] = predictions

    print_results(results)

    baseline = results["logistique / baseline_1_variable"]["loso"]
    best = max(results, key=lambda run: results[run]["loso"]["auc_duel_dep_vs_schizo"])
    best_metrics = results[best]["loso"]

    print(f"\nMeilleure combinaison sur le duel : {best}")
    print(f"  AUC duel depression / schizophrenie : {best_metrics['auc_duel_dep_vs_schizo']:.3f}")
    print(f"  Regle a une seule variable          : "
          f"{baseline['auc_duel_dep_vs_schizo']:.3f}")
    print(f"  Gain du modele multivarie           : "
          f"{best_metrics['auc_duel_dep_vs_schizo'] - baseline['auc_duel_dep_vs_schizo']:+.3f}")

    _print_matrix(best_metrics["matrice_sujet"], f"Matrice de confusion au niveau sujet - {best}")
    _print_matrix(baseline["matrice_sujet"],
                  "Matrice de confusion au niveau sujet - regle a une seule variable")

    coefficients = fitted_coefficients(frame, feat.REDUCED_FEATURE_COLUMNS)
    print("\nCoefficients standardises par classe - logistique sur le jeu reduit :")
    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:+.3f}"))

    tables_dir = config.OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(tables_dir / "coefficients_multiclasse.csv", index=False)
    predictions_by_run[best].to_csv(tables_dir / "predictions_meilleur_modele.csv", index=False)
    with open(config.OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    print(f"\nMetriques ecrites : {config.OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()