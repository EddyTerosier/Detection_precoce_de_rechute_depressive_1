"""Production des artefacts du modele retenu, pour le tableau de bord.

Ce module est la frontiere entre l'experimentation et la restitution. train.py
compare des modeles et des jeux de variables ; export.py fige la combinaison
retenue et ecrit tout ce dont le tableau de bord a besoin, afin que celui-ci
n'ait aucun calcul de modelisation a refaire au demarrage.

Honnetete des scores affiches
-----------------------------
Les probabilites exportees sont celles du protocole leave-one-subject-out :
chaque journee est notee par un modele qui n'a jamais vu le sujet concerne. Un
tableau de bord affichant les scores d'un modele entraine sur le patient
affiche donnerait une impression de performance trompeuse.

Les contributions des variables suivent la meme regle : elles sont calculees
avec les coefficients et la standardisation du pli ou le sujet etait en test.
Pour une regression logistique multinomiale, la contribution de la variable j a
la classe k vaut coefficient[k, j] multiplie par la valeur standardisee de j.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

import config
import features as feat
import load
import train

# Combinaison retenue apres comparaison : regression logistique sur le jeu reduit.
# Elle egale le jeu complet sur le duel depression / schizophrenie tout en
# conservant des coefficients interpretables un par un.
RETAINED_FEATURES = feat.REDUCED_FEATURE_COLUMNS

EXPORT_DIR_NAME = "dashboard"


def _contribution_columns() -> list[str]:
    return [
        f"contrib_{name}_{column}"
        for name in config.CLASS_NAMES.values()
        for column in RETAINED_FEATURES
    ]


def build_dashboard_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Scores et contributions hors echantillon, journee par journee."""
    X = frame[RETAINED_FEATURES].to_numpy(dtype=float)
    y = frame["class_label"].to_numpy()
    groups = frame["subject_id"].to_numpy()

    n_classes = len(config.CLASS_NAMES)
    probabilities = np.zeros((len(frame), n_classes))
    contributions = np.zeros((len(frame), n_classes, len(RETAINED_FEATURES)))

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        model = train.build_logistic().fit(X[train_index], y[train_index])
        probabilities[test_index] = model.predict_proba(X[test_index])

        scaler = model.named_steps["scaler"]
        coefficients = model.named_steps["classifier"].coef_
        standardized = scaler.transform(X[test_index])
        # Produit externe par journee : (journees, classes, variables).
        contributions[test_index] = standardized[:, None, :] * coefficients[None, :, :]

    export = frame[
        ["subject_id", "date", "day_index", "class_label", "class_name"]
    ].copy()
    for label, name in config.CLASS_NAMES.items():
        export[f"p_{name}"] = probabilities[:, label]
    export["predicted_label"] = probabilities.argmax(axis=1)
    export["predicted_name"] = export["predicted_label"].map(config.CLASS_NAMES)

    for label, name in config.CLASS_NAMES.items():
        for index, column in enumerate(RETAINED_FEATURES):
            export[f"contrib_{name}_{column}"] = contributions[:, label, index]

    # Score du duel : probabilite de schizophrenie conditionnee aux deux classes
    # de patients. C'est la quantite sur laquelle porte le seuil de decision.
    duel_denominator = (
        export[f"p_{config.CLASS_NAMES[train.DEPRESSION]}"]
        + export[f"p_{config.CLASS_NAMES[train.SCHIZOPHRENIA]}"]
    )
    export["score_duel"] = (
        export[f"p_{config.CLASS_NAMES[train.SCHIZOPHRENIA]}"] / duel_denominator
    )
    # Score de vigilance : probabilite que la journee ne corresponde pas a un
    # profil de temoin, exprimee sur 100 pour la lecture clinique.
    export["score_vigilance"] = (
        1 - export[f"p_{config.CLASS_NAMES[train.TEMOIN]}"]
    ) * 100
    return export


def build_reference_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
    """Coefficients du modele reajuste sur l'ensemble des donnees.

    Servent de reference d'interpretation globale dans le tableau de bord. Les
    contributions journalieres, elles, restent celles des plis hors echantillon.
    """
    model = train.build_logistic().fit(
        frame[RETAINED_FEATURES].to_numpy(dtype=float), frame["class_label"].to_numpy()
    )
    coefficients = model.named_steps["classifier"].coef_
    table = pd.DataFrame({"feature": RETAINED_FEATURES})
    for label, name in config.CLASS_NAMES.items():
        table[name] = coefficients[label]
    return table


def build_evaluation(export: pd.DataFrame) -> dict:
    """Metriques globales et courbe ROC du duel, pour le panneau d'evaluation."""
    truth = export["class_label"].to_numpy()
    predicted = export["predicted_label"].to_numpy()

    patients = export[export["class_label"].isin([train.DEPRESSION, train.SCHIZOPHRENIA])]
    is_schizophrenia = (patients["class_label"] == train.SCHIZOPHRENIA).astype(int)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        is_schizophrenia, patients["score_duel"]
    )

    per_subject = export.groupby("subject_id").agg(
        class_label=("class_label", "first"),
        **{
            f"p_{name}": (f"p_{name}", "mean") for name in config.CLASS_NAMES.values()
        },
    )
    subject_probabilities = per_subject[
        [f"p_{name}" for name in config.CLASS_NAMES.values()]
    ].to_numpy()
    subject_predicted = subject_probabilities.argmax(axis=1)

    return {
        "features_retenues": RETAINED_FEATURES,
        "n_sujets": int(export["subject_id"].nunique()),
        "n_journees": int(len(export)),
        "classes": [config.CLASS_NAMES[label] for label in sorted(config.CLASS_NAMES)],
        "matrice_jour": confusion_matrix(truth, predicted).tolist(),
        "matrice_sujet": confusion_matrix(
            per_subject["class_label"], subject_predicted
        ).tolist(),
        "roc_duel": {
            "fpr": false_positive_rate.tolist(),
            "tpr": true_positive_rate.tolist(),
            # roc_curve place un premier seuil infini : remplace pour rester
            # sérialisable en JSON.
            "seuils": np.where(np.isinf(thresholds), 1.0, thresholds).tolist(),
        },
    }


def main() -> None:
    minutes, subjects, _ = load.load_dataset()
    frame = feat.build_features(minutes, subjects)
    feat.check_integrity(frame)

    export = build_dashboard_data(frame)
    coefficients = build_reference_coefficients(frame)
    evaluation = build_evaluation(export)

    directory = config.OUTPUT_DIR / EXPORT_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)

    export.to_csv(directory / "scores_journaliers.csv", index=False)
    coefficients.to_csv(directory / "coefficients.csv", index=False)
    subjects.to_csv(directory / "sujets.csv", index=False)
    frame.to_csv(directory / "features.csv", index=False)
    # Les series minute par minute ne sont pas exportees : elles pesent 1,3
    # million de lignes et sont deja disponibles via load.py, que le tableau de
    # bord appelle une fois puis met en cache. Dupliquer ce volume n'apporterait
    # rien et ajouterait une dependance de format.
    with open(directory / "evaluation.json", "w", encoding="utf-8") as stream:
        json.dump(evaluation, stream, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("ARTEFACTS DU TABLEAU DE BORD")
    print("=" * 70)
    print(f"Variables retenues : {', '.join(RETAINED_FEATURES)}")
    print(f"Journees exportees : {len(export)}")
    print(f"Sujets exportes    : {export['subject_id'].nunique()}")
    print()
    print("Exactitude au niveau sujet, hors echantillon :")
    matrix = np.array(evaluation["matrice_sujet"])
    print(f"  {np.trace(matrix)} / {matrix.sum()} sujets classes correctement")
    print()
    print(f"Ecrits dans : {directory}")
    for path in sorted(directory.iterdir()):
        print(f"  - {path.name} ({path.stat().st_size / 1024:.0f} Ko)")
    print("=" * 70)


if __name__ == "__main__":
    main()