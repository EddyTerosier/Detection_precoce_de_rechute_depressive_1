"""Analyses de robustesse du modele de diagnostic differentiel.

Le resultat principal - un ecart de +0.38 d'AUC entre la regle a une seule
variable et le modele multivarie sur le duel depression / schizophrenie - n'a de
valeur que s'il resiste aux confondants du jeu de donnees. Ce module teste
chacun d'eux separement.

Chaque analyse repond a une objection precise :
  hommes_uniquement   : le groupe schizophrene est masculin a 86 %, or les
                        hommes sont en moyenne plus actifs. La separation
                        mesuree pourrait etre du sexe deguise en pathologie.
  toutes_journees     : le choix de tronquer a 12 journees par sujet est
                        arbitraire ; les conclusions doivent tenir sans lui.
  unipolaires_seuls   : la classe depression melange unipolaires et bipolaires.
  split_aleatoire     : demonstration volontaire de la fuite de donnees, pour
                        montrer ce que produirait un protocole incorrect.
  couts_cliniques     : le seuil de decision du duel depend du cout relatif des
                        deux erreurs, qui n'est pas symetrique en clinique.

Le confondant d'hospitalisation est traite a part : les patients schizophrenes de
PSYKOSE sont hospitalises alors que 18 des 23 patients deprimes sont
ambulatoires. Ne pouvant pas l'eliminer, on en mesure l'ampleur en comparant les
deux sous-groupes de patients deprimes entre eux.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

import config
import features as feat
import load
import train

# Sous-groupes de patients deprimes consideres comme unipolaires.
UNIPOLAR_SUBTYPE = "unipolaire"

# Grille de couts relatifs testee pour le duel : combien de fois une erreur sur
# un patient schizophrene coute-t-elle plus qu'une erreur sur un patient deprime.
COST_RATIOS = [0.5, 1.0, 2.0, 4.0]


def _duel_summary(frame: pd.DataFrame, columns: list[str], builder) -> dict[str, float]:
    """Metriques du duel et de la tache complete sur un sous-ensemble donne."""
    metrics, _ = train.leave_one_subject_out(frame, columns, builder)
    return {
        "n_sujets": int(frame["subject_id"].nunique()),
        "n_journees": len(frame),
        "duel": metrics["auc_duel_dep_vs_schizo"],
        "macro_f1": metrics["macro_f1_jour"],
        "exactitude_sujet": metrics["exactitude_sujet"],
    }


def run_subset_analyses(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare regle a une variable et modele reduit sur plusieurs sous-ensembles."""
    subsets = {
        "reference_77_sujets": frame,
        "hommes_uniquement": frame[frame["sex"] == "M"],
        "femmes_uniquement": frame[frame["sex"] == "F"],
        # La classe depression est restreinte aux unipolaires ; temoins et
        # schizophrenes sont conserves entiers.
        "depression_unipolaire": frame[
            (frame["class_label"] != train.DEPRESSION)
            | (frame["subtype"] == UNIPOLAR_SUBTYPE)
        ],
    }

    rows = []
    for name, subset in subsets.items():
        present = sorted(subset["class_label"].unique())
        if len(present) < 3:
            rows.append({"analyse": name, "note": "moins de trois classes, ignoree"})
            continue

        baseline = _duel_summary(subset, feat.BASELINE_FEATURE_COLUMNS, train.build_logistic)
        model = _duel_summary(subset, feat.REDUCED_FEATURE_COLUMNS, train.build_logistic)
        rows.append(
            {
                "analyse": name,
                "n_sujets": model["n_sujets"],
                "n_journees": model["n_journees"],
                "duel_baseline": baseline["duel"],
                "duel_modele": model["duel"],
                "gain": model["duel"] - baseline["duel"],
                "macro_f1_modele": model["macro_f1"],
                "exactitude_sujet_modele": model["exactitude_sujet"],
            }
        )
    return pd.DataFrame(rows)


def run_all_days_analysis() -> dict[str, float]:
    """Recharge les donnees sans troncature a 12 journees par sujet.

    Le parametre de configuration est modifie temporairement : c'est le seul
    moyen de rejouer le chargement complet sans dupliquer sa logique.
    """
    original = config.KEEP_ALL_COMPLETE_DAYS
    config.KEEP_ALL_COMPLETE_DAYS = True
    try:
        minutes, subjects, _ = load.load_dataset()
        frame = feat.build_features(minutes, subjects)
        baseline = _duel_summary(frame, feat.BASELINE_FEATURE_COLUMNS, train.build_logistic)
        model = _duel_summary(frame, feat.REDUCED_FEATURE_COLUMNS, train.build_logistic)
    finally:
        config.KEEP_ALL_COMPLETE_DAYS = original

    return {
        "n_sujets": model["n_sujets"],
        "n_journees": model["n_journees"],
        "duel_baseline": baseline["duel"],
        "duel_modele": model["duel"],
        "gain": model["duel"] - baseline["duel"],
        "macro_f1_modele": model["macro_f1"],
    }


def demonstrate_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare un decoupage par sujet a un decoupage aleatoire des journees.

    Le decoupage aleatoire place des journees d'un meme sujet a la fois dans
    l'apprentissage et dans le test. Le modele peut alors reconnaitre l'individu
    au lieu d'apprendre la pathologie, ce qui gonfle artificiellement les
    performances. Cette analyse chiffre l'ampleur du biais.
    """
    X = frame[feat.REDUCED_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame["class_label"].to_numpy()
    groups = frame["subject_id"].to_numpy()

    rows = []

    # Protocole incorrect : les journees sont melangees sans tenir compte du sujet.
    incorrect = []
    for seed in range(10):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, test_index in splitter.split(X, y):
            model = train.build_logistic().fit(X[train_index], y[train_index])
            probabilities = model.predict_proba(X[test_index])
            incorrect.append(train._duel_auc(y[test_index], probabilities))

    # Protocole correct : aucun sujet ne figure a la fois dans les deux parties.
    correct = []
    for seed in range(10):
        splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        for train_index, test_index in splitter.split(X, y, groups):
            if len(np.unique(y[test_index])) < 3:
                continue
            model = train.build_logistic().fit(X[train_index], y[train_index])
            probabilities = model.predict_proba(X[test_index])
            correct.append(train._duel_auc(y[test_index], probabilities))

    for name, scores in (("split_aleatoire_incorrect", incorrect),
                         ("split_par_sujet_correct", correct)):
        scores = np.array(scores)
        rows.append(
            {
                "protocole": name,
                "n_estimations": len(scores),
                "duel_moyen": scores.mean(),
                "duel_ecart_type": scores.std(ddof=1),
                "duel_p2_5": np.percentile(scores, 2.5),
                "duel_p97_5": np.percentile(scores, 97.5),
            }
        )
    return pd.DataFrame(rows)


def measure_hospitalisation_confound(frame: pd.DataFrame) -> pd.DataFrame:
    """Mesure l'effet de l'hospitalisation a l'interieur de la classe depression.

    Les patients schizophrenes sont tous hospitalises, les deprimes
    majoritairement ambulatoires : le confondant ne peut donc pas etre elimine.
    Il peut en revanche etre borne. Si les deprimes hospitalises et ambulatoires
    ne se distinguent pas sur les features, l'environnement de vie pese peu
    devant la pathologie. Les effectifs sont faibles (5 contre 18 sujets) : ces
    ecarts sont indicatifs, non concluants.
    """
    depressed = frame[frame["class_label"] == train.DEPRESSION]
    inpatient = depressed[depressed["inpatient"] == "hospitalise"]
    outpatient = depressed[depressed["inpatient"] == "ambulatoire"]
    schizophrenia = frame[frame["class_label"] == train.SCHIZOPHRENIA]

    rows = []
    for column in feat.REDUCED_FEATURE_COLUMNS:
        rows.append(
            {
                "feature": column,
                "dep_hospitalise": inpatient[column].mean(),
                "dep_ambulatoire": outpatient[column].mean(),
                "schizophrenie": schizophrenia[column].mean(),
                # Effet de l'hospitalisation, a l'interieur d'une meme pathologie.
                "d_hosp_vs_ambu": feat._cohen_d(inpatient[column], outpatient[column]),
                # Effet de la pathologie, sur les seuls patients ambulatoires.
                "d_schizo_vs_ambu": feat._cohen_d(schizophrenia[column], outpatient[column]),
            }
        )
    return pd.DataFrame(rows)


def cost_sensitive_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    """Seuil du duel selon le cout relatif des deux erreurs.

    Les deux erreurs n'ont pas le meme cout clinique : orienter a tort vers un
    suivi de schizophrenie ou manquer une schizophrenie n'ont pas les memes
    consequences. Le seuil optimal est celui qui minimise le cout total attendu,
    et non celui qui maximise l'exactitude.
    """
    metrics, predictions = train.leave_one_subject_out(
        frame, feat.REDUCED_FEATURE_COLUMNS, train.build_logistic
    )

    patients = predictions[
        predictions["class_label"].isin([train.DEPRESSION, train.SCHIZOPHRENIA])
    ].copy()
    depression_column = f"p_{config.CLASS_NAMES[train.DEPRESSION]}"
    schizophrenia_column = f"p_{config.CLASS_NAMES[train.SCHIZOPHRENIA]}"
    patients["score"] = patients[schizophrenia_column] / (
        patients[depression_column] + patients[schizophrenia_column]
    )
    is_schizophrenia = (patients["class_label"] == train.SCHIZOPHRENIA).to_numpy()
    scores = patients["score"].to_numpy()

    rows = []
    for ratio in COST_RATIOS:
        candidates = np.unique(scores)
        best = None
        for threshold in candidates:
            predicted = scores >= threshold
            missed_schizophrenia = int((is_schizophrenia & ~predicted).sum())
            false_schizophrenia = int((~is_schizophrenia & predicted).sum())
            cost = ratio * missed_schizophrenia + false_schizophrenia
            if best is None or cost < best["cout_total"]:
                best = {
                    "cout_relatif": ratio,
                    "seuil": float(threshold),
                    "schizo_manquees": missed_schizophrenia,
                    "schizo_a_tort": false_schizophrenia,
                    "cout_total": cost,
                    "sensibilite_schizo": float(
                        (is_schizophrenia & predicted).sum() / is_schizophrenia.sum()
                    ),
                }
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    minutes, subjects, _ = load.load_dataset()
    frame = feat.build_features(minutes, subjects)
    feat.check_integrity(frame)

    print("=" * 100)
    print("ROBUSTESSE 1 - SOUS-ENSEMBLES DE POPULATION (leave-one-subject-out)")
    print("=" * 100)
    subsets = run_subset_analyses(frame)
    print(subsets.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print()
    print("=" * 100)
    print("ROBUSTESSE 2 - SANS TRONCATURE A 12 JOURNEES PAR SUJET")
    print("=" * 100)
    all_days = run_all_days_analysis()
    for key, value in all_days.items():
        formatted = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {key:<26}: {formatted}")

    print()
    print("=" * 100)
    print("ROBUSTESSE 3 - DEMONSTRATION DE LA FUITE DE DONNEES")
    print("=" * 100)
    leakage = demonstrate_leakage(frame)
    print(leakage.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    gap = (
        leakage.loc[0, "duel_moyen"] - leakage.loc[1, "duel_moyen"]
    )
    print(f"\n  Surestimation due au decoupage aleatoire : {gap:+.3f} d'AUC")

    print()
    print("=" * 100)
    print("ROBUSTESSE 4 - AMPLEUR DU CONFONDANT D'HOSPITALISATION")
    print("=" * 100)
    confound = measure_hospitalisation_confound(frame)
    print(confound.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\n  d_hosp_vs_ambu   : effet de l'hospitalisation, pathologie constante")
    print("  d_schizo_vs_ambu : effet de la pathologie, sur patients ambulatoires")

    print()
    print("=" * 100)
    print("ROBUSTESSE 5 - SEUIL DU DUEL SELON LE COUT CLINIQUE RELATIF")
    print("=" * 100)
    costs = cost_sensitive_threshold(frame)
    print(costs.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n  cout_relatif : cout d'une schizophrenie manquee, rapporte a celui")
    print("  d'une orientation a tort vers la schizophrenie.")
    print("=" * 100)

    tables_dir = config.OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    subsets.to_csv(tables_dir / "robustesse_sous_ensembles.csv", index=False)
    leakage.to_csv(tables_dir / "robustesse_fuite_donnees.csv", index=False)
    confound.to_csv(tables_dir / "robustesse_hospitalisation.csv", index=False)
    costs.to_csv(tables_dir / "robustesse_couts_cliniques.csv", index=False)
    with open(tables_dir / "robustesse_toutes_journees.json", "w", encoding="utf-8") as stream:
        json.dump(all_days, stream, indent=2, ensure_ascii=False)

    print(f"\nTableaux ecrits : {tables_dir}")


if __name__ == "__main__":
    main()