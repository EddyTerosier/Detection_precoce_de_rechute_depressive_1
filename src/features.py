"""Transformation des series minute par minute en features journalieres.

Chaque journee de 1440 minutes devient une ligne de 19 variables, toutes
interpretables cliniquement. Les indices circadiens non parametriques (M10, L5,
amplitude relative, variabilite intra-journaliere) suivent les definitions
usuelles de la litterature actigraphique et sont calcules sur les 24 moyennes
horaires de la journee.

La stabilite inter-journaliere (IS) est volontairement absente : elle compare
une journee au profil moyen du sujet et n'a donc pas de sens a l'echelle d'une
journee isolee. Elle est calculee au niveau du sujet dans describe.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import load

# --- Definitions temporelles ------------------------------------------------
NIGHT_HOURS = range(0, 6)  # 00h00 - 05h59
DAY_HOURS = range(8, 20)  # 08h00 - 19h59
HOURS_PER_DAY = 24
M10_WINDOW = 10  # heures les plus actives
L5_WINDOW = 5  # heures les moins actives

# Colonnes servant reellement de variables explicatives au modele.
# Toute colonne absente de cette liste est une metadonnee et ne doit jamais
# entrer dans l'apprentissage.
FEATURE_COLUMNS = [
    # Niveau d'activite
    "activity_mean",
    "activity_median",
    "activity_p95",
    "activity_max",
    # Variabilite
    "activity_std",
    "activity_cv",
    "activity_iqr",
    # Sedentarite et fragmentation
    "pct_zero_minutes",
    "longest_zero_run",
    "n_active_bouts",
    # Rythme jour / nuit
    "night_activity_mean",
    "day_activity_mean",
    "night_day_ratio",
    "activity_center_hour",
    # Indices circadiens non parametriques
    "m10",
    "l5",
    "relative_amplitude",
    "intradaily_variability",
    "l5_onset_hour",
]

# Metadonnees accompagnant chaque journee. Elles proviennent du referentiel
# sujet harmonise produit par load.py et ne doivent jamais entrer dans
# l'apprentissage.
METADATA_COLUMNS = [
    "subject_id",
    "date",
    "day_index",
    "class_label",
    "class_name",
    "source",
    "sex",
    "age_range",
    "subtype",
    "severity_scale",
    "severity_score",
    "inpatient",
]

# Jeu reduit : un seul representant par famille clinique, retenu apres l'analyse
# de correlation. Douze paires de features du jeu complet depassent |r| = 0.90,
# toutes dans la famille "niveau d'activite", ce qui rend les coefficients
# individuels d'une regression instables et donc ininterpretables. Chaque
# variable ci-dessous porte une dimension clinique distincte.
REDUCED_FEATURE_COLUMNS = [
    "activity_mean",  # niveau global d'activite
    "activity_cv",  # variabilite relative
    "n_active_bouts",  # fragmentation du comportement
    "pct_zero_minutes",  # sedentarite
    "night_day_ratio",  # equilibre jour / nuit
    "l5_onset_hour",  # phase de la periode de repos
    "relative_amplitude",  # contraste activite / repos
]

# Regle de reference a une seule variable : un seuil sur l'activite moyenne
# journaliere. Sert de point de comparaison obligatoire aux modeles multivaries.
BASELINE_FEATURE_COLUMNS = ["activity_mean"]


def _longest_zero_run(values: np.ndarray) -> int:
    """Longueur de la plus longue plage continue d'activite nulle, en minutes."""
    longest = current = 0
    for is_zero in values == 0:
        current = current + 1 if is_zero else 0
        longest = max(longest, current)
    return longest


def _count_active_bouts(values: np.ndarray) -> int:
    """Nombre de plages d'activite distinctes, separees par au moins une minute nulle.

    Mesure la fragmentation du comportement : beaucoup de micro-episodes
    d'activite, ou quelques longues sequences continues.
    """
    active = values > 0
    starts = active & ~np.concatenate(([False], active[:-1]))
    return int(starts.sum())


def _circular_windows(hourly: np.ndarray, width: int) -> np.ndarray:
    """Moyennes de toutes les fenetres circulaires de `width` heures consecutives.

    La fenetre est circulaire car une periode de repos chevauche minuit : sans
    cela, un sujet endormi de 23h a 4h verrait sa periode de repos coupee.
    """
    doubled = np.concatenate([hourly, hourly])
    return np.array([doubled[start : start + width].mean() for start in range(HOURS_PER_DAY)])


def _intradaily_variability(hourly: np.ndarray) -> float:
    """Variabilite intra-journaliere : fragmentation du rythme sur 24 h.

    Rapport entre la variance des differences horaires successives et la
    variance globale. Une valeur elevee signale un rythme hache.
    """
    n = len(hourly)
    successive = np.sum(np.diff(hourly) ** 2)
    total = np.sum((hourly - hourly.mean()) ** 2)
    return float(n * successive / ((n - 1) * total))


def _day_features(day: pd.DataFrame) -> dict[str, float]:
    """Calcule les 19 features d'une journee de 1440 minutes."""
    activity = day["activity"].to_numpy(dtype=float)
    minute_of_day = day["timestamp"].dt.hour.to_numpy() * 60 + day["timestamp"].dt.minute.to_numpy()
    hour_of_day = day["timestamp"].dt.hour.to_numpy()

    hourly = np.array([activity[hour_of_day == hour].mean() for hour in range(HOURS_PER_DAY)])
    window_means_10 = _circular_windows(hourly, M10_WINDOW)
    window_means_5 = _circular_windows(hourly, L5_WINDOW)
    l5_onset = int(np.argmin(window_means_5))

    mean = activity.mean()
    night = activity[np.isin(hour_of_day, list(NIGHT_HOURS))].mean()
    daytime = activity[np.isin(hour_of_day, list(DAY_HOURS))].mean()
    m10 = float(window_means_10.max())
    l5 = float(window_means_5.min())

    return {
        "activity_mean": mean,
        "activity_median": float(np.median(activity)),
        "activity_p95": float(np.percentile(activity, 95)),
        "activity_max": float(activity.max()),
        "activity_std": float(activity.std(ddof=1)),
        "activity_cv": float(activity.std(ddof=1) / mean),
        "activity_iqr": float(np.percentile(activity, 75) - np.percentile(activity, 25)),
        "pct_zero_minutes": float((activity == 0).mean() * 100),
        "longest_zero_run": float(_longest_zero_run(activity)),
        "n_active_bouts": float(_count_active_bouts(activity)),
        "night_activity_mean": float(night),
        "day_activity_mean": float(daytime),
        "night_day_ratio": float(night / daytime),
        # Barycentre temporel de l'activite : proxy simple d'avance ou de retard
        # de phase sur la journee.
        "activity_center_hour": float((minute_of_day * activity).sum() / activity.sum() / 60),
        "m10": m10,
        "l5": l5,
        "relative_amplitude": float((m10 - l5) / (m10 + l5)),
        "intradaily_variability": _intradaily_variability(hourly),
        "l5_onset_hour": float(l5_onset),
    }


def build_features(minutes: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Construit la table journaliere a partir des series minute par minute."""
    rows = []
    for (subject_id, date), day in minutes.groupby(["subject_id", "date"], sort=True):
        if len(day) != config.MINUTES_PER_DAY:
            raise ValueError(f"{subject_id} / {date}: {len(day)} minutes")
        rows.append({"subject_id": subject_id, "date": date, **_day_features(day)})

    features = pd.DataFrame(rows)

    # Index chronologique de la journee au sein du suivi du sujet : sert d'axe
    # temporel au tableau de bord.
    features["day_index"] = features.groupby("subject_id")["date"].rank(method="first").astype(int)

    metadata_available = [column for column in METADATA_COLUMNS if column in subjects.columns]
    features = features.merge(subjects[metadata_available], on="subject_id", how="left")
    return features[METADATA_COLUMNS + FEATURE_COLUMNS].sort_values(
        ["subject_id", "day_index"], ignore_index=True
    )


def check_integrity(features: pd.DataFrame) -> None:
    """Refuse une table contenant une valeur non finie ou une volumetrie inattendue.

    Un controle unique en sortie remplace des garde-fous disperses dans chaque
    calcul : si une division degenere un jour, elle est detectee ici plutot que
    masquee par une valeur de repli arbitraire.
    """
    numeric = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        faulty = [
            column
            for column in FEATURE_COLUMNS
            if not np.isfinite(features[column].to_numpy(dtype=float)).all()
        ]
        raise ValueError(f"Valeurs non finies dans : {faulty}")

    if features["subject_id"].nunique() != config.EXPECTED_SUBJECTS:
        raise ValueError(
            f"{features['subject_id'].nunique()} sujets au lieu de "
            f"{config.EXPECTED_SUBJECTS}"
        )

    if not config.KEEP_ALL_COMPLETE_DAYS:
        counts = features.groupby("subject_id").size().unique()
        if list(counts) != [config.DAYS_PER_SUBJECT]:
            raise ValueError(f"Nombre de journees non homogene : {counts}")


def print_summary(features: pd.DataFrame) -> None:
    """Moyenne de chaque feature par classe, avec taille d'effet des duels.

    Deux duels sont affiches : chaque groupe de patients contre les temoins, et
    surtout depression contre schizophrenie, qui est le coeur de la tache de
    diagnostic differentiel.
    """
    print("=" * 96)
    print("FEATURES JOURNALIERES")
    print("=" * 96)
    print(f"Lignes : {len(features)}   Features : {len(FEATURE_COLUMNS)}   "
          f"Sujets : {features['subject_id'].nunique()}")
    for label, name in config.CLASS_NAMES.items():
        block = features[features["class_label"] == label]
        print(f"  classe {label} ({name}) : {len(block)} journees")
    print()

    header = f"{'feature':<26}{'temoin':>10}{'depress.':>10}{'schizo.':>10}"
    print(header + f"{'d dep/tem':>11}{'d sch/tem':>11}{'d sch/dep':>11}")
    print("-" * 96)
    for column in FEATURE_COLUMNS:
        by_class = {
            label: features.loc[features["class_label"] == label, column]
            for label in config.CLASS_NAMES
        }
        print(
            f"{column:<26}{by_class[0].mean():>10.2f}{by_class[1].mean():>10.2f}"
            f"{by_class[2].mean():>10.2f}"
            f"{_cohen_d(by_class[1], by_class[0]):>11.2f}"
            f"{_cohen_d(by_class[2], by_class[0]):>11.2f}"
            f"{_cohen_d(by_class[2], by_class[1]):>11.2f}"
        )
    print("-" * 96)
    print("d = ecart standardise. La derniere colonne est la plus importante :")
    print("elle mesure ce qui separe schizophrenie et depression.")
    print("=" * 96)


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    """Ecart standardise entre deux echantillons (variance intra-groupe commune)."""
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    minutes, subjects, report = load.load_dataset()
    features = build_features(minutes, subjects)
    check_integrity(features)
    print_summary(features)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = config.OUTPUT_DIR / "features.csv"
    features.to_csv(destination, index=False)
    print(f"\nTable ecrite : {destination}")


if __name__ == "__main__":
    main()