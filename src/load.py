"""Chargement et validation des donnees d'actigraphie DEPRESJON et PSYKOSE.

Ce module est responsable de quatre choses, et de rien d'autre :
  1. lire les fichiers bruts des deux sources en verifiant leur conformite,
  2. ne conserver que les journees exploitables (24 h completes a la minute),
  3. harmoniser des metadonnees cliniques de formats differents,
  4. produire une etiquette a trois classes.

Les 32 temoins sont communs aux deux jeux : les fichiers PSYKOSE/control sont
byte-identiques a ceux de DEPRESJON/control. Ils ne sont donc charges qu'une
seule fois, faute de quoi ils compteraient double.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config

EXPECTED_COLUMNS = ["timestamp", "date", "activity"]

# Valeurs traitees comme manquantes : les deux fichiers de metadonnees utilisent
# a la fois "NA", la chaine vide et un espace seul.
NA_VALUES = ["NA", "", " "]

# Colonnes du referentiel sujet harmonise. Elles constituent le contrat de
# sortie du module : tout consommateur peut s'y fier quelle que soit la source.
SUBJECT_COLUMNS = [
    "subject_id",
    "class_label",
    "class_name",
    "source",
    "sex",
    "age_range",
    "days_declared",
    "subtype",
    "severity_scale",
    "severity_score",
    "inpatient",
]


class SchemaError(Exception):
    """Le fichier brut ne respecte pas le schema attendu."""


@dataclass
class LoadReport:
    """Trace de ce qui a ete conserve ou ecarte, pour audit et pour le memoire."""

    subjects: int = 0
    days_available: int = 0
    days_kept: int = 0
    minutes_kept: int = 0
    dropped_edge_days: int = 0
    invalid_days: int = 0
    anomalies: list[str] = field(default_factory=list)


def _read_subject_file(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Lit un fichier sujet, valide son schema et signale les anomalies.

    Les horodatages dupliques ne sont pas une erreur fatale : le changement
    d'heure d'automne enregistre deux fois l'heure 02h00-02h59, ce qui porte la
    journee a 1500 minutes. Elle sera ecartee par la regle de completude. Le
    controle strict d'unicite est reapplique plus loin sur les journees
    effectivement retenues.
    """
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    anomalies: list[str] = []

    if list(frame.columns) != EXPECTED_COLUMNS:
        raise SchemaError(f"{path.name}: colonnes {list(frame.columns)}")

    if frame[["timestamp", "activity"]].isna().any().any():
        raise SchemaError(f"{path.name}: valeurs manquantes")

    if not pd.api.types.is_numeric_dtype(frame["activity"]):
        raise SchemaError(f"{path.name}: colonne activity non numerique")

    if (frame["activity"] < 0).any():
        raise SchemaError(f"{path.name}: activite negative")

    # La colonne date est redondante avec timestamp : on verifie la coherence
    # avant de s'appuyer dessus pour le decoupage journalier.
    if not (frame["timestamp"].dt.strftime("%Y-%m-%d") == frame["date"]).all():
        raise SchemaError(f"{path.name}: colonne date incoherente avec timestamp")

    duplicated = frame.loc[frame["timestamp"].duplicated(), "date"].unique()
    for date in sorted(duplicated):
        anomalies.append(f"{path.stem} / {date} : horodatages dupliques (changement d'heure)")

    frame = frame.sort_values("timestamp", ignore_index=True)
    frame["subject_id"] = path.stem
    return frame, anomalies


def _filter_valid_days(frame: pd.DataFrame, dates) -> tuple[pd.Index, dict[str, str]]:
    """Ecarte les journees dont le signal ne peut pas provenir d'un port normal.

    Trois artefacts sont distingues, chacun observe dans les donnees reelles :
    dispositif non porte, valeur de capteur figee, et journee comportant trop peu
    de minutes actives pour caracteriser un rythme. Les motifs sont retournes
    pour etre journalises plutot que silencieusement absorbes.
    """
    block = frame[frame["date"].isin(dates)]
    statistics = block.groupby("date")["activity"].agg(
        mean="mean",
        std="std",
        active_fraction=lambda values: (values > 0).mean(),
    )

    reasons: dict[str, str] = {}
    for date, row in statistics.iterrows():
        if row["mean"] < config.MIN_DAILY_ACTIVITY_MEAN:
            reasons[date] = f"non-port (moyenne {row['mean']:.2f} / minute)"
        elif row["std"] < config.MIN_DAILY_ACTIVITY_STD:
            reasons[date] = f"signal constant (ecart-type {row['std']:.2f})"
        elif row["active_fraction"] < config.MIN_ACTIVE_MINUTES_FRACTION:
            reasons[date] = f"trop peu de minutes actives ({row['active_fraction']:.1%})"

    valid = statistics.index.difference(pd.Index(list(reasons)))
    return valid, reasons


def _select_days(frame: pd.DataFrame, report: LoadReport) -> pd.DataFrame:
    """Ne garde que les journees completes, puis tronque a un nombre fixe.

    Une journee incomplete en debut ou en fin d'enregistrement est un effet de
    bord de la pose et du retrait de la montre. Une journee dont le compte de
    minutes s'ecarte de 1440 au milieu de l'enregistrement signale une anomalie
    et est journalisee.
    """
    minutes_per_date = frame.groupby("date").size()
    all_dates = list(minutes_per_date.index)
    complete = minutes_per_date[minutes_per_date == config.MINUTES_PER_DAY].index
    subject_id = frame["subject_id"].iloc[0]

    for date, count in minutes_per_date[minutes_per_date != config.MINUTES_PER_DAY].items():
        if date in (all_dates[0], all_dates[-1]):
            report.dropped_edge_days += 1
        else:
            report.anomalies.append(f"{subject_id} / {date} : {count} minutes")

    # Exclusion des journees invalides. Elles sont ecartees avant la troncature,
    # sans quoi elles occuperaient la place de journees exploitables.
    valid, reasons = _filter_valid_days(frame, complete)
    for date, reason in reasons.items():
        report.invalid_days += 1
        report.anomalies.append(f"{subject_id} / {date} : {reason}")
    complete = valid

    report.days_available += len(complete)

    kept_dates = sorted(complete)
    if not config.KEEP_ALL_COMPLETE_DAYS:
        kept_dates = kept_dates[: config.DAYS_PER_SUBJECT]
        if len(kept_dates) < config.DAYS_PER_SUBJECT:
            raise SchemaError(
                f"{subject_id}: {len(kept_dates)} journees completes et portees, "
                f"{config.DAYS_PER_SUBJECT} attendues"
            )

    selected = frame[frame["date"].isin(kept_dates)].reset_index(drop=True)

    # Controle strict reapplique sur les seules donnees conservees.
    if selected["timestamp"].duplicated().any():
        raise SchemaError(f"{subject_id}: doublons dans les journees retenues")

    return selected


def _depresjon_subjects() -> pd.DataFrame:
    """Metadonnees harmonisees des temoins et des patients deprimes."""
    scores = pd.read_csv(config.RAW_DEPRESJON_DIR / "scores.csv", na_values=NA_VALUES)
    scores = scores.rename(columns={"number": "subject_id"})

    is_patient = scores["subject_id"].str.startswith("condition")
    return pd.DataFrame(
        {
            "subject_id": scores["subject_id"],
            "class_label": is_patient.map({True: 1, False: 0}),
            "source": "depresjon",
            "sex": scores["gender"].map(config.DEPRESJON_SEX),
            "age_range": scores["age"],
            "days_declared": scores["days"],
            "subtype": scores["afftype"].map(config.AFFTYPE_LABELS),
            "severity_scale": is_patient.map({True: "MADRS", False: None}),
            "severity_score": scores["madrs1"],
            # Les temoins ne sont pas des patients : le statut n'a pas d'objet
            # pour eux, ce qui n'est pas la meme chose qu'une donnee absente.
            "inpatient": scores["inpatient"]
            .map(config.INPATIENT_LABELS)
            .where(is_patient, "sans objet"),
        }
    )


def _psykose_subjects() -> pd.DataFrame:
    """Metadonnees harmonisees des patients schizophrenes.

    Le statut d'hospitalisation n'est pas documente dans PSYKOSE : il reste donc
    inconnu, ce qui constitue une limite explicite du cas concret.
    """
    info = pd.read_csv(config.RAW_PSYKOSE_DIR / "patients_info.csv", na_values=NA_VALUES)
    info = info.rename(columns={"number": "subject_id"})

    return pd.DataFrame(
        {
            "subject_id": info["subject_id"],
            "class_label": 2,
            "source": "psykose",
            "sex": info["gender"].map(config.PSYKOSE_SEX),
            "age_range": info["age"],
            "days_declared": info["days"],
            "subtype": info["schtype"],
            "severity_scale": "BPRS",
            "severity_score": info["bprs"],
            # PSYKOSE ne documente pas ce statut : l'information est reellement
            # absente, et cette absence est une limite du cas concret.
            "inpatient": "non documente",
        }
    )


def load_subjects() -> pd.DataFrame:
    """Referentiel sujet unifie des deux sources."""
    subjects = pd.concat([_depresjon_subjects(), _psykose_subjects()], ignore_index=True)
    subjects["class_name"] = subjects["class_label"].map(config.CLASS_NAMES)

    if subjects["subject_id"].duplicated().any():
        raise SchemaError("identifiants de sujets en collision entre les deux sources")
    if subjects["sex"].isna().any():
        raise SchemaError("sexe non renseigne pour au moins un sujet")

    return subjects[SUBJECT_COLUMNS]


def load_minutes() -> tuple[pd.DataFrame, LoadReport]:
    """Charge les series minute par minute des journees retenues.

    Les temoins ne sont lus que dans DEPRESJON : les fichiers homonymes de
    PSYKOSE sont identiques et les charger a nouveau les compterait double.
    """
    directories = [
        config.RAW_DEPRESJON_DIR / "control",
        config.RAW_DEPRESJON_DIR / "condition",
        config.RAW_PSYKOSE_DIR / "patient",
    ]
    report = LoadReport()
    frames = []

    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dossier introuvable : {directory}")

        for path in sorted(directory.glob("*.csv")):
            frame, anomalies = _read_subject_file(path)
            report.anomalies.extend(anomalies)
            frames.append(_select_days(frame, report))
            report.subjects += 1

    minutes = pd.concat(frames, ignore_index=True)
    report.days_kept = len(minutes.groupby(["subject_id", "date"]))
    report.minutes_kept = len(minutes)
    return minutes, report


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, LoadReport]:
    """Point d'entree du module.

    Renvoie les minutes retenues, le referentiel sujet et le rapport de
    chargement.
    """
    minutes, report = load_minutes()
    subjects = load_subjects()

    known = set(subjects["subject_id"])
    loaded = set(minutes["subject_id"])
    if loaded - known:
        raise SchemaError(f"Sujets sans metadonnees : {sorted(loaded - known)}")
    if known - loaded:
        raise SchemaError(f"Sujets sans actigraphie : {sorted(known - loaded)}")

    days_kept = (
        minutes.groupby("subject_id")["date"].nunique().rename("days_kept").reset_index()
    )
    subjects = subjects.merge(days_kept, on="subject_id", how="inner")

    counts = subjects["class_label"].value_counts().to_dict()
    if counts != config.EXPECTED_SUBJECTS_PER_CLASS:
        raise SchemaError(f"Effectifs par classe inattendus : {counts}")

    return minutes, subjects, report


def print_report(subjects: pd.DataFrame, report: LoadReport) -> None:
    """Affiche le rapport de chargement, a reprendre dans le memoire si besoin."""
    mode = (
        "toutes les journees completes"
        if config.KEEP_ALL_COMPLETE_DAYS
        else f"{config.DAYS_PER_SUBJECT} premieres journees completes par sujet"
    )

    print("=" * 74)
    print("RAPPORT DE CHARGEMENT - DEPRESJON + PSYKOSE")
    print("=" * 74)
    print(f"DEPRESJON          : {config.RAW_DEPRESJON_DIR}")
    print(f"PSYKOSE            : {config.RAW_PSYKOSE_DIR}")
    print(f"Regle de selection : {mode}")
    print()
    print(f"Sujets charges                 : {report.subjects}")
    print(f"Journees completes disponibles : {report.days_available}")
    print(f"Journees retenues              : {report.days_kept}")
    print(f"Minutes retenues               : {report.minutes_kept:,}")
    print(f"Journees de bord ecartees      : {report.dropped_edge_days}")
    print(f"Journees invalides ecartees    : {report.invalid_days}")
    print()

    per_class = subjects.groupby(["class_label", "class_name"]).agg(
        sujets=("subject_id", "size"), journees=("days_kept", "sum")
    )
    print("Repartition par classe :")
    print(per_class.to_string())

    print("\nRepartition du sexe par classe (confondant a documenter) :")
    print(pd.crosstab(subjects["class_name"], subjects["sex"]).to_string())

    print("\nStatut d'hospitalisation :")
    print(subjects["inpatient"].value_counts(dropna=False).to_string())

    print("\nAnomalies detectees :")
    if report.anomalies:
        shown = sorted(set(report.anomalies))
        for anomaly in shown[:12]:
            print(f"  - {anomaly}")
        if len(shown) > 12:
            print(f"  ... et {len(shown) - 12} autres (voir anomalies.csv)")
    else:
        print("  aucune")
    print("=" * 74)


def main() -> None:
    minutes, subjects, report = load_dataset()
    print_report(subjects, report)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = config.OUTPUT_DIR / "subjects.csv"
    subjects.to_csv(destination, index=False)
    pd.Series(sorted(set(report.anomalies)), name="anomalie").to_csv(
        config.OUTPUT_DIR / "anomalies.csv", index=False
    )
    print(f"\nReferentiel sujet ecrit  : {destination}")
    print(f"Table minutes en memoire : {minutes.shape[0]:,} lignes")


if __name__ == "__main__":
    main()