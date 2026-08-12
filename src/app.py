"""Tableau de bord de restitution clinique.

Lecture seule : ce module n'entraine aucun modele. Il affiche les artefacts
produits par export.py, qui doit donc etre execute au prealable.

Lancement, depuis le dossier src :
    streamlit run app.py

Les scores affiches sont des probabilites hors echantillon : chaque journee a ete
notee par un modele qui n'avait jamais vu le sujet concerne. Un tableau de bord
affichant les scores d'un modele entraine sur le patient consulte donnerait une
impression de fiabilite trompeuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
import export
import load

HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
DASHBOARD_DIR = config.OUTPUT_DIR / export.EXPORT_DIR_NAME

# Libelles cliniques des variables retenues. Un nom technique dans une interface
# destinee a un soignant est un obstacle a l'interpretation.
FEATURE_LABELS = {
    "activity_mean": "Niveau d'activite moyen",
    "activity_cv": "Variabilite relative de l'activite",
    "n_active_bouts": "Nombre de plages d'activite",
    "pct_zero_minutes": "Part de la journee sans mouvement",
    "night_day_ratio": "Rapport activite nocturne / diurne",
    "l5_onset_hour": "Heure de debut de la periode de repos",
    "relative_amplitude": "Contraste activite / repos",
}

CLASS_ORDER = [config.CLASS_NAMES[label] for label in sorted(config.CLASS_NAMES)]


# --- Chargement des donnees -------------------------------------------------


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Lit les artefacts produits par export.py."""
    scores = pd.read_csv(DASHBOARD_DIR / "scores_journaliers.csv")
    subjects = pd.read_csv(DASHBOARD_DIR / "sujets.csv")
    coefficients = pd.read_csv(DASHBOARD_DIR / "coefficients.csv")
    with open(DASHBOARD_DIR / "evaluation.json", encoding="utf-8") as stream:
        evaluation = json.load(stream)
    return scores, subjects, coefficients, evaluation


@st.cache_data
def load_minutes() -> pd.DataFrame:
    """Recharge les series minute par minute, necessaires aux actogrammes.

    Volontairement non exportees par export.py : 1,3 million de lignes deja
    disponibles a la source. Le cache de Streamlit evite de relire a chaque
    interaction.
    """
    minutes, _, _ = load.load_dataset()
    return minutes


def artifacts_available() -> bool:
    required = ["scores_journaliers.csv", "sujets.csv", "coefficients.csv", "evaluation.json"]
    return all((DASHBOARD_DIR / name).is_file() for name in required)


# --- Composants d'affichage -------------------------------------------------


def plot_actogram(minutes: pd.DataFrame, subject_id: str,
                  highlighted_day: int) -> plt.Figure:
    """Actogramme du sujet, avec la journee selectionnee encadree."""
    block = minutes[minutes["subject_id"] == subject_id].copy()
    block["minute"] = (
        block["timestamp"].dt.hour * MINUTES_PER_HOUR + block["timestamp"].dt.minute
    )
    grid = block.pivot_table(index="date", columns="minute", values="activity")
    grid = grid.reindex(sorted(grid.index))

    figure, axes = plt.subplots(figsize=(9, 3.6))
    axes.imshow(
        grid.to_numpy(),
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=np.percentile(block["activity"], 99),
        extent=(0, HOURS_PER_DAY, len(grid), 0),
    )
    axes.axhspan(highlighted_day - 1, highlighted_day, facecolor="none",
                 edgecolor="crimson", linewidth=2)
    axes.set_xticks(range(0, HOURS_PER_DAY + 1, 3))
    axes.set_xlabel("Heure de la journee")
    axes.set_ylabel("Journee de suivi")
    axes.set_title("Actogramme : plus le trace est sombre, plus l'activite est elevee",
                   fontsize=10)
    figure.tight_layout()
    return figure


def plot_score_curve(subject_scores: pd.DataFrame, threshold: float,
                     highlighted_day: int) -> plt.Figure:
    """Evolution des probabilites de classe au fil des journees de suivi."""
    figure, axes = plt.subplots(figsize=(9, 3.4))
    styles = {"temoin": ("0.6", "--"), "depression": ("0.25", "-."),
              "schizophrenie": ("black", "-")}

    for name, (color, linestyle) in styles.items():
        axes.plot(subject_scores["day_index"], subject_scores[f"p_{name}"] * 100,
                  color=color, linestyle=linestyle, marker="o", markersize=4,
                  linewidth=1.8, label=name.capitalize())

    axes.axhline(threshold * 100, color="crimson", linestyle=":", linewidth=1.5,
                 label=f"Seuil d'alerte ({threshold:.0%})")
    axes.axvline(highlighted_day, color="crimson", alpha=0.25, linewidth=6)
    axes.set_xlabel("Journee de suivi")
    axes.set_ylabel("Probabilite (%)")
    axes.set_ylim(0, 100)
    axes.set_xticks(subject_scores["day_index"])
    axes.legend(fontsize=8, ncol=4, loc="upper center")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_contributions(day: pd.Series, class_name: str) -> plt.Figure:
    """Contribution de chaque variable au score de la classe consideree.

    Coefficient du pli multiplie par la valeur standardisee de la variable : une
    barre positive pousse le score vers la classe, une barre negative l'en
    eloigne.
    """
    values = {
        FEATURE_LABELS[column]: day[f"contrib_{class_name}_{column}"]
        for column in export.RETAINED_FEATURES
    }
    ordered = pd.Series(values).sort_values()

    figure, axes = plt.subplots(figsize=(7.5, 3.6))
    colors = ["0.75" if value < 0 else "0.25" for value in ordered]
    axes.barh(ordered.index, ordered.to_numpy(), color=colors)
    axes.axvline(0, color="black", linewidth=1)
    axes.set_xlabel(f"Contribution au score « {class_name} »")
    axes.tick_params(labelsize=8)
    figure.tight_layout()
    return figure


def plot_roc(evaluation: dict, threshold: float) -> plt.Figure:
    """Courbe ROC du duel depression / schizophrenie et position du seuil."""
    roc = evaluation["roc_duel"]
    thresholds = np.array(roc["seuils"])
    index = int(np.argmin(np.abs(thresholds - threshold)))

    figure, axes = plt.subplots(figsize=(4.6, 4.4))
    axes.plot(roc["fpr"], roc["tpr"], color="black", linewidth=2)
    axes.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1)
    axes.scatter([roc["fpr"][index]], [roc["tpr"][index]], color="crimson", zorder=5,
                 label="Seuil retenu")
    axes.set_xlabel("Taux de faux positifs")
    axes.set_ylabel("Taux de vrais positifs")
    axes.set_title("Duel depression / schizophrenie", fontsize=10)
    axes.legend(fontsize=8, loc="lower right")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def confusion_table(matrix: list[list[int]]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER)
    frame.index.name = "Reel"
    return frame


# --- Pages ------------------------------------------------------------------


def render_subject_page(scores: pd.DataFrame, subjects: pd.DataFrame,
                        minutes: pd.DataFrame) -> None:
    st.sidebar.header("Selection")
    subject_id = st.sidebar.selectbox(
        "Sujet", sorted(scores["subject_id"].unique()),
        help="Identifiant anonymise du participant",
    )

    subject_scores = scores[scores["subject_id"] == subject_id].sort_values("day_index")
    metadata = subjects[subjects["subject_id"] == subject_id].iloc[0]

    highlighted_day = st.sidebar.slider(
        "Journee de suivi", int(subject_scores["day_index"].min()),
        int(subject_scores["day_index"].max()), int(subject_scores["day_index"].min()),
    )
    threshold = st.sidebar.slider(
        "Seuil d'alerte", 0.05, 0.95, 0.50, 0.05,
        help="Seuil applique au score, a definir selon le cout clinique relatif "
             "des deux types d'erreur",
    )

    day = subject_scores[subject_scores["day_index"] == highlighted_day].iloc[0]

    st.subheader(f"Sujet {subject_id}")
    columns = st.columns(4)
    columns[0].metric("Classe documentee", metadata["class_name"])
    columns[1].metric("Classe la plus probable", day["predicted_name"])
    columns[2].metric("Score de vigilance", f"{day['score_vigilance']:.0f} / 100")
    columns[3].metric(
        f"{metadata['severity_scale'] if pd.notna(metadata['severity_scale']) else 'Severite'}",
        f"{metadata['severity_score']:.0f}" if pd.notna(metadata["severity_score"]) else "n.d.",
    )

    st.caption(
        f"Sexe : {metadata['sex']} — Age : {metadata['age_range']} — "
        f"Prise en charge : {metadata['inpatient']} — "
        f"Sous-type : {metadata['subtype'] if pd.notna(metadata['subtype']) else 'sans objet'}"
    )

    if day["predicted_name"] != metadata["class_name"]:
        st.warning(
            f"Pour cette journee, la classe la plus probable ({day['predicted_name']}) "
            f"differe de la classe documentee ({metadata['class_name']}). "
            "Un desaccord isole n'a pas de valeur diagnostique."
        )

    st.markdown("#### Evolution des scores au fil du suivi")
    st.pyplot(plot_score_curve(subject_scores, threshold, highlighted_day))

    st.markdown("#### Activite observee")
    st.pyplot(plot_actogram(minutes, subject_id, highlighted_day))

    st.markdown(f"#### Facteurs influents, journee {highlighted_day}")
    explained = st.selectbox("Score explique", CLASS_ORDER,
                             index=CLASS_ORDER.index(day["predicted_name"]))
    st.pyplot(plot_contributions(day, explained))
    st.caption(
        "Barre foncee vers la droite : la variable pousse le score vers cette "
        "classe. Barre claire vers la gauche : elle l'en eloigne."
    )


def render_evaluation_page(evaluation: dict, coefficients: pd.DataFrame) -> None:
    st.sidebar.header("Parametre")
    threshold = st.sidebar.slider("Seuil du duel", 0.05, 0.95, 0.50, 0.05)

    st.subheader("Performance mesuree hors echantillon")
    columns = st.columns(3)
    columns[0].metric("Sujets", evaluation["n_sujets"])
    columns[1].metric("Journees", evaluation["n_journees"])
    subject_matrix = np.array(evaluation["matrice_sujet"])
    columns[2].metric(
        "Sujets bien classes",
        f"{int(np.trace(subject_matrix))} / {int(subject_matrix.sum())}",
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Matrice de confusion, niveau sujet**")
        st.dataframe(confusion_table(evaluation["matrice_sujet"]))
        st.markdown("**Matrice de confusion, niveau journee**")
        st.dataframe(confusion_table(evaluation["matrice_jour"]))
    with right:
        st.pyplot(plot_roc(evaluation, threshold))

    st.markdown("#### Coefficients du modele de reference")
    st.caption(
        "Variables standardisees : les coefficients sont comparables entre eux. "
        "Un coefficient positif pousse vers la classe correspondante."
    )
    display = coefficients.copy()
    display["feature"] = display["feature"].map(FEATURE_LABELS)
    st.dataframe(display.set_index("feature").style.format("{:+.3f}"))


def render_method_page(evaluation: dict) -> None:
    st.subheader("Methode et limites")
    st.markdown(
        f"""
**Donnees.** Actigraphie au poignet, agregee a la minute, issue des jeux
DEPRESJON et PSYKOSE collectes a l'hopital universitaire de Haukeland.
{evaluation['n_sujets']} sujets, {evaluation['n_journees']} journees de 24 h
completes, douze journees par sujet afin qu'aucun participant ne pese plus qu'un
autre.

**Tache.** Distinguer trois profils a partir de la seule activite motrice :
temoin sain, episode depressif caracterise, schizophrenie. Il s'agit d'une aide a
l'orientation, non d'un diagnostic.

**Modele.** Regression logistique multinomiale sur sept variables journalieres,
retenue apres comparaison avec un jeu de dix-neuf variables et une foret
aleatoire. Elle egale les modeles plus complexes tout en restant interpretable
variable par variable.

**Validation.** Leave-one-subject-out : chaque sujet est evalue par un modele
entraine sur tous les autres. Les scores affiches sont donc hors echantillon.
Un decoupage aleatoire des journees, qui melangerait les journees d'un meme
sujet entre apprentissage et test, surestimerait la performance et masquerait
surtout l'incertitude reelle.
        """
    )

    st.warning(
        """
**Limites a connaitre avant tout usage.**

Les patients schizophrenes de PSYKOSE etaient hospitalises, tandis que dix-huit
des vingt-trois patients deprimes etaient suivis en ambulatoire. L'hospitalisation
modifie l'activite motrice autant que la pathologie elle-meme : une part de la
separation observee reflete donc l'environnement de vie.

L'echantillon compte soixante-dix-sept sujets. Les intervalles de confiance sont
larges et aucun ecart de performance inferieur a cinq points n'est concluant.

La cohorte ne comprend aucun episode depressif leger : l'outil n'a pas ete
evalue sur les formes frustes.

Les donnees datent des annees 2002 a 2006 et proviennent d'un seul centre.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Actigraphie - aide a l'orientation", layout="wide")
    st.title("Actigraphie et orientation diagnostique")
    st.error(
        "**Outil d'aide a la decision sous supervision medicale obligatoire.** "
        "Les scores presentes ne constituent pas un diagnostic. Toute alerte doit "
        "etre interpretee par un clinicien au regard de l'examen et du contexte du "
        "patient. Travail academique realise sur des donnees de recherche "
        "anonymisees : cet outil n'est pas un dispositif medical."
    )

    if not artifacts_available():
        st.warning(
            f"Artefacts absents de {DASHBOARD_DIR}. Executer `python export.py` "
            "depuis le dossier src avant de lancer le tableau de bord."
        )
        return

    scores, subjects, coefficients, evaluation = load_artifacts()

    page = st.sidebar.radio(
        "Vue", ["Suivi individuel", "Evaluation du modele", "Methode et limites"]
    )
    if page == "Suivi individuel":
        render_subject_page(scores, subjects, load_minutes())
    elif page == "Evaluation du modele":
        render_evaluation_page(evaluation, coefficients)
    else:
        render_method_page(evaluation)


if __name__ == "__main__":
    main()