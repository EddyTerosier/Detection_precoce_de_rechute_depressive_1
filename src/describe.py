"""Analyse descriptive des features et generation des figures du memoire.

Produit trois tableaux et cinq figures. Le fil directeur est le duel depression /
schizophrenie : c'est lui qui porte la demonstration, puisque les deux groupes
sont hypoactifs par rapport aux temoins et qu'un seuil sur l'activite moyenne ne
peut donc pas les separer.

Tableaux :
  comparaison_features.csv  moyenne par classe, tailles d'effet et AUC des duels
  correlations_fortes.csv   paires de features redondantes
  stabilite_interjournaliere.csv  IS par sujet, mesure multi-journaliere

Figures (niveaux de gris, lisibles a l'impression) :
  fig1_correlations.png     matrice de correlation des 19 features
  fig2_boxplots.png         distributions des features separant les pathologies
  fig3_profil_horaire.png   activite moyenne sur 24 h par classe
  fig4_actogrammes.png      un actogramme representatif par classe
  fig5_discrimination.png   AUC de chaque feature sur les trois duels
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import features as feat
import load

HOURS_PER_DAY = 24
CORRELATION_THRESHOLD = 0.90
N_BOXPLOT_FEATURES = 6
FIGURE_DPI = 200

TEMOIN, DEPRESSION, SCHIZOPHRENIA = 0, 1, 2

# Duels analyses. Le troisieme est celui qui porte la demonstration.
DUELS = {
    "dep_vs_tem": (DEPRESSION, TEMOIN),
    "sch_vs_tem": (SCHIZOPHRENIA, TEMOIN),
    "sch_vs_dep": (SCHIZOPHRENIA, DEPRESSION),
}

# Nuances de gris par classe, du plus clair au plus fonce.
CLASS_SHADES = {TEMOIN: "0.85", DEPRESSION: "0.60", SCHIZOPHRENIA: "0.30"}
CLASS_STYLES = {TEMOIN: ("0.55", "--"), DEPRESSION: ("0.30", "-."), SCHIZOPHRENIA: ("black", "-")}


# --- Tableaux ---------------------------------------------------------------


def _duel_auc(frame: pd.DataFrame, column: str, positive: int, negative: int) -> float:
    """AUC d'une feature seule sur un duel entre deux classes.

    Symetrisee : 0.5 signifie aucun pouvoir discriminant, quel que soit le sens
    de l'ecart entre les deux groupes.
    """
    subset = frame[frame["class_label"].isin([positive, negative])]
    truth = (subset["class_label"] == positive).astype(int)
    auc = roc_auc_score(truth, subset[column])
    return max(auc, 1 - auc)


def comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Moyenne par classe, tailles d'effet et AUC univariee sur les trois duels."""
    rows = []
    for column in feat.FEATURE_COLUMNS:
        by_class = {
            label: frame.loc[frame["class_label"] == label, column]
            for label in config.CLASS_NAMES
        }
        row = {
            "feature": column,
            "temoin_moy": by_class[TEMOIN].mean(),
            "depression_moy": by_class[DEPRESSION].mean(),
            "schizophrenie_moy": by_class[SCHIZOPHRENIA].mean(),
            "d_dep_vs_tem": feat._cohen_d(by_class[DEPRESSION], by_class[TEMOIN]),
            "d_sch_vs_tem": feat._cohen_d(by_class[SCHIZOPHRENIA], by_class[TEMOIN]),
            "d_sch_vs_dep": feat._cohen_d(by_class[SCHIZOPHRENIA], by_class[DEPRESSION]),
        }
        for name, (positive, negative) in DUELS.items():
            row[f"auc_{name}"] = _duel_auc(frame, column, positive, negative)
        rows.append(row)

    table = pd.DataFrame(rows)
    # Tri par pouvoir discriminant sur le duel qui porte la demonstration.
    return table.sort_values("auc_sch_vs_dep", ascending=False, ignore_index=True)


def strong_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Liste les paires de features dont la correlation absolue depasse le seuil."""
    matrix = frame[feat.FEATURE_COLUMNS].corr()
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    pairs = pairs[pairs["correlation"].abs() >= CORRELATION_THRESHOLD]
    return pairs.sort_values("correlation", key=abs, ascending=False, ignore_index=True)


def interdaily_stability(minutes: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Stabilite inter-journaliere par sujet.

    IS compare la variance du profil horaire moyen a la variance totale du
    signal : une valeur proche de 1 traduit un rythme tres regulier d'un jour a
    l'autre. Cette mesure est multi-journaliere par nature et ne peut donc pas
    figurer parmi les features journalieres.
    """
    rows = []
    for subject_id, block in minutes.groupby("subject_id"):
        hourly = (
            block.assign(hour=block["timestamp"].dt.hour)
            .groupby(["date", "hour"])["activity"]
            .mean()
        )
        series = hourly.to_numpy()
        profile = hourly.groupby("hour").mean().to_numpy()
        grand_mean = series.mean()

        between = np.sum((profile - grand_mean) ** 2) / HOURS_PER_DAY
        total = np.sum((series - grand_mean) ** 2) / len(series)
        rows.append({"subject_id": subject_id, "interdaily_stability": between / total})

    stability = pd.DataFrame(rows)
    return stability.merge(subjects[["subject_id", "class_label", "class_name"]], on="subject_id")


def representative_subjects(frame: pd.DataFrame) -> dict[int, str]:
    """Sujet le plus proche de la mediane de sa classe, pour chaque classe.

    Choisir des cas medians plutot qu'extremes evite d'illustrer le propos avec
    des sujets atypiques.
    """
    per_subject = frame.groupby(["subject_id", "class_label"])["activity_mean"].mean().reset_index()
    selected = {}
    for label in config.CLASS_NAMES:
        block = per_subject[per_subject["class_label"] == label]
        target = block["activity_mean"].median()
        selected[label] = block.loc[(block["activity_mean"] - target).abs().idxmin(), "subject_id"]
    return selected


# --- Figures ---------------------------------------------------------------


def plot_correlations(frame: pd.DataFrame, path) -> None:
    matrix = frame[feat.FEATURE_COLUMNS].corr()
    fig, axes = plt.subplots(figsize=(9, 8))
    image = axes.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes.set_xticks(range(len(feat.FEATURE_COLUMNS)))
    axes.set_xticklabels(feat.FEATURE_COLUMNS, rotation=90, fontsize=7)
    axes.set_yticks(range(len(feat.FEATURE_COLUMNS)))
    axes.set_yticklabels(feat.FEATURE_COLUMNS, fontsize=7)
    axes.set_title("Matrice de correlation des features journalieres", fontsize=11)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Correlation de Pearson")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_boxplots(frame: pd.DataFrame, table: pd.DataFrame, path) -> None:
    """Distributions des features separant le mieux les deux pathologies."""
    selected = table["feature"].head(N_BOXPLOT_FEATURES).tolist()
    labels = sorted(config.CLASS_NAMES)
    names = ["Temoin", "Depress.", "Schizo."]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for axis, column in zip(axes.ravel(), selected):
        data = [frame.loc[frame["class_label"] == label, column] for label in labels]
        parts = axis.boxplot(data, patch_artist=True)
        # Etiquettes posees apres coup : le parametre tick_labels de boxplot
        # n'existe qu'a partir de matplotlib 3.9.
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(names, fontsize=8)
        for patch, label in zip(parts["boxes"], labels):
            patch.set_facecolor(CLASS_SHADES[label])
        for median in parts["medians"]:
            median.set_color("black")
        axis.set_title(column, fontsize=9)
        axis.tick_params(labelsize=8)

    fig.suptitle(
        "Features separant le mieux schizophrenie et depression", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_hourly_profile(minutes: pd.DataFrame, subjects: pd.DataFrame, path) -> None:
    """Activite moyenne heure par heure, mediane et intervalle interquartile."""
    hourly = (
        minutes.assign(hour=minutes["timestamp"].dt.hour)
        .groupby(["subject_id", "hour"])["activity"]
        .mean()
        .reset_index()
        .merge(subjects[["subject_id", "class_label"]], on="subject_id")
    )

    fig, axes = plt.subplots(figsize=(9.5, 4.8))
    for label, (color, linestyle) in CLASS_STYLES.items():
        block = hourly[hourly["class_label"] == label]
        stats = block.groupby("hour")["activity"].agg(
            median="median",
            low=lambda values: values.quantile(0.25),
            high=lambda values: values.quantile(0.75),
        )
        axes.plot(stats.index, stats["median"], color=color, linestyle=linestyle,
                  linewidth=2, label=config.CLASS_NAMES[label].capitalize())
        axes.fill_between(stats.index, stats["low"], stats["high"], color=color, alpha=0.12)

    axes.set_xlabel("Heure de la journee")
    axes.set_ylabel("Activite moyenne (comptes / minute)")
    axes.set_title("Profil d'activite sur 24 h : mediane et intervalle interquartile",
                   fontsize=11)
    axes.set_xticks(range(0, HOURS_PER_DAY, 2))
    axes.legend()
    axes.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_actograms(minutes: pd.DataFrame, selected: dict[int, str], path) -> None:
    """Actogrammes representatifs : une ligne par journee, une colonne par minute."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    vmax = np.percentile(minutes["activity"], 99)

    for axis, (label, subject_id) in zip(axes, sorted(selected.items())):
        block = minutes[minutes["subject_id"] == subject_id]
        grid = (
            block.assign(
                minute=block["timestamp"].dt.hour * 60 + block["timestamp"].dt.minute
            )
            .pivot_table(index="date", columns="minute", values="activity")
            .to_numpy()
        )
        axis.imshow(grid, aspect="auto", cmap="Greys", vmin=0, vmax=vmax,
                    extent=(0, HOURS_PER_DAY, grid.shape[0], 0))
        axis.set_xticks(range(0, HOURS_PER_DAY + 1, 4))
        axis.set_xlabel("Heure de la journee")
        axis.set_ylabel("Journee de suivi")
        axis.set_title(f"{config.CLASS_NAMES[label].capitalize()} - {subject_id}", fontsize=10)

    fig.suptitle("Actogrammes : plus le trace est sombre, plus l'activite est elevee",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_discrimination(table: pd.DataFrame, path) -> None:
    """AUC univariee de chaque feature sur les trois duels.

    Met en evidence l'asymetrie centrale : les features de niveau d'activite
    separent bien chaque pathologie des temoins, mais tres mal les deux
    pathologies entre elles.
    """
    ordered = table.sort_values("auc_sch_vs_dep", ascending=True)
    positions = np.arange(len(ordered))
    height = 0.26

    fig, axes = plt.subplots(figsize=(9.5, 7))
    series = [
        ("auc_dep_vs_tem", "Depression / temoin", "0.80"),
        ("auc_sch_vs_tem", "Schizophrenie / temoin", "0.55"),
        ("auc_sch_vs_dep", "Schizophrenie / depression", "0.15"),
    ]
    for index, (column, legend, shade) in enumerate(series):
        axes.barh(positions + (index - 1) * height, ordered[column], height=height,
                  color=shade, label=legend)

    axes.axvline(0.5, color="black", linestyle=":", linewidth=1)
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["feature"], fontsize=8)
    axes.set_xlim(0.45, 1.0)
    axes.set_xlabel("AUC univariee (0.5 = aucun pouvoir discriminant)")
    axes.set_title("Pouvoir discriminant de chaque feature, duel par duel", fontsize=11)
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


# --- Orchestration ---------------------------------------------------------


def print_summary(table: pd.DataFrame, correlations: pd.DataFrame,
                  stability: pd.DataFrame) -> None:
    print("=" * 92)
    print("POUVOIR DISCRIMINANT UNIVARIE (tri par duel schizophrenie / depression)")
    print("=" * 92)
    print(f"{'feature':<26}{'AUC dep/tem':>13}{'AUC sch/tem':>13}{'AUC sch/dep':>13}"
          f"{'d sch/dep':>12}")
    print("-" * 92)
    for row in table.itertuples():
        print(f"{row.feature:<26}{row.auc_dep_vs_tem:>13.3f}{row.auc_sch_vs_tem:>13.3f}"
              f"{row.auc_sch_vs_dep:>13.3f}{row.d_sch_vs_dep:>12.2f}")
    print("-" * 92)
    # Constat calcule et non ecrit en dur : le classement depend des donnees.
    best = table.iloc[0]
    print(f"Meilleure feature seule sur le duel schizophrenie / depression : "
          f"{best['feature']} (AUC {best['auc_sch_vs_dep']:.3f}).")
    print(f"La meme feature n'atteint que {best['auc_dep_vs_tem']:.3f} sur le duel")
    print("depression / temoin : le pouvoir discriminant depend de la tache posee.")

    print()
    print("=" * 92)
    print(f"PAIRES DE FEATURES REDONDANTES (|r| >= {CORRELATION_THRESHOLD})")
    print("=" * 92)
    if correlations.empty:
        print("aucune")
    else:
        for row in correlations.itertuples():
            print(f"  {row.feature_a:<24} {row.feature_b:<24} r = {row.correlation:+.3f}")

    print()
    print("=" * 92)
    print("STABILITE INTER-JOURNALIERE (niveau sujet)")
    print("=" * 92)
    summary = stability.groupby("class_name")["interdaily_stability"].agg(
        ["count", "mean", "std"]
    )
    for name, row in summary.iterrows():
        print(f"  {name:<16} n = {int(row['count']):<4} "
              f"IS = {row['mean']:.3f} +/- {row['std']:.3f}")
    print("=" * 92)


def main() -> None:
    minutes, subjects, _ = load.load_dataset()
    frame = feat.build_features(minutes, subjects)
    feat.check_integrity(frame)

    tables_dir = config.OUTPUT_DIR / "tables"
    figures_dir = config.OUTPUT_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = comparison_table(frame)
    correlations = strong_correlations(frame)
    stability = interdaily_stability(minutes, subjects)
    print_summary(table, correlations, stability)

    table.to_csv(tables_dir / "comparaison_features.csv", index=False)
    correlations.to_csv(tables_dir / "correlations_fortes.csv", index=False)
    stability.to_csv(tables_dir / "stabilite_interjournaliere.csv", index=False)

    selected = representative_subjects(frame)
    plot_correlations(frame, figures_dir / "fig1_correlations.png")
    plot_boxplots(frame, table, figures_dir / "fig2_boxplots.png")
    plot_hourly_profile(minutes, subjects, figures_dir / "fig3_profil_horaire.png")
    plot_actograms(minutes, selected, figures_dir / "fig4_actogrammes.png")
    plot_discrimination(table, figures_dir / "fig5_discrimination.png")

    print("\nSujets illustratifs retenus :")
    for label, subject_id in sorted(selected.items()):
        print(f"  {config.CLASS_NAMES[label]:<16}: {subject_id}")
    print(f"\nTableaux ecrits : {tables_dir}")
    print(f"Figures ecrites : {figures_dir}")


if __name__ == "__main__":
    main()