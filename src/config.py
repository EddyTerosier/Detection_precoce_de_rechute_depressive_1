"""Parametres centralises du cas concret.

Tout choix de conception discutable est expose ici, de facon a pouvoir etre
modifie sans toucher au code et a etre cite tel quel dans le memoire.
"""

from pathlib import Path

# --- Chemins ---------------------------------------------------------------
# Racine du projet, deduite de l'emplacement de ce fichier.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Chaque dossier doit contenir directement les sous-dossiers d'actigraphie et le
# fichier de metadonnees. Attention : selon la facon dont les archives ont ete
# decompressees, un niveau "data/" supplementaire peut s'intercaler.
#
# DEPRESJON attendu : condition/, control/, scores.csv
RAW_DEPRESJON_DIR = PROJECT_DIR / "data" / "raw" / "depresjon" / "data"
# PSYKOSE attendu : patient/, patients_info.csv
RAW_PSYKOSE_DIR = PROJECT_DIR / "data" / "raw" / "psykose"

OUTPUT_DIR = PROJECT_DIR / "outputs"

# --- Classes ---------------------------------------------------------------
# Les 32 temoins sont communs aux deux jeux de donnees : les fichiers de
# PSYKOSE/control sont byte-identiques a ceux de DEPRESJON/control. Ils ne sont
# donc charges qu'une seule fois, depuis DEPRESJON.
CLASS_NAMES = {
    0: "temoin",
    1: "depression",
    2: "schizophrenie",
}

# Effectifs attendus, verifies a l'audit des archives. Sert de garde-fou : si le
# chargement produit autre chose, c'est que les donnees sources ont change.
EXPECTED_SUBJECTS_PER_CLASS = {0: 32, 1: 23, 2: 22}
EXPECTED_SUBJECTS = 77

# --- Selection des journees ------------------------------------------------
# Une journee valide compte exactement 1440 minutes (24 h a la minute).
MINUTES_PER_DAY = 1440

# Nombre de journees conservees par sujet. Fixe pour que chaque sujet pese le
# meme poids dans l'apprentissage : aucun individu tres enregistre ne peut
# dominer le modele. 12 est le maximum possible, impose par le sujet ayant le
# moins de journees completes (condition_8 de DEPRESJON).
DAYS_PER_SUBJECT = 12

# Variante de robustesse : si True, toutes les journees completes sont
# conservees, sans troncature.
KEEP_ALL_COMPLETE_DAYS = False

# --- Validite d'une journee -------------------------------------------------
# Trois artefacts distincts ont ete observes dans les journees completes des deux
# jeux de donnees. Chaque seuil ci-dessous en ecarte un. Ces journees sont
# exclues et non corrigees : imputer une activite inventee serait plus
# dommageable que perdre l'observation.

# 1. Dispositif non porte. 86 journees sont a activite strictement nulle sur 24 h
#    et 61 autres a moyenne inferieure a 1 compte par minute, ce qui est
#    physiologiquement impossible pour une personne portant l'appareil.
MIN_DAILY_ACTIVITY_MEAN = 1.0

# 2. Signal constant. Six journees presentent la meme valeur exactement sur les
#    1440 minutes (par exemple 3 ou 7 comptes), ce qui correspond a une valeur de
#    repos figee du capteur et non a de l'activite humaine.
MIN_DAILY_ACTIVITY_STD = 1.0

# 3. Journee trop peu documentee. Une journee valide compte typiquement entre 55
#    et 70 % de minutes actives ; en dessous du seuil ci-dessous, la journee ne
#    permet pas de caracteriser un rythme (un cas observe a 1,8 %).
MIN_ACTIVE_MINUTES_FRACTION = 0.05

# --- Codage des metadonnees sources ----------------------------------------
# DEPRESJON, colonne gender : 1 = femme, 2 = homme.
DEPRESJON_SEX = {1: "F", 2: "M"}
# PSYKOSE, colonne gender : chaine de caracteres.
PSYKOSE_SEX = {"female": "F", "male": "M"}

# DEPRESJON, colonne afftype : sous-type du trouble de l'humeur.
AFFTYPE_LABELS = {1: "bipolaire II", 2: "unipolaire", 3: "bipolaire I"}

# DEPRESJON, colonne inpatient : 1 = hospitalise, 2 = ambulatoire.
# PSYKOSE ne documente pas ce statut : la valeur reste inconnue pour les
# patients schizophrenes, ce qui constitue une limite explicite du cas concret.
INPATIENT_LABELS = {1: "hospitalise", 2: "ambulatoire"}

# --- Reproductibilite ------------------------------------------------------
RANDOM_SEED = 42