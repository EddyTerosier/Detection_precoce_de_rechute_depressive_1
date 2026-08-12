# Cas concret — orientation diagnostique différentielle par actigraphie

Support technique de la partie 3 du mémoire *« L'intelligence artificielle pour
l'analyse des comportements psychiatriques et l'aide à la décision médicale »*.

---

## 1. Objet

À partir de la seule activité motrice enregistrée par un bracelet
actigraphique, produire pour chaque journée un score d'orientation entre trois
profils : **témoin sain**, **épisode dépressif caractérisé**, **schizophrénie**.

Cet outil ne pose pas de diagnostic, ne prescrit rien et ne prédit pas le risque
suicidaire. Il constitue une aide à l'orientation sous supervision médicale
obligatoire. Travail académique réalisé sur des données de recherche
anonymisées : ce n'est pas un dispositif médical.

### Pourquoi cette tâche, et pas la prédiction de rechute

Le projet visait initialement la détection précoce de rechute dépressive. Les
données publiques disponibles ne contiennent pas d'événement de rechute
longitudinal : annoncer une prédiction de rechute et livrer une détection d'état
aurait été une surpromesse. La tâche a donc été recadrée sur ce que les données
portent réellement.

Un premier cadrage binaire — patient contre témoin — a ensuite été abandonné
pour une raison mesurée et non supposée : sur cette tâche, **un simple seuil sur
l'activité moyenne journalière égale un modèle à dix-neuf variables** (AUC 0,755
contre 0,758). Le signal y est unidimensionnel, et aucun modèle multivarié ne
peut faire mieux qu'un seuil optimal sur cet axe unique. Le détail de cette
démonstration figure en section 8.7 : c'est un résultat du mémoire, pas un échec
écarté.

La tâche à trois classes change la situation par construction. Dépression et
schizophrénie sont **toutes deux hypoactives** par rapport aux témoins, donc
situées du même côté de n'importe quel seuil sur l'activité. Les distinguer
exige de mobiliser d'autres dimensions — fragmentation, durée des plages de
repos, variabilité du rythme. La règle à une seule variable est donc
structurellement incapable de résoudre cette tâche, et l'écart mesuré entre elle
et le modèle multivarié constitue la démonstration.

---

## 2. Données

Deux jeux publics, collectés par la même équipe, dans le même hôpital, avec le
même appareil et le même protocole.

| Jeu | Contenu | Source |
|---|---|---|
| **DEPRESJON** | 23 patients en épisode dépressif, 32 témoins sains | http://datasets.simula.no/depresjon/ — DOI 10.5281/zenodo.1219550 |
| **PSYKOSE** | 22 patients schizophrènes, mêmes 32 témoins | https://datasets.simula.no/psykose/ |

- Actiwatch AW4 au poignet, échantillonnage 32 Hz, comptes d'activité agrégés à
  la minute
- Trois colonnes par fichier sujet : `timestamp`, `date`, `activity`
- Évaluation clinique par des experts de l'hôpital universitaire de Haukeland
- Métadonnées : sexe, tranche d'âge, sous-type, sévérité (MADRS pour la
  dépression, BPRS pour la schizophrénie)

**Les 32 témoins sont communs aux deux jeux.** Les fichiers `PSYKOSE/control`
sont byte-identiques à `DEPRESJON/control` : ils ne sont chargés qu'une seule
fois, faute de quoi ils compteraient double.

### Mise en place

Les données ne sont pas versionnées avec le code. Après décompression,
`src/config.py` doit pointer sur les dossiers contenant **directement** les
sous-dossiers d'actigraphie :

```
data/raw/depresjon/data/     condition/  control/  scores.csv
data/raw/psykose/            patient/    patients_info.csv
```

L'archive DEPRESJON de Zenodo introduit un niveau `data/` supplémentaire, d'où
le chemin ci-dessus ; celle de PSYKOSE non. Ces deux chemins sont les seuls
paramètres à ajuster.

### Composition de la cohorte

| Classe | Sujets | Journées | Sévérité (moyenne) |
|---|---|---|---|
| Témoin | 32 | 384 | — |
| Dépression | 23 | 276 | MADRS 22,7 (13–29) |
| Schizophrénie | 22 | 264 | BPRS 50,0 (34–59) |
| **Total** | **77** | **924** | |

Sous-types : dépression — 15 unipolaires, 7 bipolaires II, 1 bipolaire I ;
schizophrénie — 17 paranoïdes, 5 non paranoïdes.

---

## 3. Installation et exécution

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Les scripts se lancent depuis `src/`, dans cet ordre :

| Commande | Rôle | Durée |
|---|---|---|
| `python load.py` | chargement, validation, sélection des journées | ~15 s |
| `python features.py` | extraction des 19 features journalières | ~15 s |
| `python describe.py` | tableaux et figures descriptives | ~30 s |
| `python train.py` | comparaison des modèles et jeux de variables | ~3 min |
| `python robustness.py` | analyses de robustesse et de sensibilité | ~20 s |
| `python export.py` | artefacts du modèle retenu | ~10 s |
| `streamlit run app.py` | tableau de bord de restitution | — |

`export.py` doit avoir été exécuté avant `app.py`.

---

## 4. Architecture

```
src/config.py       paramètres et décisions de conception — source de vérité unique
src/load.py         lecture des deux sources, validation, sélection des journées
src/features.py     1440 minutes -> 19 features journalières
src/describe.py     analyse descriptive, tableaux et figures du mémoire
src/train.py        validation croisée groupée, comparaison des modèles
src/robustness.py   confondants, fuite de données, coûts cliniques
src/export.py       artefacts du modèle retenu, pour la restitution
src/app.py          tableau de bord (lecture seule, aucun entraînement)
outputs/            artefacts générés, non versionnés
```

Le découpage sépare trois responsabilités : `train.py` compare, `export.py` fige
la combinaison retenue, `app.py` affiche sans jamais calculer. Le tableau de bord
démarre donc sans réentraîner quoi que ce soit.

---

## 5. Journal des décisions de conception

Chaque décision est paramétrée dans `src/config.py`. Les justifications sont
rédigées pour être citables directement dans le mémoire.

### 5.1 Périmètre clinique

La classe « dépression » regroupe les 23 patients, **toutes polarités
confondues**. La cible est l'épisode dépressif caractérisé, non le trouble de
fond.

L'actigraphie capte un état psychomoteur, pas une polarité — c'est également ce
que mesure le MADRS. Restreindre aux unipolaires ramènerait l'effectif à 15
sujets, au détriment de la stabilité de la validation. Une analyse de
sensibilité vérifie que le résultat tient sur les unipolaires seuls (section
9.1).

### 5.2 Unité d'analyse

Une ligne = un sujet × une journée. Ce choix multiplie les observations et
produit une courbe de score jour par jour, indispensable à un outil de suivi.

**Il implique une contrainte de validation absolue** : les journées d'un même
sujet ne sont pas indépendantes. Tous les protocoles regroupent par sujet
(section 7).

### 5.3 Journées calendaires

Les journées vont de minuit à minuit, donc **une nuit de sommeil est coupée en
deux**. Conséquence assumée : aucune estimation de durée de sommeil continue
n'est produite. Les features de rythme retenues sont insensibles à cette coupure
(activité nocturne 00h–06h, L5, ratio nuit/jour).

L'alternative — des fenêtres de midi à midi — préservait les nuits mais
réduisait l'échantillon à 11 journées par sujet. Le signal exploitable dans ces
données étant le niveau et la fragmentation de l'activité, et non la
microstructure du sommeil, le gain ne justifiait pas le coût.

### 5.4 Douze journées fixes par sujet

Seules les **12 premières journées valides** de chaque sujet sont conservées.

La colonne `days` des fichiers de métadonnées ne correspond pas au nombre de
journées réellement enregistrées : elle est donc conservée sous le nom explicite
`days_declared`. Certains témoins comportent plus de 40 journées complètes là où
la médiane se situe autour de 14. Sans troncature, une poignée de sujets très
enregistrés pèserait davantage que les autres et le modèle apprendrait leurs
habitudes individuelles.

**Ce n'est pas une précaution théorique** : la variante sans troncature fait
chuter l'AUC du duel de 0,855 à 0,671 (section 9.2). Avec troncature, chaque
sujet pèse exactement le même poids.

Douze est le maximum possible, imposé par le sujet le moins enregistré.

### 5.5 Validité d'une journée

Une journée est retenue si elle compte **exactement 1440 minutes** et si son
signal peut provenir d'un port normal du dispositif. Trois artefacts distincts,
tous observés dans les données réelles, sont écartés :

| Critère | Seuil | Journées écartées |
|---|---|---|
| Dispositif non porté | activité moyenne < 1 compte/minute | **147** |
| Signal de capteur figé | écart-type journalier < 1 | **40** |
| Journée trop peu documentée | < 5 % de minutes actives | **15** |

Total : **202 journées écartées** sur les 1369 journées syntaxiquement
complètes. À titre d'illustration, 86 journées présentent une activité
strictement nulle sur 24 heures — un témoin en compte 20 sur ses 45 journées, la
montre étant restée dans un tiroir. D'autres affichent la même valeur exactement
sur les 1440 minutes (par exemple 3 ou 7 comptes), ce qui correspond à une valeur
de repos du capteur et non à de l'activité humaine.

Ces journées sont **exclues et non corrigées** : imputer une activité inventée
serait plus dommageable que perdre l'observation. Toutes les exclusions sont
journalisées dans `outputs/anomalies.csv`.

Point méthodologique à retenir pour le mémoire : **une journée syntaxiquement
complète peut être cliniquement vide.** La règle des 1440 minutes ne suffit pas.

### 5.6 Journées de bord et changements d'heure

Les journées incomplètes en début ou fin d'enregistrement sont un effet de la
pose et du retrait de la montre : 154 sont écartées sans commentaire. Les
journées incomplètes en milieu d'enregistrement sont journalisées comme
anomalies — 11 au total, dont les deux changements d'heure de 2003 : le 30 mars
produit des journées de 1380 minutes (heure manquante), le 26 octobre des
journées de 1500 minutes avec horodatages dupliqués (heure enregistrée deux
fois).

Les doublons d'horodatage ne sont donc pas traités comme une erreur fatale : la
règle de complétude écarte ces journées, et le contrôle strict d'unicité est
**réappliqué sur les seules journées retenues**.

### 5.7 Validation du schéma

Toute anomalie de schéma lève une exception au lieu d'être corrigée
silencieusement : sur des données cliniques, une correction implicite est un
biais invisible. Contrôles : noms de colonnes, unicité des horodatages, absence
de valeurs manquantes, positivité de l'activité, cohérence entre la colonne
`date` et l'horodatage, effectifs attendus par classe.

### 5.8 Variables exclues du modèle

Le **sexe, l'âge, la sévérité et le statut de prise en charge** sont conservés
comme métadonnées mais n'entrent jamais dans l'apprentissage. Deux raisons : un
outil qui exige ces informations est moins déployable qu'un outil branché sur un
bracelet, et le déséquilibre de sexe entre les groupes ferait courir le risque
que le modèle apprenne le sexe plutôt que la pathologie.

La liste `FEATURE_COLUMNS` de `src/features.py` est la source de vérité : toute
colonne absente de cette liste est une métadonnée.

---

## 6. Les features

Chaque journée de 1440 minutes est résumée par 19 variables. Le jeu retenu par le
modèle final en compte 7, marquées **★**.

### Niveau d'activité

| Feature | Définition |
|---|---|
| `activity_mean` **★** | Moyenne des comptes sur les 1440 minutes |
| `activity_median` | Médiane, moins sensible aux pics |
| `activity_p95` | 95e percentile : niveau des périodes actives |
| `activity_max` | Compte maximal de la journée |

### Variabilité

| Feature | Définition |
|---|---|
| `activity_std` | Écart-type des comptes |
| `activity_cv` **★** | Coefficient de variation : écart-type rapporté à la moyenne |
| `activity_iqr` | Écart interquartile |

### Sédentarité et fragmentation

| Feature | Définition |
|---|---|
| `pct_zero_minutes` **★** | Pourcentage de minutes à activité nulle |
| `longest_zero_run` | Durée de la plus longue plage continue d'inactivité |
| `n_active_bouts` **★** | Nombre de plages d'activité distinctes, séparées par au moins une minute nulle |

### Rythme jour / nuit

| Feature | Définition |
|---|---|
| `night_activity_mean` | Activité moyenne 00h00–05h59 |
| `day_activity_mean` | Activité moyenne 08h00–19h59 |
| `night_day_ratio` **★** | Rapport activité nocturne / diurne |
| `activity_center_hour` | Barycentre temporel de l'activité : proxy de phase |

### Indices circadiens non paramétriques

Calculés sur les 24 moyennes horaires, avec fenêtres **circulaires** afin qu'une
période de repos chevauchant minuit ne soit pas coupée.

| Feature | Définition |
|---|---|
| `m10` | Activité moyenne des 10 heures consécutives les plus actives |
| `l5` | Activité moyenne des 5 heures consécutives les moins actives |
| `relative_amplitude` **★** | `(m10 − l5) / (m10 + l5)` : contraste activité / repos |
| `intradaily_variability` | Fragmentation du rythme sur 24 h |
| `l5_onset_hour` **★** | Heure de début de la période L5 : proxy d'horaire de repos |

### Variables volontairement écartées

- **Stabilité inter-journalière (IS)** : multi-journalière par définition, elle
  compare une journée au profil moyen du sujet. L'intégrer à une ligne
  « journée » serait un contresens et introduirait une empreinte du sujet dans
  chaque observation. Calculée au niveau du sujet dans `describe.py`, pour
  l'analyse descriptive seule.
- **Rapport M10 / L5** : transformation monotone de `relative_amplitude`, donc
  strictement redondant, et non borné si L5 vaut zéro.

### Contrôle d'intégrité

`features.py` refuse d'écrire un fichier contenant une valeur non finie, un
nombre de sujets inattendu ou un nombre de journées non homogène. Ce contrôle
unique en sortie remplace des garde-fous dispersés dans chaque calcul : si une
division dégénère, elle est signalée plutôt que masquée par une valeur de repli
arbitraire. C'est ce contrôle qui a révélé les journées de signal figé.

### Multicolinéarité

**Douze paires de features dépassent |r| = 0,90**, toutes dans la famille
« niveau d'activité » (jusqu'à r = 0,968 entre `activity_p95` et
`activity_std`). Sur une régression, des variables aussi corrélées se partagent
le poids de façon instable et leurs coefficients individuels deviennent
ininterprétables. D'où le jeu réduit à 7 variables, un représentant par famille
clinique.

---

## 7. Protocole de validation

Deux protocoles complémentaires, **tous deux groupés par sujet**.

**Leave-one-subject-out** — métriques principales. 77 plis ; chaque sujet est
évalué par un modèle entraîné sur les 76 autres. Les métriques ne peuvent pas
être calculées pli par pli (un pli ne contient qu'une classe) : les probabilités
hors échantillon sont donc rassemblées avant tout calcul.

**StratifiedGroupKFold répété** — dispersion. 5 plis × 10 répétitions, soit 50
estimations, donc des intervalles de confiance. Sur 77 sujets, cette mesure n'est
pas une coquetterie : elle détermine quels écarts sont concluants.

**Standardisation ajustée dans chaque pli d'entraînement**, via un pipeline
scikit-learn. La calculer une fois sur l'ensemble du jeu ferait fuiter la
distribution du test.

**Seuils choisis sur les données d'entraînement**, jamais sur le test.

---

## 8. Résultats

### 8.1 Le résultat central

Leave-one-subject-out, 77 sujets, 924 journées :

| Modèle | Variables | Exact. jour | Macro F1 | AUC macro | **Duel dép./schizo.** | Exact. sujet |
|---|---|---|---|---|---|---|
| Logistique | baseline 1 var. | 0,528 | 0,473 | 0,662 | **0,476** | 0,545 |
| Logistique | **réduit 7 var.** | 0,644 | 0,632 | 0,792 | **0,855** | **0,753** |
| Logistique | complet 19 var. | 0,654 | 0,638 | 0,818 | 0,847 | 0,753 |
| Random Forest | réduit 7 var. | 0,604 | 0,588 | 0,774 | 0,786 | 0,688 |
| Random Forest | complet 19 var. | 0,600 | 0,578 | 0,779 | 0,756 | 0,675 |

**AUC du duel : 0,476 pour la règle à une seule variable, 0,855 pour le modèle
multivarié. Gain de +0,379.**

Le 0,476 est le chiffre décisif : il est **en dessous du hasard**. La règle simple
n'échoue pas par manque de réglage, elle échoue exactement comme la théorie le
prévoit — les deux groupes de patients étant hypoactifs, un seuil sur l'activité
moyenne les mélange au lieu de les départager.

### 8.2 Dispersion

5 plis × 10 répétitions :

| Modèle | Variables | Macro F1 | Écart-type | AUC duel | Intervalle 95 % |
|---|---|---|---|---|---|
| Logistique | baseline 1 var. | 0,496 | 0,067 | 0,571 | [0,296 ; 0,837] |
| Logistique | réduit 7 var. | 0,641 | 0,066 | 0,863 | [0,690 ; 0,969] |
| Logistique | complet 19 var. | 0,646 | 0,056 | 0,847 | [0,649 ; 0,962] |
| Random Forest | réduit 7 var. | 0,593 | 0,071 | 0,792 | [0,616 ; 0,932] |
| Random Forest | complet 19 var. | 0,581 | 0,075 | 0,774 | [0,598 ; 0,917] |

Les intervalles sont larges : **aucun écart inférieur à environ 5 points n'est
concluant sur 77 sujets.** Les jeux à 7 et 19 variables sont donc équivalents ;
le jeu réduit est retenu pour son interprétabilité, non pour sa performance.

### 8.3 Matrices de confusion au niveau sujet

Modèle retenu — 58 sujets sur 77 correctement classés :

| Réel \ Prédit | Témoin | Dépression | Schizophrénie |
|---|---|---|---|
| Témoin | **27** | 4 | 1 |
| Dépression | 8 | **13** | 2 |
| Schizophrénie | 2 | 2 | **18** |

Règle à une seule variable — 42 sur 77 :

| Réel \ Prédit | Témoin | Dépression | Schizophrénie |
|---|---|---|---|
| Témoin | **27** | 3 | 2 |
| Dépression | 10 | **2** | 11 |
| Schizophrénie | 5 | 4 | **13** |

La comparaison est éloquente : les deux modèles identifient les témoins aussi
bien (27 sur 32). C'est sur les patients que tout se joue — la règle simple ne
reconnaît que **2 patients déprimés sur 23** et en classe 11 comme schizophrènes.

### 8.4 Ce que le modèle a appris

Coefficients standardisés, régression logistique multinomiale sur le jeu réduit :

| Feature | Témoin | Dépression | Schizophrénie |
|---|---|---|---|
| `activity_mean` | +1,095 | +0,356 | **−1,451** |
| `activity_cv` | +0,567 | +0,827 | −1,394 |
| `n_active_bouts` | +0,201 | **+0,635** | −0,836 |
| `pct_zero_minutes` | −0,623 | −0,718 | **+1,341** |
| `night_day_ratio` | +0,317 | +0,247 | −0,563 |
| `relative_amplitude` | +0,275 | +0,304 | −0,579 |
| `l5_onset_hour` | −0,211 | +0,118 | +0,094 |

Lecture clinique : l'activité moyenne décroît régulièrement du témoin vers la
schizophrénie, mais surtout les deux pathologies se séparent **en directions
opposées** sur la fragmentation. Le patient déprimé fragmente son activité en de
nombreux épisodes courts (`n_active_bouts` +0,635) ; le schizophrène présente au
contraire de longues plages d'inactivité continue (`pct_zero_minutes` +1,341).

Ce sont deux profils moteurs qualitativement distincts, et non deux degrés d'un
même ralentissement. C'est précisément ce qu'un seuil unique ne peut pas
représenter.

### 8.5 Validation externe

Ces résultats convergent avec la littérature clinique produite par la même
équipe : *Actigraphic registration of motor activity reveals a more structured
behavioural pattern in schizophrenia than in major depression* (Haukeland).
L'article rapporte une activité motrice réduite chez les patients schizophrènes
**et** déprimés par rapport aux témoins, ainsi qu'un pattern comportemental plus
structuré dans la schizophrénie que dans la dépression majeure.

Le pipeline retrouve indépendamment les deux observations : l'hypoactivité
commune (qui explique l'échec du seuil unique) et la structure plus régulière
dans la schizophrénie. La stabilité inter-journalière mesurée le confirme :

| Classe | IS moyenne |
|---|---|
| Schizophrénie | **0,521 ± 0,148** |
| Témoin | 0,437 ± 0,113 |
| Dépression | 0,418 ± 0,118 |

### 8.6 Le pouvoir discriminant dépend de la tâche

AUC univariée de chaque feature, duel par duel (extrait) :

| Feature | Dép./tém. | Schizo./tém. | **Schizo./dép.** |
|---|---|---|---|
| `longest_zero_run` | **0,514** | 0,774 | **0,742** |
| `n_active_bouts` | 0,755 | 0,599 | 0,675 |
| `activity_p95` | 0,778 | **0,904** | 0,626 |
| `activity_mean` | 0,779 | 0,866 | 0,572 |
| `activity_median` | **0,780** | 0,799 | **0,537** |

`longest_zero_run` était **inutile** sur la tâche binaire — 0,514, soit le
hasard. Sur le duel entre pathologies, elle devient la **meilleure feature de
toutes**. Symétriquement, `activity_median`, championne du binaire, s'effondre à
0,537.

Conclusion méthodologique : **le pouvoir discriminant n'est pas une propriété de
la variable, mais du couple variable-tâche.** Une variable qu'un tri automatique
aurait éliminée sur la première tâche est celle qui résout la seconde. Argument
direct contre la sélection de variables aveugle.

Son ajout au jeu retenu a été testé : AUC du duel 0,852 contre 0,855, exactitude
sujet 0,727 contre 0,753. Aucun gain — l'information est déjà portée par
`pct_zero_minutes`. La variable n'est donc pas retenue, mais le test illustre la
différence entre **pertinence marginale et pertinence conditionnelle**.

### 8.7 La phase binaire, à conserver dans le mémoire

Sur la tâche patient contre témoin (55 sujets, 660 journées), toutes les
approches convergent :

| Modèle | Variables | AUC jour |
|---|---|---|
| Logistique | baseline 1 var. | 0,755 |
| Logistique | réduit 7 var. | 0,704 |
| Logistique | complet 19 var. | **0,758** |
| Logistique réglée (C par CV imbriquée) | complet 19 var. | 0,743 |
| Random Forest | complet 19 var. | 0,744 |

Gain maximal du multivarié : **+0,003**, pour un écart-type inter-plis de 0,09.

Deux enseignements. Le réglage de `C` par validation imbriquée **dégrade** le
résultat : sur 55 sujets, la recherche d'hyperparamètre ajoute plus de variance
qu'elle n'apporte de gain. Et le Random Forest, censé capter des interactions, ne
dépasse pas la régression linéaire : **il n'y a pas d'interaction à capter**, le
signal est bien unidimensionnel.

Ce résultat n'est pas un échec à masquer. Il démontre, mesures en main, l'écart
entre performance affichée et valeur ajoutée réelle — et il donne tout son sens
au +0,379 obtenu sur la tâche à trois classes.

---

## 9. Robustesse et sensibilité

### 9.1 Sous-groupes de population

| Analyse | Sujets | Journées | Duel baseline | Duel modèle | Gain |
|---|---|---|---|---|---|
| Référence | 77 | 924 | 0,476 | 0,855 | +0,379 |
| **Hommes uniquement** | 44 | 528 | 0,391 | **0,857** | **+0,466** |
| Femmes uniquement | 33 | 396 | 0,431 | 0,784 | +0,353 |
| Dépression unipolaire | 69 | 828 | 0,511 | 0,835 | +0,324 |

Le confondant du sexe est levé. À sexe constant, le duel passe de 0,855 à
**0,857** — aucune dégradation — et le gain face à la baseline *augmente*. **La
séparation entre les deux pathologies n'est donc pas du sexe déguisé en
pathologie**, malgré un groupe schizophrène masculin à 86 %. Le résultat tient
aussi chez les femmes (0,784, avec seulement 3 patientes schizophrènes) et sur
les unipolaires seuls.

### 9.2 Sans troncature à 12 journées

| | Sujets | Journées | Duel baseline | Duel modèle | Macro F1 |
|---|---|---|---|---|---|
| Toutes journées valides | 77 | 1167 | 0,460 | **0,671** | 0,522 |

Chute de 0,855 à 0,671. Ce n'est pas un défaut du modèle mais la validation
chiffrée de la décision 5.4 : sans troncature, quelques sujets très enregistrés
pèsent plusieurs fois plus que les autres et le modèle apprend leurs habitudes.

### 9.3 Démonstration de la fuite de données

| Protocole | Estimations | AUC duel | Écart-type | Intervalle 95 % |
|---|---|---|---|---|
| Découpage aléatoire (incorrect) | 50 | **0,895** | **0,027** | [0,837 ; 0,947] |
| Découpage par sujet (correct) | 50 | 0,855 | 0,083 | [0,689 ; 0,986] |

Surestimation : **+0,040 d'AUC**. Mais l'enseignement principal est ailleurs :
l'écart-type passe de 0,083 à 0,027. Le protocole incorrect ne gonfle pas
seulement la performance, il **masque surtout l'incertitude réelle** — il fait
croire à une précision que l'échantillon ne permet pas.

### 9.4 Ampleur du confondant d'hospitalisation

Comparaison à l'intérieur de la classe dépression (5 hospitalisés contre 18
ambulatoires) :

| Feature | d hospitalisé vs ambulatoire | d schizophrénie vs ambulatoire |
|---|---|---|
| `pct_zero_minutes` | **+0,87** | +0,54 |
| `activity_cv` | +0,55 | −0,25 |
| `activity_mean` | −0,48 | −0,44 |
| `night_day_ratio` | −0,48 | −0,50 |
| `n_active_bouts` | +0,26 | −0,40 |

**C'est la limite la plus sérieuse du cas concret.** L'hospitalisation déplace le
profil moteur d'une ampleur comparable à celle de la pathologie elle-même. Les 5
sujets concernés interdisent toute conclusion ferme, mais l'indication est nette.
Voir section 10.

### 9.5 Seuil du duel selon le coût clinique

| Coût relatif | Seuil | Schizo. manquées | Schizo. à tort | Sensibilité |
|---|---|---|---|---|
| 0,5 | 0,655 | 85 | 28 | 0,678 |
| 1,0 | 0,469 | 52 | 58 | 0,803 |
| 2,0 | 0,421 | 43 | 74 | 0,837 |
| 4,0 | 0,224 | 20 | 136 | 0,924 |

Passer de 80 % à 92 % de sensibilité coûte **plus du double de faux positifs**.
Le seuil n'est pas un paramètre technique mais une **décision clinique**, qui
doit être arbitrée par l'équipe soignante en fonction de sa capacité de réponse.

---

## 10. Limites

À reprendre dans l'analyse critique du mémoire.

**Confondant d'hospitalisation — limite principale.** Les patients schizophrènes
de PSYKOSE étaient hospitalisés, tandis que 18 des 23 patients déprimés étaient
suivis en ambulatoire. Le statut n'est pas documenté dans les métadonnées de
PSYKOSE : il a fallu le retrouver dans la littérature. Or l'hospitalisation
modifie l'activité motrice d'une ampleur comparable à la pathologie (section
9.4). **Une part du 0,855 reflète donc l'environnement de vie et non la
maladie.** Cette limite ne peut pas être éliminée avec ces données ; elle a été
mesurée et doit être énoncée.

**Taille d'échantillon.** L'unité statistique réelle est le sujet, non la
journée : l'effectif est donc de 77, pas de 924. C'est ce nombre qui plafonne la
généralisation et explique des intervalles de confiance de l'ordre de 0,2 point
d'AUC.

**Déséquilibre démographique.** Sexe : 86 % d'hommes chez les schizophrènes,
57 % chez les déprimés, 37 % chez les témoins — testé et levé (section 9.1).
Âge : le groupe schizophrène est plus âgé, sans test de sensibilité dédié.

**Absence de formes légères.** MADRS de 13 à 29 : tous les patients déprimés sont
en épisode modéré à sévère. L'outil n'a pas été évalué sur les formes frustes,
qui sont pourtant les plus difficiles à repérer en clinique.

**Monocentrique et daté.** Un seul centre, enregistrements de 2002 à 2006, un
seul modèle d'actimètre. La transposition à un bracelet grand public actuel n'est
pas acquise.

**Détection d'état, non prédiction.** L'outil caractérise une journée observée.
Il ne prédit ni rechute ni évolution : ces données ne contiennent aucun événement
longitudinal permettant de l'évaluer.

**Pas de dispositif médical.** Aucune validation clinique prospective, aucune
évaluation de conformité au sens du règlement européen sur l'intelligence
artificielle, alors qu'un tel usage relèverait du haut risque.

---

## 11. Restitution

Le tableau de bord (`app.py`) est en **lecture seule** : il n'entraîne aucun
modèle et affiche les artefacts produits par `export.py`.

**Les scores présentés sont hors échantillon** : chaque journée est notée par un
modèle qui n'a jamais vu le sujet concerné. Les contributions de variables
suivent la même règle, calculées avec les coefficients du pli où le sujet était
en test. Un tableau de bord affichant les scores d'un modèle entraîné sur le
patient consulté donnerait une impression de fiabilité trompeuse.

Trois vues :

- **Suivi individuel** — sélection du sujet et de la journée, courbe d'évolution
  des trois probabilités avec seuil réglable, actogramme avec journée
  sélectionnée mise en évidence, contributions des variables en libellés
  cliniques. Un avertissement signale les désaccords entre classe prédite et
  classe documentée, en précisant qu'un désaccord isolé n'a pas de valeur
  diagnostique.
- **Évaluation du modèle** — matrices de confusion, courbe ROC du duel avec
  position du seuil, coefficients standardisés.
- **Méthode et limites** — méthode, protocole de validation et l'ensemble des
  limites de la section 10.

Un bandeau rappelle en tête de chaque vue que l'outil relève de l'aide à la
décision sous supervision médicale et n'est pas un dispositif médical.

---

## 12. Artefacts générés

```
outputs/
  subjects.csv                        référentiel sujet harmonisé
  anomalies.csv                       journal des exclusions
  features.csv                        924 journées × 19 features
  metrics.json                        métriques de toutes les combinaisons
  tables/
    comparaison_features.csv          moyennes par classe, effets, AUC par duel
    correlations_fortes.csv           paires |r| >= 0,90
    stabilite_interjournaliere.csv    IS par sujet
    coefficients_multiclasse.csv      coefficients par classe
    predictions_meilleur_modele.csv   probabilités hors échantillon
    robustesse_*.csv                  analyses de la section 9
  figures/
    fig1_correlations.png             matrice de corrélation
    fig2_boxplots.png                 distributions par classe
    fig3_profil_horaire.png           activité sur 24 h par classe
    fig4_actogrammes.png              un actogramme par classe
    fig5_discrimination.png           AUC par feature et par duel
  dashboard/                          artefacts du tableau de bord
```

Figures en niveaux de gris, lisibles à l'impression.

---

## 13. Références

- Garcia-Ceja E. et al., *Depresjon: A Motor Activity Database of Depression
  Episodes in Unipolar and Bipolar Patients*, ACM MMSys, 2018.
  DOI 10.5281/zenodo.1219550
- Jakobsen P. et al., *PSYKOSE: A Motor Activity Database of Patients with
  Schizophrenia*, https://datasets.simula.no/psykose/
- *Actigraphic registration of motor activity reveals a more structured
  behavioural pattern in schizophrenia than in major depression*, hôpital
  universitaire de Haukeland.