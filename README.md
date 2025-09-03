# Machine Learning avec Scikit-learn 🧠

*Parce que non, le ML ce n'est pas de la magie noire !*

Salut ! 👋 Tu trouveras ici mon parcours complet d'apprentissage du machine learning avec scikit-learn. J'ai essayé de rendre ça aussi accessible que possible - pas de jargon compliqué, juste du code qui marche et des explications qui ont du sens.

Après avoir galéré avec des tutos confus, j'ai décidé de créer le cours que j'aurais aimé avoir au début. Spoiler alert: c'est bien plus simple qu'on le dit !

## Le parcours (ou comment j'ai survécu au ML)

### 🚀 Module 1 : Les bases (sans se prendre la tête)
*Dans `ml_report_Module1/ml_report_Module1.md`*

On commence tranquille avec :
- Comment ça marche le ML ? (spoiler: c'est juste des maths avec du code)
- Nettoyer ses données (parce que c'est toujours le bordel au début)
- Premier algorithme avec K-NN (facile, même mon chat pourrait le comprendre)
- Les pipelines pour pas se perdre dans son code
- La validation croisée (pour éviter de se mentir sur ses résultats)
- Gérer les données catégorielles sans s'arracher les cheveux

### 📊 Module 2 : Quand ça marche pas comme prévu
*Dans `ml_report_Module2/ml_report_module2.md`*

Parce que oui, parfois ça plante :
- Overfitting vs underfitting (ou comment ton modèle peut être trop malin ou trop bête)
- Les courbes qui te disent si ton modèle va bien
- Support Vector Machines pour les cas compliqués

### ⚙️ Module 3 : Optimiser sans devenir fou
*Dans `ml_report_Module3/ml_report_module3.md`*

L'art de trouver les bons réglages :
- Les hyperparamètres (ces trucs qu'il faut ajuster à la main)
- Grid Search vs Random Search (bataille épique)
- Comment pas passer 3 jours à tester des paramètres

### 📐 Module 4 : Les modèles linéaires qui font le taff
*Dans `ml_report_Module4/ml_report_module4.md`*

Simple mais efficace :
- Régression linéaire from scratch (pour comprendre)
- Classification linéaire et régularisation
- Quand utiliser du linéaire (plus souvent qu'on pense)

### 🌳 Module 5 : Les arbres de décision
*Dans `ml_report_Module5/ml_report_module5.md`*

Parce que parfois il faut des décisions claires :
- Comment un arbre "réfléchit"
- Classification et régression avec les arbres
- Les hyperparamètres qui comptent vraiment

### 🌲 Module 6 : Les forêts et le boosting
*Dans `ml_report_Module6/ml_report_module6.md`*

Quand un arbre ne suffit pas :
- Random Forest (la force du nombre)
- Gradient Boosting (la méthode qui gagne souvent)
- Importance des features (enfin savoir ce qui compte)

### 🎯 Module 7 : Mesurer ses résultats comme un pro
*Dans `ml_report_Module7/ml_report_module7.md`*

Comment savoir si c'est vraiment bien :
- Stratégies de validation croisée avancées
- Métriques qui ont du sens
- Gérer les données temporelles et les groupes

## Pourquoi ce cours ?

J'en avais marre des tutoriels qui te balancent 50 lignes de code sans expliquer, ou des cours académiques impossibles à suivre. Ici, on prend le temps de comprendre ce qu'on fait et pourquoi on le fait.

Au final, tu vas apprendre à :
- Construire des pipelines ML qui marchent vraiment
- Choisir le bon algorithme pour le bon problème
- Débugger quand ça marche pas (et ça arrivera)
- Évaluer tes modèles sans te mentir

## 🛠️ Ce dont tu auras besoin

- **Python** (évidemment)
- **Scikit-learn** (la star du show)
- **Pandas** et **NumPy** (pour les données)
- **Matplotlib** (pour les jolis graphiques)
- Un peu de patience (c'est normal de pas tout comprendre du premier coup)

## 📁 Comment c'est organisé

```
Mooc/
├── README.md                          # Tu es ici !
├── RAPPORT_COMPLET_ML.md             # Résumé de tout le parcours
├── images/                           # Tous les schémas et graphiques
│   ├── kfold.png                     # Validation croisée expliquée
│   ├── colkumnTransformer.png        # Preprocessing visualisé
│   └── [autres schémas utiles]      
├── ml_report_Module1/                # Les bases du ML
├── ml_report_Module2/                # Validation et diagnostic
├── ml_report_Module3/                # Optimisation des hyperparamètres
├── ml_report_Module4/                # Modèles linéaires
├── ml_report_Module5/                # Arbres de décision
├── ml_report_Module6/                # Méthodes d'ensemble
└── ml_report_Module7/                # Validation avancée et métriques
```

## 🚀 Comment s'y prendre

1. **Start with Module 1** - même si tu penses déjà savoir
2. **Code en même temps** - la théorie c'est bien, la pratique c'est mieux
3. **N'hésite pas à revenir en arrière** - c'est normal de pas tout capter du premier coup
4. **Teste sur tes propres données** - c'est là que ça devient vraiment utile

## 🧰 Ce que tu vas maîtriser

**Algorithmes qu'on va dompter :**
- K-Nearest Neighbors (pour commencer en douceur)
- Logistic Regression (le classique qui marche)
- Decision Trees (facile à expliquer au chef)
- Random Forest et Gradient Boosting (les gros calibres)
- Support Vector Machine (pour frimer un peu)

**Techniques de preprocessing :**
- StandardScaler (parce que les algos aiment quand c'est normalisé)
- Encodage des variables catégorielles (sans se tromper)
- Pipelines (pour pas perdre le fil)
- ColumnTransformer (le couteau suisse du preprocessing)

**Validation qui tient la route :**
- Train-Test Split (la base)
- K-Fold Cross-Validation (pour être sûr de son coup)
- Courbes de validation (pour voir ce qui se passe)
- Grid Search et Random Search (pour les perfectionnistes)

## 📊 Pourquoi il y a plein d'images

Parce que comprendre avec des schémas, c'est plus facile ! Tu trouveras des visualisations pour :
- Comment les données se transforment
- Pourquoi la validation croisée c'est important  
- Comment les algorithmes prennent leurs décisions
- Quand un modèle dérive (overfitting, tout ça...)

## 🎯 C'est pour qui ?

- **Les débutants** qui veulent comprendre sans se perdre
- **Les devs** qui ont entendu parler de ML mais savent pas par où commencer
- **Les curieux** qui en ont marre des explications incompréhensibles
- **Les pragmatiques** qui veulent du concret, pas de la théorie pure

## 📝 Il faut savoir quoi avant ?

- **Python de base** (variables, fonctions, listes... rien de fou)
- **Un peu de stats** (c'est mieux mais pas obligatoire)
- **Pandas/NumPy** (utile mais on revoit les bases)
- **De la motivation** (le plus important !)

## 🎉 Bonus : le rapport complet

Tu trouveras aussi `RAPPORT_COMPLET_ML.md` qui résume tout le parcours. Pratique pour réviser ou retrouver un concept rapidement.

---

*P.S. : Ce cours, c'est du vécu. J'ai fait toutes les erreurs possibles pour que tu puisses les éviter !* 