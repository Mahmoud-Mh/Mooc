# Cours de Machine Learning avec Scikit-learn

## Description du projet

Ce repository contient un cours complet d'introduction au Machine Learning utilisant la bibliothèque Python **scikit-learn**. Le cours est structuré en modules progressifs qui couvrent les concepts fondamentaux et les techniques avancées de l'apprentissage automatique.

## Structure du cours

### 📚 Module 1 : Introduction au Machine Learning avec Scikit-learn
**Fichier :** `ml_report_Module1/ml_report_Module1.md`

Ce premier module couvre les concepts fondamentaux :

- **Introduction générale au Machine Learning**
  - Concepts de features, target, et dataset
  - Différence entre classification et régression
  
- **Exploration et préparation des données**
  - Analyse des types de données (numériques vs catégorielles)
  - Techniques de preprocessing et normalisation
  
- **Premiers modèles avec K-Nearest Neighbors (KNN)**
  - Compréhension de l'algorithme KNN
  - API scikit-learn et workflow de base
  
- **Préprocessing avancé**
  - StandardScaler pour la normalisation
  - Pipelines pour automatiser les étapes
  
- **Validation croisée (Cross-validation)**
  - K-Fold cross-validation
  - Estimation robuste des performances
  
- **Gestion des données catégorielles**
  - Ordinal Encoding vs One-Hot Encoding
  - Choix d'encodage selon le type de modèle
  
- **Combinaison de types de données**
  - ColumnTransformer pour données mixtes
  - Pipelines complets avec preprocessing
  
- **Modèles avancés**
  - Introduction au Gradient Boosting
  - Comparaison entre différents algorithmes

### 📈 Module 2 : Overfitting, Underfitting et Validation
**Fichier :** `ml_report_Module2/ml_report_module2.md`

Le second module approfondit les concepts de validation et d'optimisation :

- **Concepts d'Overfitting vs Underfitting**
  - Analyse des erreurs d'entraînement vs test
  - Interprétation des performances de généralisation
  
- **Courbes de validation**
  - Effet des hyperparamètres sur les performances
  - Optimisation des paramètres de modèle
  
- **Courbes d'apprentissage**
  - Impact de la taille de l'échantillon
  - Détermination du besoin en données supplémentaires
  
- **Exercice pratique avec SVM**
  - Support Vector Machine avec validation croisée
  - Analyse du paramètre gamma
  
- **Bonnes pratiques**
  - Stratégies pour éviter le surapprentissage
  - Workflow d'analyse et d'optimisation

## 🎯 Objectifs pédagogiques

Ce cours vise à fournir une compréhension complète et pratique du Machine Learning :

1. **Compréhension théorique** des concepts fondamentaux
2. **Maîtrise pratique** de scikit-learn et de son API
3. **Développement d'une méthodologie** rigoureuse pour les projets ML
4. **Capacité d'analyse** des performances et de diagnostic des modèles

## 🛠️ Technologies utilisées

- **Python** : Langage de programmation principal
- **Scikit-learn** : Bibliothèque de Machine Learning
- **Pandas** : Manipulation et analyse de données
- **Matplotlib** : Visualisation de données
- **NumPy** : Calculs numériques

## 📁 Organisation du projet

```
Mooc/
├── README.md                                    # Documentation principale
├── images/                                      # Toutes les images des cours
│   ├── Culmen Depth.png
│   ├── colkumnTransformer.png
│   ├── kfold.png
│   ├── pipeline_fit_data.png
│   ├── pipeline_predictdata.png
│   ├── transformer_fitdata.png
│   ├── transformer_transformer_data.png
│   ├── ValidationCurveDisplay.png
│   └── numberOfSamplesInTrainingSet.png
├── ml_report_Module1/                           # Module 1 - Bases du ML
│   └── ml_report_Module1.md
└── ml_report_Module2/                           # Module 2 - Validation avancée
    └── ml_report_module2.md
```

## 📖 Comment utiliser ce cours

1. **Commencez par le Module 1** pour acquérir les bases
2. **Pratiquez** avec les exemples de code fournis
3. **Passez au Module 2** pour approfondir les concepts avancés
4. **Appliquez** les techniques apprises sur vos propres datasets

## 🔍 Contenu détaillé

### Algorithmes couverts
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Trees
- Gradient Boosting
- Support Vector Machine (SVM)

### Techniques de preprocessing
- Normalisation avec StandardScaler
- Encodage des variables catégorielles
- Pipelines automatisés
- ColumnTransformer

### Méthodes de validation
- Train-Test Split
- K-Fold Cross-Validation
- Courbes de validation
- Courbes d'apprentissage

## 📊 Visualisations incluses

Le cours comprend de nombreuses visualisations et diagrammes pour illustrer :
- Distributions de données
- Processus de transformation
- Courbes de performance
- Comparaisons de modèles

## 🎓 Public cible

Ce cours s'adresse à :
- **Débutants** en Machine Learning
- **Développeurs** souhaitant apprendre scikit-learn
- **Étudiants** en data science
- **Professionnels** voulant acquérir des bases solides en ML

## 📝 Prérequis

- Connaissances de base en **Python**
- Notions de **statistiques** (recommandé)
- Familiarité avec **Pandas** et **NumPy** (utile)

## 🚀 Pour aller plus loin

Après avoir terminé ce cours, vous serez capable de :
- Construire des pipelines ML complets
- Choisir et optimiser des modèles appropriés
- Évaluer et diagnostiquer les performances
- Appliquer les bonnes pratiques du domaine

---

*Ce cours fait partie d'un MOOC (Massive Open Online Course) sur le Machine Learning avec Python et scikit-learn.* 