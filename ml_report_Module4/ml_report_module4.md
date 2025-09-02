# Module 4 : Mod�les Lin�aires - Rapport Complet

## Table des Mati�res
1. [Introduction aux Mod�les Lin�aires](#introduction-aux-mod�les-lin�aires)
2. [R�gression Lin�aire sans Scikit-learn](#r�gression-lin�aire-sans-scikit-learn)
3. [Mod�les Lin�aires pour la Classification](#mod�les-lin�aires-pour-la-classification)
4. [Ing�nierie de Caract�ristiques Non-lin�aires](#ing�nierie-de-caract�ristiques-non-lin�aires)
5. [Techniques de R�gularisation](#techniques-de-r�gularisation)
6. [M�thodes d'Approximation de Noyaux](#m�thodes-dapproximation-de-noyaux)
7. [Exercices Pratiques et Applications](#exercices-pratiques-et-applications)
8. [�valuations et Quiz](#�valuations-et-quiz)

## Introduction aux Mod�les Lin�aires

Les mod�les lin�aires constituent une famille fondamentale d'algorithmes d'apprentissage automatique qui �tablissent des relations lin�aires entre les caract�ristiques d'entr�e et la variable cible. Ce module explore en profondeur l'impl�mentation, l'utilisation et l'optimisation de ces mod�les dans le contexte de scikit-learn.

### Concepts Fondamentaux

Les mod�les lin�aires s'appuient sur l'hypoth�se que la relation entre les variables explicatives et la variable � pr�dire peut �tre exprim�e sous forme d'une combinaison lin�aire. Pour la r�gression, cette relation s'exprime comme :

```
y = a * x + b
```

o� `a` repr�sente la pente et `b` l'ordonn�e � l'origine.

## R�gression Lin�aire sans Scikit-learn

### Impl�mentation Manuelle

Le module d�marre par une approche p�dagogique consistant � impl�menter la r�gression lin�aire sans utiliser scikit-learn, permettant de comprendre les m�canismes sous-jacents :

```python
import numpy as np
import matplotlib.pyplot as plt

# Impl�mentation manuelle de la r�gression lin�aire
def fit_linear_regression(x, y):
    a = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    b = y.mean() - a * x.mean()
    return a, b

def predict_linear_regression(x, a, b):
    return a * x + b
```

Cette approche permet de visualiser clairement la param�trisation du mod�le et de comprendre comment les coefficients sont calcul�s.

### Transition vers Scikit-learn

L'introduction progressive de `sklearn.linear_model.LinearRegression` montre les avantages de l'utilisation d'une biblioth�que standardis�e :

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Mod�les Lin�aires pour la Classification

### R�gression Logistique

La r�gression logistique �tend les concepts de la r�gression lin�aire au domaine de la classification binaire. Le module explique en d�tail :

- **Fonction sigmo�de** : Transformation des scores lin�aires en probabilit�s
- **Seuillage de d�cision** : M�canisme de classification bas� sur les probabilit�s
- **Fronti�res de d�cision** : Visualisation des zones de classification

```python
from sklearn.linear_model import LogisticRegression

# Exemple d'utilisation avec le dataset des pingouins
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Visualisation des fronti�res de d�cision
from sklearn.inspection import DecisionBoundaryDisplay
DecisionBoundaryDisplay.from_estimator(logistic_model, X, alpha=0.8)
```

![Graphique des profondeurs de bec](../images/culmendepthadelie.png)

### Applications Pratiques

Le module utilise le dataset des pingouins pour illustrer concr�tement l'application de la r�gression logistique, avec des visualisations d�taill�es des fronti�res de d�cision et de l'interpr�tation des coefficients.

## Ing�nierie de Caract�ristiques Non-lin�aires

### Transformations Polynomiales

L'ing�nierie de caract�ristiques permet d'�tendre les capacit�s des mod�les lin�aires en cr�ant des transformations non-lin�aires des donn�es d'entr�e :

```python
from sklearn.preprocessing import PolynomialFeatures

# Cr�ation de caract�ristiques polynomiales
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

Cette technique permet aux mod�les lin�aires de capturer des relations non-lin�aires complexes dans les donn�es.

### Visualisation des Effets

Le module inclut des visualisations d�taill�es montrant comment les transformations polynomiales modifient l'espace des caract�ristiques et permettent de mieux ajuster les donn�es complexes.

![Composantes principales](../images/n_components.png)

## Techniques de R�gularisation

### Ridge Regression

La r�gularisation Ridge constitue une technique essentielle pour contr�ler la complexit� du mod�le et pr�venir le surajustement :

```python
from sklearn.linear_model import Ridge, RidgeCV

# Ridge avec validation crois�e pour s�lection d'alpha
ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13))
ridge_cv.fit(X_train, y_train)

print(f"Meilleur alpha: {ridge_cv.alpha_}")
```

### Param�tre de R�gularisation Alpha

Le module explore en profondeur l'impact du param�tre `alpha` :

- **Valeurs faibles** : Mod�le proche de la r�gression lin�aire classique
- **Valeurs �lev�es** : Forte r�gularisation, coefficients r�duits
- **S�lection optimale** : Utilisation de la validation crois�e

![Importance des caract�ristiques](../images/feature importance absolute coefficients.png)

### Mise � l'�chelle des Caract�ristiques

La r�gularisation n�cessite une attention particuli�re � la mise � l'�chelle :

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardisation des caract�ristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Alternative avec MinMaxScaler
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
```

## M�thodes d'Approximation de Noyaux

### M�thode de Nystr�m

La m�thode de Nystr�m permet d'approximer efficacement les noyaux non-lin�aires tout en conservant les avantages computationnels des mod�les lin�aires :

```python
from sklearn.kernel_approximation import Nystroem

# Approximation RBF avec Nystr�m
nystroem = Nystroem(kernel='rbf', gamma=0.1, n_components=100)
X_nystroem = nystroem.fit_transform(X)
```

![R�gression logistique Nystr�m](../images/Nystroemlogreg.png)

### Avantages et Applications

- **Efficacit� computationnelle** : R�duction de la complexit� par rapport aux SVM
- **Flexibilit�** : Possibilit� d'utiliser diff�rents noyaux (RBF, polynomial)
- **�volutivit�** : Adaptation aux grands datasets

## Exercices Pratiques et Applications

### Dataset des Pingouins

Le module utilise extensivement le dataset des pingouins d'Ad�lie pour illustrer :

- Classification binaire avec r�gression logistique
- Impact de la s�lection de caract�ristiques
- Visualisation des fronti�res de d�cision
- �valuation des performances

### Dataset de l'Immobilier

Les exercices avanc�s portent sur un dataset immobilier complet, int�grant :

- Preprocessing complexe avec `ColumnTransformer`
- Gestion des variables cat�gorielles et num�riques
- Pipelines de transformation compl�tes
- Optimisation des hyperparam�tres

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Pipeline complet de preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Pipeline complet avec mod�le
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
```

## �valuations et Quiz

### Quiz Interm�diaires

Le module comprend plusieurs quiz �valuant la compr�hension :

1. **Param�trisation des mod�les lin�aires**
2. **Choix des m�triques d'�valuation**
3. **Impact de la r�gularisation**
4. **S�lection des transformations de caract�ristiques**

### Exercices de Synth�se

Les exercices finaux int�grent tous les concepts abord�s :

- Comparaison de diff�rentes approches de r�gularisation
- Analyse de l'impact des transformations non-lin�aires
- Optimisation des pipelines de preprocessing
- �valuation comparative des performances

## Concepts Cl�s et Bonnes Pratiques

### Points Essentiels

1. **Mise � l'�chelle obligatoire** : Les mod�les lin�aires r�gularis�s n�cessitent une standardisation des caract�ristiques
2. **Validation crois�e** : S�lection des hyperparam�tres par validation crois�e syst�matique
3. **Ing�nierie de caract�ristiques** : Extension des capacit�s par transformations appropri�es
4. **Interpr�tabilit�** : Maintien de l'avantage interpr�tatif des mod�les lin�aires

### Recommandations Pratiques

- Commencer par des mod�les simples avant d'ajouter de la complexit�
- Utiliser `RidgeCV` pour la s�lection automatique d'alpha
- Visualiser les fronti�res de d�cision pour comprendre le comportement du mod�le
- �valuer l'impact des transformations sur les performances

## Conclusion

Ce module offre une exploration compl�te des mod�les lin�aires, de leur impl�mentation fondamentale aux techniques avanc�es de r�gularisation et d'extension non-lin�aire. Les concepts pr�sent�s constituent une base solide pour aborder des algorithmes plus complexes tout en conservant les avantages d'interpr�tabilit� et d'efficacit� computationnelle des approches lin�aires.

L'approche p�dagogique progressive, associ�e aux visualisations d�taill�es et aux exercices pratiques, permet une compr�hension approfondie des m�canismes sous-jacents et de leur application dans des contextes r�els d'analyse de donn�es.