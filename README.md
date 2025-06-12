# Dashboard interactif – Délinquance en France (2016–2024)

Ce projet propose une analyse statistique complète des infractions enregistrées par la police et la gendarmerie françaises entre 2016 et 2024, à partir des données officielles disponibles sur data.gouv.fr. Il s’appuie sur une application Streamlit interactive permettant d’explorer les dynamiques territoriales et temporelles de la délinquance à l’échelle départementale et régionale.

👉 [Accéder à l'application Streamlit déployée](https://dashboard-delinquance-france-fnaeh3gqf9axpj7weuykwh.streamlit.app/)

## Objectifs

- Fournir un outil d'exploration et de visualisation rigoureux et pédagogique des données de délinquance.
- Comparer les départements et régions selon des indicateurs normalisés (nombre pour 1000 habitants).
- Identifier les dynamiques temporelles et les ruptures statistiques.
- Segmenter les territoires selon leurs profils multi-indicateurs (ACP + clustering).
- Détecter les départements hors norme à l’aide de modèles non supervisés (Isolation Forest).
- Projeter les tendances pour l’année 2025 via des régressions linéaires locales.

## Données

- **Source** : [data.gouv.fr – Bases statistiques de la délinquance enregistrée par la police et la gendarmerie](https://www.data.gouv.fr/fr/datasets/bases-statistiques-communale-departementale-et-regionale-de-la-delinquance-enregistree-par-la-police-et-la-gendarmerie-nationales/)
- **Période couverte** : 2016 à 2024
- **Granularité** : commune, département, région
- **Indicateurs** : atteintes aux personnes, atteintes aux biens, violences sexuelles, vols, etc.
- **Population de référence** : INSEE (millésime 2024)

## Fonctionnalités de l’application

- **Vue brute** : infractions totales (volume absolu) par département.
- **Indicateurs normalisés** : nombre pour 1000 habitants, comparaisons équitables.
- **Évolution temporelle** : visualisation des tendances par département ou indicateur.
- **Comparaison régionale** : évolution agrégée et heatmaps régionales.
- **Clustering territorial** : classification des départements (ACP + K-means).
- **Détection d’anomalies** : identification automatique des départements hors normes.
- **Prévisions 2025** : projection des infractions par département via régression.

## Technologies

- Python
- Streamlit
- Pandas, NumPy, Scikit-learn
- Plotly (visualisations interactives)
- GeoJSON pour les cartes choroplèthes

## Installation locale

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-utilisateur/dashboard-delinquance-france.git
cd dashboard-delinquance-france
