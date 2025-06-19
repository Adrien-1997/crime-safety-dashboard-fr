import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Configuration de la page
st.set_page_config(page_title="Délinquance en France", layout="wide")

st.title("📊 Délinquance en France : évolution 2016–2024")
st.caption("Analyse statistique des infractions enregistrées par département et région")

# Lecture des données
df = pd.read_csv("donnee-dep-data.gouv-2024-geographie2024-produit-le2025-03-14.csv", sep=';', encoding='utf-8')

# Nettoyage des colonnes
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["indicateur"] = df["indicateur"].str.strip()
df["code_departement"] = df["code_departement"].astype(str).str.zfill(2)

# Supprimer les DROM : ne garder que la France métropolitaine + Corse
df = df[df["code_departement"].isin([f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"])]

# Nettoyage des taux : gérer les virgules et pourcentages
df["taux_pour_mille"] = df["taux_pour_mille"].astype(str).str.strip()
df["taux_pour_mille"] = df["taux_pour_mille"].apply(
    lambda x: float(x.replace(",", ".").replace("%", "")) / 100 if "%" in x else float(x.replace(",", "."))
)

# Ajout manuel des noms des départements
noms_depts = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence", "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes", "09": "Ariège", "10": "Aube",
    "11": "Aude", "12": "Aveyron", "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal",
    "16": "Charente", "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "2A": "Corse-du-Sud",
    "2B": "Haute-Corse", "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne",
    "25": "Doubs", "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde", "34": "Hérault",
    "35": "Ille-et-Vilaine", "36": "Indre", "37": "Indre-et-Loire", "38": "Isère", "39": "Jura",
    "40": "Landes", "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire", "44": "Loire-Atlantique",
    "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère", "49": "Maine-et-Loire",
    "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle",
    "55": "Meuse", "56": "Morbihan", "57": "Moselle", "58": "Nièvre", "59": "Nord",
    "60": "Oise", "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin", "68": "Haut-Rhin", "69": "Rhône",
    "70": "Haute-Saône", "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines", "79": "Deux-Sèvres",
    "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne", "83": "Var", "84": "Vaucluse",
    "85": "Vendée", "86": "Vienne", "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne",
    "90": "Territoire de Belfort", "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise"
}
df["nom_departement"] = df["code_departement"].map(noms_depts)

# Ajout manuel des noms des régions
noms_regions = {
    "84": "Auvergne-Rhône-Alpes", "27": "Bourgogne-Franche-Comté", "53": "Bretagne",
    "24": "Centre-Val de Loire", "94": "Corse", "44": "Grand Est", "32": "Hauts-de-France",
    "11": "Île-de-France", "28": "Normandie", "75": "Nouvelle-Aquitaine",
    "76": "Occitanie", "52": "Pays de la Loire", "93": "Provence-Alpes-Côte d'Azur"
}
df["code_region"] = df["code_region"].astype(str)
df["nom_region"] = df["code_region"].map(noms_regions)

# Chargement du GeoJSON des départements
url_geojson = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
geojson_dept = requests.get(url_geojson).json()

# Navigation par onglets
tabs = st.tabs([
    "Introduction",
    "Données brutes", 
    "Vue annuelle", 
    "Évolution temporelle", 
    "Dynamique régionale",
	"Segmentation territoriale",
	"Détection des profils atypiques",
    "Prévisions 2025"
])


st.sidebar.title("Filtres")

annees = sorted(df['annee'].unique())
indicateurs = sorted(df['indicateur'].unique())

annee_select = st.sidebar.selectbox("Année", annees)
indicateur_select = st.sidebar.selectbox("Indicateur", indicateurs)

st.set_page_config(
    page_title="Délinquance en France",
    layout="wide",
    initial_sidebar_state="expanded"  # 👈 ceci force l'ouverture
)

def footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size: 0.85rem; color: gray;'>"
        "© Adrien Morel – 2025<br>"
        "Source : Ministère de l’Intérieur – Données ouvertes disponibles sur "
        "<a href='https://www.data.gouv.fr/fr/datasets/bases-statistiques-communale-departementale-et-regionale-de-la-delinquance-enregistree-par-la-police-et-la-gendarmerie-nationales/' target='_blank' style='color:#aaa;'>data.gouv.fr</a>"
        "</div>",
        unsafe_allow_html=True
    )
	
# --- 0. PAGE D’INTRODUCTION ---
with tabs[0]:
    st.title("Tableau de bord national des indicateurs de délinquance")
    st.markdown("""
    Ce tableau de bord interactif présente une synthèse des principaux indicateurs de délinquance enregistrés en France
    métropolitaine et d’outre-mer, à partir des données collectées par la police et la gendarmerie nationales.

    **Objectifs :**
    - Offrir une vision claire de la répartition territoriale et temporelle des infractions.
    - Identifier les écarts significatifs entre départements ou régions.
    - Mettre en lumière les évolutions structurelles et les signaux faibles.
    - Fournir des éléments d’aide à la décision publique à travers des visualisations rigoureuses et pédagogiques.

    **Périmètre :**
    - Données disponibles de **2016 à 2024**
    - Calculs normalisés en **nombre pour 1000 habitants**, sauf mention contraire.
    - Données issues de l’**État 4001**, transmises par les forces de sécurité.

    **Navigation :**
    - Les onglets ci-dessus permettent d’explorer les données selon plusieurs axes : spatial, temporel, prévisionnel et analytique.
    """)

    st.markdown("---")
    st.subheader("Données utilisées")
    st.markdown("""
    - **Sources :** data.gouv.fr – Ministère de l’Intérieur
    - **Indicateurs :** atteintes aux personnes, atteintes aux biens, violences sexuelles, vols, etc.
    - **Granularité :** commune, département, région (aggrégée à la demande)
    - **Population de référence :** INSEE (millésime 2024)

    Ce tableau de bord s’inscrit dans une démarche de transparence, d’exploration, et d’optimisation de l’action publique.
    """)

    footer()

# --- 1. DONNÉES BRUTES ---
with tabs[1]:

    st.title("Nombre brut d'infractions")

    st.markdown("""
    Cette vue présente le **nombre total d'infractions enregistrées** par les forces de l'ordre pour un indicateur donné,
    à l'échelle départementale. Elle permet d’identifier les départements **les plus exposés en volume absolu**, sans tenir
    compte de la population locale.

    **Remarques importantes :**
    - Ces chiffres **ne sont pas rapportés à la population** (non comparables directement entre territoires),
    - Ils reflètent à la fois la réalité des infractions et les dynamiques de signalement et d’enregistrement,
    - L’analyse détaillée et la mise en perspective se trouvent dans les onglets suivants.

    """)
    
    df_nb = df[(df['annee'] == annee_select) & (df['indicateur'] == indicateur_select)]

    if df_nb.empty:
        st.warning("Aucune donnée disponible pour cette combinaison.")
    else:
        col1, col2 = st.columns(2)

        # 1. Carte des nombres bruts
        fig_nb_map = px.choropleth(
            df_nb,
            geojson=geojson_dept,
            locations="code_departement",
            color="nombre",
            featureidkey="properties.code",
            hover_name="nom_departement",
            color_continuous_scale="Blues",
            title=f"Carte des infractions totales – {indicateur_select} ({annee_select})"
        )
        fig_nb_map.update_geos(fitbounds="locations", visible=False)
        col1.plotly_chart(fig_nb_map, use_container_width=True)

        # 2. Bar chart top 10 départements
        top10_nb = df_nb.sort_values("nombre", ascending=False).head(10)
        fig_top_nb = px.bar(
            top10_nb,
            x="nom_departement",
            y="nombre",
            color="nombre",
            color_continuous_scale="Blues",
            title="Top 10 départements – nombre d'infractions",
            labels={"nombre": "Nombre d'infractions", "nom_departement": "Département"}
        )
        fig_top_nb.update_layout(xaxis_title="Département", yaxis_title="Infractions (nb)")
        col2.plotly_chart(fig_top_nb, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)

        # 3. Histogramme des infractions
        fig_hist_nb = px.histogram(
            df_nb,
            x="nombre",
            nbins=30,
            title="Distribution du nombre d'infractions",
            labels={"nombre": "Nombre d'infractions"}
        )
        col3.plotly_chart(fig_hist_nb, use_container_width=True)

        # 4. Boxplot
        fig_box_nb = px.box(
            df_nb,
            y="nombre",
            points="all",
            title="Boîte à moustaches – nombre d'infractions",
            labels={"nombre": "Nombre d'infractions"}
        )
        col4.plotly_chart(fig_box_nb, use_container_width=True)

    footer()


# --- 2. VUE ANNUELLE ---
with tabs[2]:
    st.title("Nombre d’infractions pour 1000 habitants")

    st.markdown("""
    Cette vue présente les **données d’infractions ramenées à 1000 habitants**, permettant une comparaison plus juste entre territoires,
    indépendamment de leur taille ou densité de population.

    **Objectifs :**
    - Mettre en lumière les départements ayant un niveau d’infractions anormalement élevé au regard de leur population,
    - Identifier les extrêmes statistiques pour chaque indicateur,
    - Explorer la distribution des valeurs au niveau national.

    **Méthodologie :**
    - Les données sont issues des enregistrements annuels d'infractions par département,
    - La valeur affichée correspond au **nombre moyen d’infractions pour 1000 habitants** pour l’année sélectionnée.

    ⚠️ Ces valeurs doivent être interprétées en tenant compte de la densité, des dynamiques territoriales et des effets de sous- ou sur-déclaration.
    """)
	
    st.markdown("---")

    df_filtered = df[(df['annee'] == annee_select) & (df['indicateur'] == indicateur_select)]

    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour cette combinaison.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Moyenne France – nombre pour 1000 habitants", f"{df_filtered['taux_pour_mille'].mean():.2f}")
        col2.metric("Maximum départemental", f"{df_filtered['taux_pour_mille'].max():.2f}")
        col3.metric("Nombre de départements", df_filtered['code_departement'].nunique())

        st.markdown("---")

        col_map, col_bar = st.columns(2)

        # Carte
        fig_map = px.choropleth(
            df_filtered,
            geojson=geojson_dept,
            locations="code_departement",
            color="taux_pour_mille",
            featureidkey="properties.code",
            hover_name="nom_departement",
            color_continuous_scale="OrRd",
            range_color=(df_filtered["taux_pour_mille"].min(), df_filtered["taux_pour_mille"].max()),
            title=f"Carte – Nombre d’infractions pour 1000 habitants ({annee_select})"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            coloraxis_colorbar_title="Nombre pour 1000 habitants"
        )
        col_map.plotly_chart(fig_map, use_container_width=True)

        # Bar chart Top 10
        top10 = df_filtered.sort_values("taux_pour_mille", ascending=False).head(10)
        fig_bar = px.bar(
            top10,
            x="nom_departement",
            y="taux_pour_mille",
            color="taux_pour_mille",
            color_continuous_scale="OrRd",
            title="Top 10 départements – Nombre pour 1000 habitants",
            labels={
                "taux_pour_mille": "Nombre pour 1000 habitants",
                "nom_departement": "Département"
            }
        )
        fig_bar.update_layout(xaxis_title="Département", yaxis_title="Nombre pour 1000 habitants")
        col_bar.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        col_box, col_hist = st.columns(2)

        # Boxplot
        fig_box = px.box(
            df_filtered,
            y="taux_pour_mille",
            points="all",
            title="Boîte à moustaches – Nombre pour 1000 habitants",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants"}
        )
        col_box.plotly_chart(fig_box, use_container_width=True)

        # Histogramme
        fig_hist = px.histogram(
            df_filtered,
            x="taux_pour_mille",
            nbins=30,
            title="Distribution – Nombre pour 1000 habitants",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants"}
        )
        col_hist.plotly_chart(fig_hist, use_container_width=True)

    footer()


# --- 3. ÉVOLUTION TEMPORELLE ---
with tabs[3]:
    st.title("Évolution temporelle par département")

    st.markdown("""
    Cette vue permet d’examiner l’évolution des indicateurs de délinquance **au sein d’un département donné**, sur plusieurs années,
    avec une granularité annuelle.

    **Objectifs :**
    - Identifier des tendances structurelles (hausse, stabilité, baisse),
    - Comparer plusieurs indicateurs sur une même période,
    - Détecter d’éventuels pics ou ruptures anormales.

    **Fonctionnalités :**
    - Sélection libre du département à explorer,
    - Affichage combiné de plusieurs indicateurs,
    - Comparaison visuelle et synthétique via courbes, heatmaps et barres empilées.

    Toutes les données sont exprimées en **nombre pour 1000 habitants**, pour assurer une lecture cohérente dans le temps.
    """)

    dep_select = st.selectbox("Choisissez un département", sorted(df['nom_departement'].dropna().unique()))
    indicateurs_multi = st.multiselect(
        "Indicateurs à afficher",
        sorted(df['indicateur'].unique()),
        default=["Coups et blessures volontaires", "Violences sexuelles"]
    )

    df_line = df[(df['nom_departement'] == dep_select) & (df['indicateur'].isin(indicateurs_multi))]

    if df_line.empty:
        st.warning("Aucune donnée disponible pour cette sélection.")
    else:
        col1, col2 = st.columns(2)

        # 1. Courbe temporelle
        fig_line = px.line(
            df_line,
            x="annee",
            y="taux_pour_mille",
            color="indicateur",
            markers=True,
            title=f"Évolution annuelle – {dep_select}",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants"}
        )
        fig_line.update_layout(xaxis_title="Année", yaxis_title="Nombre pour 1000 habitants")
        col1.plotly_chart(fig_line, use_container_width=True)

        # 2. Heatmap année × indicateur
        pivot = df_line.pivot_table(
            index="indicateur", columns="annee", values="taux_pour_mille", aggfunc="mean"
        )
        fig_heatmap = px.imshow(
            pivot,
            labels=dict(color="Nombre pour 1000 habitants", x="Année", y="Indicateur"),
            aspect="auto",
            title=f"Carte thermique – {dep_select}"
        )
        col2.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)

        # 3. Histogramme par tranche
        fig_bin = px.histogram(
            df_line,
            x="taux_pour_mille",
            nbins=20,
            color="indicateur",
            title="Distribution des valeurs (groupées)",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants"}
        )
        col3.plotly_chart(fig_bin, use_container_width=True)

        # 4. Barres empilées année x indicateur
        fig_stacked = px.bar(
            df_line,
            x="annee",
            y="taux_pour_mille",
            color="indicateur",
            barmode="stack",
            title="Évolution annuelle par indicateur",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants"}
        )
        col4.plotly_chart(fig_stacked, use_container_width=True)

    footer()


# --- 4. MOYENNE RÉGIONALE ---
with tabs[4]:
    st.title("Vue régionale agrégée")

    st.markdown("""
    Cette vue agrège les données à l’échelle **régionale**, permettant une lecture synthétique de la situation sur
    l’ensemble du territoire. Elle met en évidence les dynamiques géographiques larges et les écarts persistants
    entre régions.

    **Objectifs :**
    - Comparer les régions en termes de **niveau moyen d’infractions pour 1000 habitants**,
    - Visualiser l’évolution dans le temps pour chaque région,
    - Repérer les régions présentant des niveaux durablement élevés ou en forte progression.

    **Méthodologie :**
    - Calcul de la **moyenne des valeurs départementales** pour chaque région et chaque année,
    - Visualisation en courbes, cartes thermiques et diagrammes de dispersion.

    Tous les indicateurs sélectionnés sont exprimés en **nombre pour 1000 habitants**.
    """)

    indicateurs_region = st.multiselect(
        "Indicateurs à inclure",
        sorted(df['indicateur'].unique()),
        default=["Coups et blessures volontaires"]
    )

    if not indicateurs_region:
        st.warning("Veuillez sélectionner au moins un indicateur.")
    else:
        df_region = df[df['indicateur'].isin(indicateurs_region)].copy()
        df_grouped = df_region.groupby(['nom_region', 'annee'])['taux_pour_mille'].mean().reset_index()

        latest_year = df_grouped['annee'].max()
        df_latest = df_grouped[df_grouped['annee'] == latest_year]

        col1, col2 = st.columns(2)

        # 1. Évolution régionale dans le temps
        fig_line_region = px.line(
            df_grouped,
            x='annee',
            y='taux_pour_mille',
            color='nom_region',
            title="Évolution du nombre moyen pour 1000 habitants par région",
            markers=True,
            labels={"taux_pour_mille": "Nombre pour 1000 habitants", "annee": "Année", "nom_region": "Région"}
        )
        fig_line_region.update_layout(
            xaxis_title="Année",
            yaxis_title="Nombre pour 1000 habitants",
            legend_title="Région"
        )
        col1.plotly_chart(fig_line_region, use_container_width=True)

        # 2. Bar chart horizontal pour la dernière année
        fig_bar_latest = px.bar(
            df_latest.sort_values("taux_pour_mille", ascending=False),
            x="taux_pour_mille",
            y="nom_region",
            orientation="h",
            color="taux_pour_mille",
            color_continuous_scale="OrRd",
            title=f"Nombre moyen pour 1000 habitants par région – {latest_year}",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants", "nom_region": "Région"}
        )
        fig_bar_latest.update_layout(
            xaxis_title="Nombre pour 1000 habitants",
            yaxis_title="",
            yaxis=dict(autorange="reversed")
        )
        col2.plotly_chart(fig_bar_latest, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)

        # 3. Boxplot
        fig_box_region = px.box(
            df_grouped,
            x="nom_region",
            y="taux_pour_mille",
            points="all",
            title="Dispersion du nombre moyen pour 1000 habitants (régions)",
            labels={"taux_pour_mille": "Nombre pour 1000 habitants", "nom_region": "Région"}
        )
        fig_box_region.update_layout(xaxis_tickangle=-45)
        col3.plotly_chart(fig_box_region, use_container_width=True)

        # 4. Heatmap région vs année
        heat_df = df_grouped.pivot(index='nom_region', columns='annee', values='taux_pour_mille')
        fig_heatmap = px.imshow(
            heat_df,
            labels=dict(x="Année", y="Région", color="Nombre pour 1000 habitants"),
            color_continuous_scale="OrRd",
            aspect="auto",
            title="Évolution du nombre pour 1000 habitants par région"
        )
        col4.plotly_chart(fig_heatmap, use_container_width=True)

    footer()


	
# --- 5. SEGMENTATION MULTI-INDICATEURS ---
with tabs[5]:
    st.title("Segmentation des départements selon les indicateurs de délinquance")

    st.markdown("""
    Cette page propose une classification des départements français en groupes homogènes, selon leur profil global
    de délinquance observée entre **2016 et 2024**, sur la base de l’ensemble des indicateurs exprimés en **nombre pour 1000 habitants**.

    **Objectifs :**
    - Mettre en évidence des profils territoriaux proches ou atypiques,
    - Faciliter l’identification de dynamiques locales spécifiques,
    - Appuyer des politiques publiques différenciées.

    **Méthodologie :**
    - Moyennes départementales calculées pour chaque indicateur entre 2016 et 2024.
    - Réduction de dimension par **analyse en composantes principales (ACP)**.
    - Segmentation en **4 groupes** à l’aide d’un algorithme de classification non supervisée (**K-means**).

    L’interprétation repose sur deux visualisations complémentaires : une **projection géographique** et une **représentation factorielle**.
    """)

    # --- Préparation des données
    pivot_taux = df.pivot_table(index="code_departement", columns="indicateur", values="taux_pour_mille", aggfunc="mean")
    pivot_taux.columns = [f"{col}_pour_1000_hab" for col in pivot_taux.columns]
    df_clust = pivot_taux.dropna(thresh=int(pivot_taux.shape[1] * 0.8))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    df_clust["Groupe"] = labels.astype(str)
    df_clust["Axe 1"] = X_pca[:, 0]
    df_clust["Axe 2"] = X_pca[:, 1]
    df_clust = df_clust.reset_index().merge(df[["code_departement", "nom_departement"]].drop_duplicates(), on="code_departement")

    # --- Carte des groupes
    st.subheader("Répartition géographique des groupes identifiés")

    fig_map = px.choropleth(
        df_clust,
        geojson=geojson_dept,
        locations="code_departement",
        color="Groupe",
        featureidkey="properties.code",
        hover_name="nom_departement",
        title="Carte des départements par groupe"
    )
    fig_map.update_geos(
        projection_type="mercator",
        center={"lat": 46.6, "lon": 2.5},
        fitbounds="geojson",
        visible=False
    )
    fig_map.update_layout(
        height=650,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # --- Projection ACP
    st.subheader("Représentation des groupes dans l’espace factoriel")

    fig_pca = px.scatter(
        df_clust,
        x="Axe 1",
        y="Axe 2",
        color="Groupe",
        hover_name="nom_departement",
        title="Projection des départements selon les deux premières composantes",
        labels={"Axe 1": "Composante principale 1", "Axe 2": "Composante principale 2"}
    )
    fig_pca.update_layout(legend_title="Groupe")
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")

    # --- Contributions par axe
    st.markdown("### Principaux indicateurs expliquant la segmentation")

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=[f.replace("_pour_1000_hab", "") for f in pivot_taux.columns],
        columns=["Axe 1", "Axe 2"]
    )

    top_pca1 = loadings_df["Axe 1"].abs().sort_values(ascending=False).head(10)
    top_pca2 = loadings_df["Axe 2"].abs().sort_values(ascending=False).head(10)

    colb1, colb2 = st.columns(2)

    with colb1:
        fig_bar1 = px.bar(
            top_pca1[::-1],
            orientation="h",
            title="Top 10 indicateurs – Axe 1",
            labels={"value": "Contribution", "index": "Indicateur"}
        )
        st.plotly_chart(fig_bar1, use_container_width=True)

    with colb2:
        fig_bar2 = px.bar(
            top_pca2[::-1],
            orientation="h",
            title="Top 10 indicateurs – Axe 2",
            labels={"value": "Contribution", "index": "Indicateur"}
        )
        st.plotly_chart(fig_bar2, use_container_width=True)

    footer()



# --- 6. Identification des départements au profil atypique ---
with tabs[6]:
    st.title("Départements hors normes – Analyse comparative")

    df_anom = df[(df["annee"] == annee_select) & (df["indicateur"] == indicateur_select)].copy()
    df_anom = df_anom.dropna(subset=["taux_pour_mille", "insee_pop", "insee_log"])

    if df_anom.empty:
        st.warning("Pas de données suffisantes pour cette combinaison.")
    else:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Préparation des données
        features = df_anom[["taux_pour_mille", "insee_pop", "insee_log"]].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        contamination_rate = 0.1  # seuil de détection
        model = IsolationForest(contamination=contamination_rate, random_state=42)
        df_anom["profil"] = model.fit_predict(X_scaled)
        df_anom["profil"] = df_anom["profil"].map({1: "Dans la norme", -1: "Hors norme"})

        # Données pour les départements hors norme
        atypiques = df_anom[df_anom["profil"] == "Hors norme"].copy()
        moyenne_nationale = df_anom["taux_pour_mille"].mean()
        atypiques["Écart à la moyenne"] = (atypiques["taux_pour_mille"] - moyenne_nationale).round(2)
        atypiques["Rang parmi les hors norme"] = atypiques["taux_pour_mille"].rank(ascending=False).astype(int)

        atypiques = atypiques.rename(columns={
            "nom_departement": "Département",
            "taux_pour_mille": "Nombre pour 1000 habitants",
            "insee_pop": "Population (INSEE)",
            "insee_log": "Logements (INSEE)"
        })

        # --- Affichage plein écran : CARTE ---
        st.subheader(f"Départements au profil atypique – {indicateur_select} ({annee_select})")

        fig_anom_map = px.choropleth(
            df_anom,
            geojson=geojson_dept,
            locations="code_departement",
            color="profil",
            featureidkey="properties.code",
            hover_name="nom_departement",
            hover_data={
                "code_departement": False,
                "profil": True,
                "taux_pour_mille": ":.2f"
            },
            color_discrete_map={"Dans la norme": "lightgray", "Hors norme": "crimson"},
        )

        fig_anom_map.update_geos(
            projection_type="mercator",
            center={"lat": 46.6, "lon": 2.5},
            fitbounds="geojson",
            visible=False
        )

        fig_anom_map.update_layout(
            height=750,
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        st.plotly_chart(fig_anom_map, use_container_width=True)

        # --- Affichage tableau explicatif ---
        st.subheader("Départements au profil hors norme")
        st.markdown(
            "Ce tableau met en évidence les départements identifiés comme ayant un **profil atypique** en matière d'infractions, "
            "en tenant compte de la population et du nombre de logements.\n\n"
            "- Les valeurs sont exprimées en **nombre pour 1000 habitants**.\n"
            "- L’**écart à la moyenne nationale** permet d’identifier les cas les plus significatifs.\n"
            "- Le **rang** classe les départements hors norme selon cette valeur.\n"
            "- **Méthodologie** : un algorithme de détection automatique a identifié 10 % des départements comme hors norme (Isolation Forest)."
        )

        st.dataframe(
            atypiques[[
                "Département", "Nombre pour 1000 habitants", "Écart à la moyenne",
                "Rang parmi les hors norme", "Population (INSEE)", "Logements (INSEE)"
            ]].sort_values("Nombre pour 1000 habitants", ascending=False).reset_index(drop=True)
        )

    footer()

# --- 7. PRÉVISIONS 2025 (nombre estimé pour 1000 habitants) ---
with tabs[7]:
    st.title("Prévisions 2025 – Nombre estimé pour 1000 habitants")

    st.markdown("""
    Cette section propose une **projection départementale** du nombre d’infractions **attendues en 2025**, ramenées à 1000 habitants,
    à partir des tendances observées entre les années précédentes.

    **Objectif :**
    - Fournir une estimation anticipée pour les départements les plus concernés,
    - Offrir une base de réflexion pour le déploiement de moyens ou d’actions ciblées.

    **Méthodologie :**
    - Données historiques agrégées par département (nombre total de faits annuels),
    - Modélisation par **régression linéaire simple** pour chaque département (si ≥ 5 années disponibles),
    - Projection 2025 rapportée à la population INSEE 2024.

    ⚠️ Ces prévisions sont **indicatives** : elles ne tiennent pas compte d’effets exogènes (conjoncture, politique locale, phénomènes exceptionnels).
    """)

    # Agrégation annuelle
    df_agg = df.groupby(["annee", "code_departement", "nom_departement"])["nombre"].sum().reset_index()
    df_pivot = df_agg.pivot(index="annee", columns="code_departement", values="nombre")

    from sklearn.linear_model import LinearRegression
    forecast_2025_counts = {}

    for dept in df_pivot.columns:
        serie = df_pivot[dept].dropna()
        if len(serie) >= 5:
            X = np.array(serie.index).reshape(-1, 1)
            y = serie.values
            model = LinearRegression()
            model.fit(X, y)
            forecast = model.predict(np.array([[2025]]))[0]
            forecast_2025_counts[dept] = forecast

    df_forecast = pd.DataFrame(list(forecast_2025_counts.items()), columns=["code_departement", "faits_2025"])
    df_forecast["code_departement"] = df_forecast["code_departement"].astype(str)

    # Récupération des populations 2024
    df_pop_2024 = df[df["annee"] == 2024][["code_departement", "insee_pop"]].drop_duplicates()
    df_forecast = df_forecast.merge(df_pop_2024, on="code_departement", how="left")

    # Calcul du nombre pour 1000 habitants
    df_forecast["nombre_pour_1000_habitants"] = (df_forecast["faits_2025"] / df_forecast["insee_pop"]) * 1000

    # Ajout des noms
    df_forecast = df_forecast.merge(
        df[["code_departement", "nom_departement"]].drop_duplicates(), on="code_departement"
    )

    # --- Carte
    st.subheader("Carte des prévisions départementales")

    fig_map_pred = px.choropleth(
        df_forecast,
        geojson=geojson_dept,
        locations="code_departement",
        color="nombre_pour_1000_habitants",
        featureidkey="properties.code",
        hover_name="nom_departement",
        color_continuous_scale="Oranges",
        title="Prévision 2025 – Nombre estimé pour 1000 habitants"
    )

    fig_map_pred.update_geos(
        projection_type="mercator",
        center={"lat": 46.6, "lon": 2.5},
        fitbounds="geojson",
        visible=False
    )

    fig_map_pred.update_layout(
        height=750,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar_title="Nombre pour 1000 habitants"
    )

    st.plotly_chart(fig_map_pred, use_container_width=True)

    # --- Bar chart Top 10
    st.subheader("Départements les plus concernés")

    top_pred = df_forecast.sort_values("nombre_pour_1000_habitants", ascending=False).head(10)
    fig_bar_pred = px.bar(
        top_pred,
        x="nom_departement",
        y="nombre_pour_1000_habitants",
        color="nombre_pour_1000_habitants",
        color_continuous_scale="Oranges",
        title="Top 10 départements – Nombre estimé pour 1000 habitants (2025)",
        labels={
            "nom_departement": "Département",
            "nombre_pour_1000_habitants": "Nombre pour 1000 habitants"
        }
    )
    fig_bar_pred.update_layout(
        xaxis_title="Département",
        yaxis_title="Nombre estimé pour 1000 habitants"
    )

    st.plotly_chart(fig_bar_pred, use_container_width=True)

    footer()

	








