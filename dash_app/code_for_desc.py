import pandas as pd
import numpy as np

path = r'C:\Users\Julien\Documents\WildCodeSchool\Projet\Projet_3\dash_app\dataset\desciption_dash.csv'


x='nothing for help code'



# desc Acceuil
renale = """La maladie chronique rénale est une condition pathologique caractérisée par une détérioration progressive et persistante de la fonction rénale.
Elle implique une diminution de la capacité des reins à filtrer et à éliminer les déchets et les toxines du sang, ainsi qu'à maintenir l'équilibre électrolytique et hydrique dans le corps.
La maladie rénale chronique peut résulter de diverses causes, notamment l'hypertension artérielle, le diabète, les infections rénales récurrentes ou d'autres affections sous-jacentes."""

diabete = """Le diabète est une maladie chronique qui se caractérise par des niveaux élevés de glucose dans le sang, souvent causés par une production insuffisante d'insuline par le pancréas
ou une résistance à l'insuline chez les cellules du corps. Le diabète peut entraîner de graves complications à long terme, telles que des lésions des vaisseaux sanguins, des problèmes cardiaques,
des atteintes nerveuses, des lésions rénales et des problèmes de vision."""

foie = """Le cancer du foie est une maladie maligne qui se développe au niveau des cellules. Il résulte généralement de l'exposition à des agents cancérigènes due à une consommation excessive d'alcool.
Le cancer du foie se caractérise par une croissance anormale et incontrôlée des cellules hépatiques, formant une masse ou une tumeur dans le foie.
Les cellules cancéreuses peuvent se propager à d'autres parties du foie ou à des organes distants du corps par le système sanguin. il est souvent diagnostiqué à un stade avancé, ce qui rend son traitement plus difficile.
"""

cardiaque = """Le cancer cardiaque, bien que rare, peut se développer dans les tissus du cœur. Il peut affecter différentes parties du cœur, y compris les muscles, les valves, ou les vaisseaux sanguins.
Les cellules cancéreuses qui se forment dans le cœur peuvent compromettre son fonctionnement normal en perturbant le flux sanguin ou en interférant avec les contractions cardiaques régulières.
Comme pour d'autres cancers, le cancer cardiaque peut potentiellement se propager à d'autres parties du corps par le biais du système lymphatique ou sanguin.
Les symptômes peuvent inclure des palpitations, des douleurs thoraciques, une fatigue extrême, et des difficultés respiratoires.
Le traitement du cancer cardiaque peut impliquer une combinaison de chirurgie, de chimiothérapie et de radiothérapie, selon le stade et la localisation de la maladie.
"""

sein = """Le cancer du sein est une maladie maligne qui se développe au niveau des cellules du sein. Il s'agit d'une croissance anormale et incontrôlée de ces cellules, 
formant une masse ou une tumeur. Les cellules cancéreuses peuvent envahir les tissus voisins et se propager à d'autres parties du corps par le biais du système lymphatique ou sanguin, 
un processus appelé métastase. Le cancer du sein peut toucher aussi bien les hommes que les femmes, bien que ce soit plus fréquent chez les femmes."""



# desc Predict
#Renale   -> ['Haemoglobin', 'Specific_Gravity', 'Blood_Urea', 'Blood_Glucose_Random', 'Blood_Pressure', 'Pus_Cell', 'appetit', 'Sugar'] <- Renale
vR1 = "quantité d'hémoglobine dans le sang, une protéine qui transporte l'oxygène des poumons vers les tissus du corps."
vR2 = "concentration des solutés dans l'urine par rapport à l'eau."
vR3 = "Il s'agit de la quantité d'urée présente dans le sang. L'urée est un produit de dégradation des protéines."
vR4 = "concentration de glucose dans le sang à un moment donné"
vR5 = "force du sang exercée contre les parois des artères. "
vR6 = "présence de cellules de pus dans l'urine, 0: pas de présence, 1: présence"
vR7 = "votre envie de manger, 0: vous avez de l'appetit, 1: vous n'avez pas d'appetit"
vR8 = "présence de sucre dans l'urine"

#Diabèthe   -> ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
vD1 = "désigne le nombre de fois où une femme a été enceinte"
vD2 = "concentration de glucose plasmatique"
vD3 = "pression artérielle diastolique"
vD4 = "épaisseur de la peau"
vD5 = "insuline sérique sur 2h"
vD6 = "indice de masse corporelle"
vD7 = "fonction pedigree du diabete"
vD8 = "Votre age actuel"

#Foie   ->  ['Direct_Bilirubin', 'Total_Bilirubin', 'Alamine_Aminotransferase', 'Alkaline_Phosphotase', 'Albumin', 'Albumin_and_Globulin_Ratio']  <- Foie
vF1 = "Votre taux de bilirubine directe dans le sang."
vF2 = "Votre taux de bilirubine totale dans le sang."
vF3 = "Votre taux d'alanine aminotransférase dans le sang."
vF4 = "Votre taux de phosphatase alcaline dans le sang."
vF5 = "Votre taux d'albumine dans le sang."
vF6 = "Votre rapport entre l'albumine et la globuline (les deux principales protéines du sang)."

#Cardiaque    -> ['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'] <- Cardiaque
vC1 = "Votre age actuel"
vC2 = "Votre genre : 0 pour homme et 1 pour feamme"
vC3 = "Votre type de douleurs thoracique (Chest Pain), 1: angine de poitrine typique, 2: angine de poitrine atypique, 3: douleurs autre angine de poitrine, 4: asymptomatic"
vC4 = "Votre tension artérielle au repos"
vC5 = "Votre cholestérol sérique"
vC6 = "Votre résultats électrocardiographiques au repos., 0 : hypertrophie ventriculaire gauche probable ou certaine selon les critères d'Estes, 1 : RAS, 2 : anomalie de l'onde ST-T"
vC7 = "Votre fréquence cardiaque maximale atteinte."
vC8 = "Votre angine de poitrine induite par l'exercice., 1 : présence d'angine, Valeur 2 : RAS"
vC9 = "Votre dépression du segment ST induite par l'exercice par rapport au repos"
vC10 = "Votre pente du segment ST à l'effort maximal., 0 : en pente descendante, 1 : plat, 2 : en pente ascendante"
vC11 = "Votre nombre de vaisseaux majeurs colorés par fluoroscopie (de 0 à 3)"
vC12 = "Votre type de thalassémie, Valeur 0 : NULL, Valeur 1 : défaut fixe, Valeur 2 : flux sanguin normal, Valeur 3 : anomalie réversible"

#Seins     -> ['area_mean', 'concavity_mean', 'texture_mean', 'smoothness_mean'] <- Seins
vS1 = "Votre moyenne de la surface de la tumeur."
vS2 = "Votre moyenne de la gravité des parties concaves du contour de la tumeur"
vS3 = "Votre moyenne de la texture de la tumeur"
vS4 = "Votre moyenne de lissé de la tumeur"

# desc Graphique



# desc info
dataset_utilise = "Nous avons utilisé des datasets annonyme données par Soufiane Maski puis nous les avons retravaillé pour qu'il soit plus facile à utilisé"
team_member = "Ce projet à eté fait par BESSON Nicholas, KETTE Gilles, GUTIERREZ Jesus et FROMONT Julien"
biblio_use = "Nousa avons utilisé les bibliothèque suivant: Pandas, Dash, Plotly et sklearn"


data = {'descAcceuil': [renale, diabete, foie, cardiaque, sein],
        'descPredict': [x, vR1, vR2, vR3, vR4, vR5, vR6, vR7, vR8, vD1, vD2, vD3, vD4, vD5, vD6, vD7, vD8, vF1, vF2, vF3, vF4, vF5, vF6, vC1, vC2, vC3, vC4, vC5, vC6, vC7, vC8, vC9, vC10, vC11, vC12, vS1, vS2, vS3, vS4],
        'descInfo': [dataset_utilise, team_member, biblio_use]}

data = {k: v + [np.nan] * ((max(len(v) for v in data.values())) - len(v)) for k, v in data.items()} # Trouver la longueur maximale parmi toutes les colonnes et Remplir les listes avec NaN si nécessaire
df = pd.DataFrame(data)

df.to_csv(path ,index=False) 
