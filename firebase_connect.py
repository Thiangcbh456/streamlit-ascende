# arquivo: firebase_connect.py
from firebase_connect import df_vagas
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# -------------------------
# 1. Inicializar Firebase
# -------------------------

# Caminho para a sua chave JSON
chave_json = r"C:\Users\thiag\Documents\ASCENDE-DASHBOARD\ascende-firebase.json"

cred = credentials.Certificate(chave_json)
firebase_admin.initialize_app(cred)

# Criar cliente do Firestore
db = firestore.client()

# -------------------------
# 2. Ler coleção do Firestore
# -------------------------

# Substitua 'vagas' pelo nome da sua coleção no Firestore
colecao_vagas = db.collection("vagas")

# Pegar todos os documentos
docs = colecao_vagas.stream()

# -------------------------
# 3. Transformar em DataFrame
# -------------------------

lista_vagas = []

for doc in docs:
    vaga = doc.to_dict()
    vaga['id'] = doc.id  # opcional, adiciona o ID do documento
    lista_vagas.append(vaga)

# Converter para DataFrame do pandas
df_vagas = pd.DataFrame(lista_vagas)

# Visualizar as primeiras linhas
print(df_vagas.head())
