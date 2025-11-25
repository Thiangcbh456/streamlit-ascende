# arquivo: ascende_dashboard_template.py
# ===========================================================
# ASCENDE Dashboard – Firebase + RAIS/CAGED + Home filtrável + Sidebar
# ===========================================================

import os, re, time, tempfile, urllib.request, requests
import streamlit as st
import pandas as pd
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, firestore

# ---------- PyArrow ou DuckDB ----------
try:
    import pyarrow.parquet as pq
    USE_PYARROW = True
except Exception:
    import duckdb
    USE_PYARROW = False


# ===========================================================
# 1️⃣ Firebase
# ===========================================================
@st.cache_data(show_spinner=True)
def carregar_dados_firebase(caminho_credencial: str) -> pd.DataFrame:
    if not firebase_admin._apps:
        cred = credentials.Certificate(caminho_credencial)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    vagas = [doc.to_dict() | {"id": doc.id} for doc in db.collection("vagas").stream()]
    df = pd.DataFrame(vagas)
    if df.empty:
        return pd.DataFrame()

    df.rename(columns={
        "Título da Vaga": "Titulo da Vaga",
        "Salário": "Salario",
        "Localização": "Localizacao"
    }, inplace=True, errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]

    if "Localizacao" in df.columns:
        df["Localizacao"] = df["Localizacao"].astype(str).str.strip()

    def limpar_titulo(v):
        if pd.isna(v):
            return pd.NA
        t = str(v).strip()
        if t.lower() in ["", "nan", "none", "n/a", "não informado", "nao informado"]:
            return pd.NA
        return t
    df["Titulo da Vaga"] = df.get("Titulo da Vaga", "").apply(limpar_titulo)

    def limpar_habilidades(v):
        if not v or str(v).strip().lower() in ["nan", "none", "não informado"]:
            return ""
        parts = re.split(r"[,;/]", str(v))
        parts = [p.strip().title() for p in parts if p.strip()]
        return ", ".join(sorted(set(parts)))
    df["Habilidades"] = df.get("Habilidades", "").apply(limpar_habilidades)

    def extrair_salario(v):
        try:
            if not v or not isinstance(v, (str, int, float)):
                return float("nan")
            txt = str(v).replace(".", "").replace(",", ".")
            nums = re.findall(r"\d+[.,]?\d*", txt)
            return float(nums[0]) if nums else float("nan")
        except Exception:
            return float("nan")
    df["Salario_num"] = df.get("Salario", "").apply(extrair_salario)

    def extract_state_uf(localizacao: str) -> str:
        if not isinstance(localizacao, str):
            return None
        loc = localizacao.strip().upper()
        match = re.search(r"/\s*([A-Z]{2})\b", loc)
        uf = match.group(1) if match else None
        ufs = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS",
               "MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"]
        return uf if uf in ufs else None
    if "Localizacao" in df.columns:
        df["Estado_UF"] = df["Localizacao"].apply(extract_state_uf)
    else:
        df["Estado_UF"] = None

    return df.reset_index(drop=True)


# ===========================================================
# 2️⃣ Ler Parquet (CAGED / RAIS)
# ===========================================================
@st.cache_data(show_spinner=True)
def carregar_parquet(origem, sample=None) -> pd.DataFrame:
    try:
        if isinstance(origem, str):
            if "drive.google.com" in origem:
                fid = origem.split("/d/")[1].split("/")[0]
                origem = f"https://drive.google.com/uc?export=download&id={fid}"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
            urllib.request.urlretrieve(origem, tmp.name)
            path = tmp.name
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
            tmp.write(origem.read()); tmp.flush(); path = tmp.name
        if USE_PYARROW:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pandas()
        else:
            import duckdb
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
            con.close()
            try:
                os.remove(path)
            except PermissionError:
                pass
        if sample:
            df = df.head(sample)
        return df
    except Exception as e:
        st.error(f"Erro ao ler Parquet: {e}")
        return pd.DataFrame()


# ===========================================================
# 3️⃣ Datasets .parquet — corrigidos
# ===========================================================
DATASETS = {
    "CAGED 2025-01":"https://drive.google.com/file/d/1G7aT5GQwX886EIJiu1EVgl7s3Cl1-Wyr/view",
    "CAGED 2025-02":"https://drive.google.com/file/d/1nOIpt9UUUOpeUtXb0DCP-0j2EPtrrzTO/view",
    "CAGED 2025-03":"https://drive.google.com/file/d/14hILDiIooTM65AYjksMGaW83EvQs3xLz/view",
    "CAGED 2025-04":"https://drive.google.com/file/d/1G7cYBtCRK-BcolVP2GuEM210gU8aM7_I/view",
    "CAGED 2025-05":"https://drive.google.com/file/d/1iTMPWsAkLUJ9OltEecFaokO-hOL9w4Ka/view",
    "CAGED 2025-06":"https://drive.google.com/file/d/1ASFVCvpTMRlF5dqhnLgbXeZQiisj04jM/view",
    "CAGED 2025-07":"https://drive.google.com/file/d/1vL92J_yG2KqaUwbz-r-TcEo3Rj5zFaBU/view",
    "RAIS 2023":"https://drive.google.com/file/d/13lEIFqkdtZ6FPZ4be8qcQnCww1H58IgV/view"
}


# ===========================================================
# 4️⃣ Inicializar e Criar Filtros
# ===========================================================
cred_path_firebase = r"C:\Users\thiag\Documents\ASCENDE-DASHBOARD\ascende-firebase.json"
df_vagas = carregar_dados_firebase(cred_path_firebase)
if df_vagas.empty:
    st.warning("Nenhuma vaga encontrada.")
    st.stop()

# ======== Filtros laterais fixos =========
st.sidebar.header("📊 Filtros")

tipos = sorted(df_vagas.get("Tipo de Vaga", pd.Series([], dtype=str)).dropna().unique().tolist())
cidades = sorted(df_vagas.get("Localizacao", pd.Series([], dtype=str)).dropna().unique().tolist())
habs = sorted(
    h for h in df_vagas.get("Habilidades", pd.Series([], dtype=str))
      .dropna().astype(str).str.split(",").explode().str.strip().unique() if h
)

svals = df_vagas["Salario_num"].dropna()
sal_min, sal_max = (0, int(svals.max()) if not svals.empty else 10000)

faixa = st.sidebar.slider("Faixa Salarial (R$)", sal_min, sal_max, (sal_min, sal_max), step=100)
tipo_sel = st.sidebar.multiselect("Tipo de Vaga", tipos)
cidade_sel = st.sidebar.multiselect("Localização", cidades)
hab_sel = st.sidebar.multiselect("Habilidades", habs)


# ===========================================================
# 🕒 Filtro de Horário — completamente flexível
# ===========================================================
st.sidebar.markdown("🕒 **Horário personalizado**")

col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    hora_inicio = st.time_input("Começa às", value=pd.Timestamp("08:00").time())
with col_h2:
    hora_fim = st.time_input("Termina às", value=pd.Timestamp("18:00").time())

st.sidebar.markdown(
    """
    <div style='background-color:#f9f9f9; border:1px solid #ddd; border-radius:6px;
                padding:8px 10px; margin-top:6px; margin-bottom:4px;'>
        <small style='color:#555;'>⚙️ Ajustes adicionais de horário</small>
    </div>
    """,
    unsafe_allow_html=True
)

ativar_filtro_indefinido = st.sidebar.toggle(
    "Ativar remoção de horários específicos", value=False
)

def obter_opcoes_horario(df: pd.DataFrame) -> list[str]:
    if "Horario" not in df.columns:
        return []
    horarios = (
        df["Horario"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(set(horarios))

horarios_indesejados = []
if ativar_filtro_indefinido:
    opcoes_horarios = obter_opcoes_horario(df_vagas)
    if not opcoes_horarios:
        st.sidebar.info("Nenhuma descrição de horário encontrada no Firebase.")
        opcoes_horarios = []
    horarios_indesejados = st.sidebar.multiselect(
        "Selecione os horários que deseja ocultar (exceções):",
        options=opcoes_horarios,
        help="Essas opções são carregadas automaticamente a partir da coluna 'Horario' do Firebase."
    )


# ===========================================================
# Função de filtragem geral
# ===========================================================
def extrair_hora(texto):
    if not isinstance(texto, str):
        return None
    match = re.search(r'(\d{1,2})[:hH](\d{2})', texto)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        if 0 <= h < 24 and 0 <= m < 60:
            return pd.Timestamp(f"{h:02d}:{m:02d}").time()
    return None


def aplicar_filtros(df):
    df_f = df.copy()
    mn, mx = faixa
    df_f = df_f[df_f["Salario_num"].fillna(0).between(mn, mx)]

    if tipo_sel:
        df_f = df_f[df_f["Tipo de Vaga"].isin(tipo_sel)]
    if cidade_sel:
        df_f = df_f[df_f["Localizacao"].isin(cidade_sel)]
    if hab_sel:
        mask = df_f["Habilidades"].fillna("").apply(
            lambda v: any(h.lower() in str(v).lower() for h in hab_sel)
        )
        df_f = df_f[mask]

    if "Horario" in df_f.columns and not df_f["Horario"].isna().all():
        df_f["hora_extraida"] = df_f["Horario"].apply(extrair_hora)
        if hora_inicio and hora_fim:
            df_f = df_f[
                df_f["hora_extraida"].isna() |
                df_f["hora_extraida"].apply(
                    lambda t: isinstance(t, type(hora_inicio)) and hora_inicio <= t <= hora_fim
                )
            ]
        if ativar_filtro_indefinido and horarios_indesejados:
            termos = [t.lower() for t in horarios_indesejados]
            padrao = "|".join(map(re.escape, termos))
            df_f = df_f[
                ~df_f["Horario"].astype(str).str.lower().str.contains(padrao, regex=True, na=False)
            ]

    df_f = df_f.drop(columns=["hora_extraida"], errors="ignore")
    return df_f


df_filt = aplicar_filtros(df_vagas)


# ===========================================================
# 5️⃣ Sidebar – Navegação
# ===========================================================
st.sidebar.title("🧭 Navegação")
menu = st.sidebar.radio(
    "Ir para:",
    ["🏠 Home", "💼 Vagas", "🧾 CAGED", "📊 RAIS", "📈 Análises Avançadas"]
)

st.title("ASCENDE — Dashboard de Oportunidades em TI")


# ===========================================================
# 🏠 HOME
# ===========================================================
if menu == "🏠 Home":
    st.header("🏠 Visão Geral das Oportunidades")
    total, filtradas = len(df_vagas), len(df_filt)
    sal_med = df_filt["Salario_num"].dropna().mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Vagas (Total)", f"{total:,}")
    col2.metric("Vagas Filtradas", f"{filtradas:,}")
    col3.metric("Salário Médio (Filtro)", f"{sal_med:,.2f}" if not pd.isna(sal_med) else "N/A")

    if "Tipo de Vaga" in df_filt:
        tipo_freq = df_filt["Tipo de Vaga"].dropna().value_counts().reset_index()
        tipo_freq.columns = ["Tipo de Vaga", "Vagas"]
        st.subheader("🧩 Distribuição por Tipo de Vaga")
        fig = px.pie(tipo_freq, names="Tipo de Vaga", values="Vagas",
                     hole=0.45, color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(textinfo="percent+label", pull=[0.03]*len(tipo_freq))
        st.plotly_chart(fig, use_container_width=True)

    if "Empresa" in df_filt:
        st.subheader("🏢 Empresas que Mais Contratam")
        top_emp = (
            df_filt["Empresa"]
            .dropna()
            .replace("", pd.NA)
            .replace("Não Informada", pd.NA)
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_emp.columns = ["Empresa", "Total Vagas"]

        if not top_emp.empty:
            top_emp = top_emp.sort_values(by="Total Vagas", ascending=True)
            fig_emp = px.bar(
                top_emp, x="Total Vagas", y="Empresa",
                orientation="h", color="Total Vagas",
                color_continuous_scale=px.colors.sequential.Tealgrn,
                text="Total Vagas"
            )
            fig_emp.update_layout(
                height=480, margin=dict(l=120, r=40, t=40, b=40),
                plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Número de Vagas", yaxis_title="",
                coloraxis_showscale=False
            )
            fig_emp.update_traces(
                texttemplate="%{text}",
                textposition="outside",
                marker_line_color="teal",
                marker_line_width=1
            )
            st.plotly_chart(fig_emp, use_container_width=True)
        else:
            st.info("Nenhuma empresa informada disponível para exibir.")

    if "Habilidades" in df_filt:
        todas = df_filt["Habilidades"].dropna().astype(str).str.split(",").explode().str.strip().str.title()
        todas = todas[todas != ""]
        freq = todas.value_counts()
        if not freq.empty:
            st.subheader("☁️ Nuvem de Habilidades")
            todas_habs = sorted(freq.index.tolist())
            remover = st.multiselect("Remover habilidades da visualização:", todas_habs)
            freq_filtrado = freq.drop(labels=remover, errors="ignore")

            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            wc = WordCloud(width=800, height=300, background_color="white", max_words=80)
            wc.generate_from_frequencies(freq_filtrado.to_dict())
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig, use_container_width=True)

            st.subheader("🏅 Principais Habilidades Demandadas")
            top10 = freq_filtrado.head(10).sort_values(ascending=True).reset_index()
            top10.columns = ["Habilidade", "Frequência"]
            fig2 = px.bar(top10, x="Frequência", y="Habilidade",
                          orientation="h", color="Frequência",
                          color_continuous_scale="Tealgrn", text="Frequência")
            fig2.update_layout(height=400, margin=dict(l=100, r=20, t=30, b=30))
            st.plotly_chart(fig2, use_container_width=True)

    if "Estado_UF" in df_filt:
        medias_uf = df_filt.groupby("Estado_UF")["Salario_num"]\
                           .mean().dropna().sort_values(ascending=False).reset_index()
        st.subheader("💰 Salário Médio por UF")
        st.plotly_chart(px.bar(medias_uf, x="Estado_UF", y="Salario_num",
                               color="Salario_num", color_discrete_sequence=["#2b83ba"]),
                        use_container_width=True)


# ===========================================================
# 💼 VAGAS
# ===========================================================
elif menu == "💼 Vagas":
    st.header("💼 Vagas (Detalhes)")
    if df_filt.empty:
        st.info("Nenhuma vaga encontrada com os filtros atuais.")
    else:
        vagas_por_pagina = 10
        total_vagas = len(df_filt)
        total_paginas = (total_vagas - 1) // vagas_por_pagina + 1
        pagina = st.number_input("Página", 1, total_paginas, 1)
        inicio, fim = (pagina - 1) * vagas_por_pagina, pagina * vagas_por_pagina
        st.caption(f"Mostrando vagas {inicio+1}–{min(fim,total_vagas)} de {total_vagas}")
        for _, r in df_filt.iloc[inicio:fim].iterrows():
            titulo = str(r.get("Titulo da Vaga") or "").strip() or "Não Informado"
            st.subheader(titulo)
            st.write(f"**Empresa:** {r.get('Empresa','-')}  **Local:** {r.get('Localizacao','-')}")

            # 💡 Corrigido: separar Tipo e Salário em linhas diferentes
            st.write(f"**Tipo:** {r.get('Tipo de Vaga','-')}")
            st.write(f"**Salário:** {r.get('Salario','-')}")

            st.write(f"**Horário:** {r.get('Horario','-')}")
            st.write(f"**Habilidades:** {r.get('Habilidades','-')}")
            st.divider()


# ===========================================================
# OUTRAS ABAS
# ===========================================================
elif menu == "🧾 CAGED":
    st.header("🧾 CAGED (Parquet)")
    nome = st.selectbox("Selecione Dataset CAGED", [k for k in DATASETS if "CAGED" in k])
    if st.button("🚀 Carregar CAGED"):
        df_c = carregar_parquet(DATASETS[nome], sample=10000)
        if not df_c.empty:
            st.dataframe(df_c.head(), use_container_width=True)
            st.success(f"✅ {len(df_c):,} linhas carregadas")

elif menu == "📊 RAIS":
    st.header("📊 RAIS (Parquet)")
    nome = [k for k in DATASETS if "RAIS" in k][0]
    if st.button("🚀 Carregar RAIS"):
        df_r = carregar_parquet(DATASETS[nome], sample=10000)
        if not df_r.empty:
            st.dataframe(df_r.head(), use_container_width=True)
            st.success(f"✅ {len(df_r):,} linhas carregadas")

elif menu == "📈 Análises Avançadas":
    st.header("📈 Análises Avançadas – Firebase")
    if "Estado_UF" in df_filt:
        med = df_filt.groupby("Estado_UF")["Salario_num"].mean().dropna().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(med, x="Estado_UF", y="Salario_num",
                               color="Salario_num", color_discrete_sequence=["#e37222"]),
                        use_container_width=True)
    if "Empresa" in df_filt:
        top_emp = df_filt["Empresa"].dropna().replace("", "Não Informada").value_counts().head(10).reset_index()
        top_emp.columns = ["Empresa", "Total Vagas"]
        st.plotly_chart(px.bar(top_emp, x="Empresa", y="Total Vagas",
                               color="Total Vagas", color_discrete_sequence=["#76b7b2"]),
                        use_container_width=True)


# ===========================================================
# Rodapé
# ===========================================================
st.markdown(f"""
---
💡 **Notas**
- Filtro de *Horário* agora lista todas as descrições vindas da própria coluna 'Horario' no Firebase e permite escolher livremente quais ocultar.  
- “🏢 Empresas que Mais Contratam” reestilizado e ignora “Não Informada”.  
- Nuvem e gráfico de habilidades permitem remover termos manualmente.  
- Paginação de vagas = 10 por página.  
- PyArrow ativo? → **{USE_PYARROW}**
""")