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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import numpy as np # Adicionado para RAIS/numpy

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

    # ---------- Normalização de nomes de colunas ----------
    df.columns = [re.sub(r"[\s_]+", " ", c).strip().title() for c in df.columns]
    renomeios = {
        "Título Da Vaga": "Titulo Da Vaga",
        "Salário": "Salario",
        "Localização": "Localizacao",
        "Tipo De Vaga": "Tipo De Vaga",
    }
    df.rename(columns=renomeios, inplace=True)
    # Garante presença da coluna "Tipo De Vaga" qualquer que seja o nome original
    possiveis_nom = ["Tipo De Vaga", "Tipo Vaga", "Tipo Da Vaga", "Tipo_vaga", "Tipo_vaga", "Tipo De  Vaga"]
    for alt in possiveis_nom:
        if alt in df.columns and alt != "Tipo De Vaga":
            if "Tipo De Vaga" not in df.columns:
                df["Tipo De Vaga"] = df[alt]
            else:
                df["Tipo De Vaga"] = df["Tipo De Vaga"].fillna(df[alt])

    df = df.loc[:, ~df.columns.duplicated()]

    if "Localizacao" in df.columns:
        df["Localizacao"] = df["Localizacao"].astype(str).str.strip()

    # ---------- Limpeza: Título ----------
    def limpar_titulo(v):
        if pd.isna(v):
            return pd.NA
        t = str(v).strip()
        if t.lower() in ["", "nan", "none", "n/a", "não informado", "nao informado"]:
            return pd.NA
        return t
    if "Titulo Da Vaga" in df.columns:
        df["Titulo Da Vaga"] = df["Titulo Da Vaga"].apply(limpar_titulo)

    # ---------- Limpeza: Tipo de Vaga ----------
    def limpar_tipo(v):
        if pd.isna(v):
            return pd.NA
        t = str(v).strip()
        if t.lower() in ["", "nan", "none", "n/a", "não informado", "nao informado"]:
            return pd.NA
        if t.isupper():  # preserva siglas
            return t
        return t.capitalize()
    if "Tipo De Vaga" in df.columns:
        df["Tipo De Vaga"] = df["Tipo De Vaga"].apply(limpar_tipo)

    # ---------- Limpeza: Habilidades ----------
    def limpar_habilidades(v):
        if not v or str(v).strip().lower() in ["nan", "none", "não informado"]:
            return ""
        parts = re.split(r"[,;/]", str(v))
        parts = [p.strip().title() for p in parts if p.strip()]
        return ", ".join(sorted(set(parts)))
    df["Habilidades"] = df.get("Habilidades", "").apply(limpar_habilidades)

    # ---------- Extração: Salário Numérico ----------
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

    # ---------- UF ----------
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
    "CAGED 2025-01":"https://drive.google.com/file/d/1S9swGQ51Oy7tUTK-d4iyHMSCIMMizec9/view?usp=drive_link",
    "CAGED 2025-02":"https://drive.google.com/file/d/1xQp9JtSkOpqIcnKByDq7huIsSArm6JX9/view?usp=drive_link",
    "CAGED 2025-03":"https://drive.google.com/file/d/1vxjzWcqWAhx1Eow2fZYtuuLsDe49HJg4/view?usp=drive_link",
    "CAGED 2025-04":"https://drive.google.com/file/d/1dVB-uwH2QZv9ZA4CYbHlWBCfelc1KMFA/view?usp=drive_link",
    "CAGED 2025-05":"https://drive.google.com/file/d/1_mJelDso0EvMVmeTfZtkASJZVTmA0ncF/view?usp=drive_link",
    "CAGED 2025-06":"https://drive.google.com/file/d/1pde7nKTRp4V3YEUz2KdUd3mtTyBgni6y/view?usp=drive_link",
    "CAGED 2025-07":"https://drive.google.com/file/d/11L3vsgLLXNTokZ9eojRzPt5mWNNmv9MQ/view?usp=drive_link",
    "RAIS 2023":"https://drive.google.com/file/d/13lEIFqkdtZ6FPZ4be8qcQnCww1H58IgV/view"
}

DICTS = {
    "CBO 2002": "https://drive.google.com/file/d/1ftYSYIlN0NzxAFF5y2K3lIag-mJWfB96/view?usp=drive_link"
}

# ===========================================================
# 4️⃣ Inicializar e Criar Filtros
# ===========================================================
# Substitua o caminho abaixo pelo seu caminho local real
cred_path_firebase = r"C:\Users\thiag\Documents\ASCENDE-DASHBOARD\ascende-firebase.json" 
df_vagas = carregar_dados_firebase(cred_path_firebase)
if df_vagas.empty:
    st.warning("Nenhuma vaga encontrada.")
    st.stop()

# ======== Filtros laterais fixos =========
st.sidebar.header("📊 Filtros")

tipos = sorted(df_vagas.get("Tipo De Vaga", pd.Series([], dtype=str)).dropna().unique().tolist())
cidades = sorted(df_vagas.get("Localizacao", pd.Series([], dtype=str)).dropna().unique().tolist())
habs = sorted(
    h for h in df_vagas.get("Habilidades", pd.Series([], dtype=str))
      .dropna().astype(str).str.split(",").explode().str.strip().unique() if h
)

sal_min, sal_max = 0, 12000
faixa = st.sidebar.slider("Faixa Salarial (R$)", sal_min, sal_max, (sal_min, sal_max), step=100)

st.sidebar.write("🧹 **Ocultar tipos de salário (baseado nos dados):**")

def obter_descricoes_salario(df):
    if "Salario" not in df.columns:
        return []
    descricoes = (
        df["Salario"]
        .dropna().astype(str).str.strip()
        .replace("", pd.NA)
        .dropna().unique()
    )
    resultado = []
    for v in descricoes:
        texto = str(v).strip()
        if re.search(r"\d", texto):
            continue
        resultado.append(texto)
    return sorted(set(resultado))

opcoes_salario_texto = obter_descricoes_salario(df_vagas)
if not opcoes_salario_texto:
    st.sidebar.info("Nenhuma descrição textual de salário encontrada no Firebase.")
salarios_indesejados = st.sidebar.multiselect(
    "Selecione descrições de salário que deseja ocultar:",
    options=opcoes_salario_texto,
    default=[],
    help="Essas opções vêm das descrições textuais atuais da coluna 'Salario' do Firebase."
)

tipo_sel = st.sidebar.multiselect("Tipo de Vaga", tipos)
cidade_sel = st.sidebar.multiselect("Localização", cidades)
hab_sel = st.sidebar.multiselect("Habilidades", habs)


# ===========================================================
# 🕒 Filtro de Horário — flexível
# ===========================================================
st.sidebar.markdown("🕒 **Horário personalizado**")

col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    hora_inicio = st.time_input("Começa às", value=pd.Timestamp("08:00").time())
with col_h2:
    hora_fim = st.time_input("Termina às", value=pd.Timestamp("18:00").time())

ativar_filtro_indefinido = st.sidebar.toggle("Ativar remoção de horários específicos", value=False)

def obter_opcoes_horario(df: pd.DataFrame) -> list[str]:
    if "Horario" not in df.columns:
        return []
    horarios = (
        df["Horario"]
        .dropna().astype(str).str.strip()
        .replace("", pd.NA)
        .dropna().unique().tolist()
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
# Função de filtragem
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

    if salarios_indesejados and "Salario" in df_f.columns:
        padrao = "|".join(map(re.escape, [s.lower() for s in salarios_indesejados]))
        df_f = df_f[
            ~df_f["Salario"].astype(str).str.lower().str.contains(padrao, regex=True, na=False)
        ]

    if tipo_sel:
        df_f = df_f[df_f["Tipo De Vaga"].isin(tipo_sel)]
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
# Navegação
# ===========================================================
st.sidebar.title("🧭 Navegação")
menu = st.sidebar.radio("Ir para:", ["🏠 Home", "💼 Vagas", "🧾 CAGED"])

st.title("ASCENDE — Dashboard de Oportunidades em TI")

# ---
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

    # ===========================================================
    # 🧩 Distribuição por Tipo de Vaga
    # ===========================================================
    if "Tipo De Vaga" in df_filt:
        tipo_freq = df_filt["Tipo De Vaga"].dropna().value_counts().reset_index()
        tipo_freq.columns = ["Tipo De Vaga", "Vagas"]
        st.subheader("🧩 Distribuição por Tipo de Vaga")
        fig = px.pie(tipo_freq, names="Tipo De Vaga", values="Vagas",
                     hole=0.45, color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(textinfo="percent+label", pull=[0.03]*len(tipo_freq))
        st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # 🏢 Empresas que Mais Contratam
    # ===========================================================
    if "Empresa" in df_filt:
        st.subheader("🏢 Empresas que Mais Contratam")
        top_emp = (
            df_filt["Empresa"]
            .dropna().replace("", pd.NA)
            .replace("Não Informada", pd.NA)
            .dropna().value_counts()
            .head(10).reset_index()
        )
        top_emp.columns = ["Empresa", "Total Vagas"]

        if not top_emp.empty:
            top_emp = top_emp.sort_values(by="Total Vagas", ascending=True)
            fig_emp = px.bar(top_emp, x="Total Vagas", y="Empresa",
                orientation="h", color="Total Vagas",
                color_continuous_scale=px.colors.sequential.Tealgrn,
                text="Total Vagas")
            fig_emp.update_layout(height=480, margin=dict(l=120, r=40, t=40, b=40),
                plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Número de Vagas", yaxis_title="", coloraxis_showscale=False)
            fig_emp.update_traces(texttemplate="%{text}", textposition="outside",
                marker_line_color="teal", marker_line_width=1)
            st.plotly_chart(fig_emp, use_container_width=True)
        else:
            st.info("Nenhuma empresa informada disponível para exibir.")

    # ===========================================================
    # ☁️ Nuvem de Habilidades + Top 10
    # ===========================================================
    if "Habilidades" in df_filt:
        todas = df_filt["Habilidades"].dropna().astype(str).str.split(",").explode().str.strip().str.title()
        todas = todas[todas != ""]
        freq = todas.value_counts()
        if not freq.empty:
            st.subheader("☁️ Nuvem de Habilidades")
            todas_habs = sorted(freq.index.tolist())
            remover = st.multiselect("Remover habilidades da visualização:", todas_habs)
            freq_filtrado = freq.drop(labels=remover, errors="ignore")

            wc = WordCloud(width=800, height=300, background_color="white", max_words=80)
            wc.generate_from_frequencies(freq_filtrado.to_dict())
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig, use_container_width=True)

            st.subheader("🏅 Principais Habilidades Demandadas")
            top10 = freq_filtrado.head(10).sort_values(ascending=True).reset_index()
            top10.columns = ["Habilidade", "Frequência"]
            fig2 = px.bar(top10, x="Frequência", y="Habilidade", orientation="h",
                          color="Frequência", color_continuous_scale="Tealgrn", text="Frequência")
            fig2.update_layout(height=400, margin=dict(l=100, r=20, t=30, b=30))
            st.plotly_chart(fig2, use_container_width=True)

    # ===========================================================
    # 💰 Salário Médio por UF
    # ===========================================================
    if "Estado_UF" in df_filt:
        medias_uf = df_filt.groupby("Estado_UF")["Salario_num"].mean().dropna().sort_values(ascending=False).reset_index()
        st.subheader("💰 Salário Médio por UF")
        st.plotly_chart(px.bar(medias_uf, x="Estado_UF", y="Salario_num",
                                 color="Salario_num", color_discrete_sequence=["#2b83ba"]),
                         use_container_width=True)
    else:
        st.info("Nenhum salário disponível para exibir.")

    # ===========================================================
    # 🗺️ Top 10 Estados com Mais Vagas – NOVO GRÁFICO
    # ===========================================================
    if "Estado_UF" in df_filt.columns:
        st.subheader("🗺️ Top 10 Estados com Mais Vagas de TI")

        uf_top = (
            df_filt["Estado_UF"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )
        uf_top.columns = ["UF", "Vagas"]

        if not uf_top.empty:
            fig_uf = px.bar(
                uf_top.sort_values("Vagas"),
                x="Vagas",
                y="UF",
                orientation="h",
                color="Vagas",
                color_continuous_scale="Tealgrn",
                text="Vagas",
                title="🗺️ Top 10 Estados com Mais Vagas de TI",
            )

            fig_uf.update_traces(
                textposition="outside",
                marker_line_color="#117a65",
                marker_line_width=1.2,
                opacity=0.9,
            )

            fig_uf.update_layout(
                height=500,
                margin=dict(l=120, r=40, t=70, b=50),
                plot_bgcolor="#ffffff",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                xaxis_title="Número de Vagas de TI",
                yaxis_title="Estado (UF)",
                coloraxis_showscale=False,
            )

            st.plotly_chart(fig_uf, use_container_width=True)
            # Botão de download removido

        else:
            st.info("Nenhum dado válido de UF encontrado para gerar o gráfico.")

    # ===========================================================
    # 🌐 Origens das Vagas (NOVO GRÁFICO)
    # ===========================================================

    if "Link Da Vaga" in df_filt.columns:
        st.subheader("🌐 Origens das Vagas (TI)")

        def extrair_dominio(link):
            try:
                dominio = urlparse(str(link)).netloc.lower()
                dominio = dominio.replace("www.", "").strip()
                dominio = re.sub(r":\d+$", "", dominio)
                dominio = dominio.split("/")[0]

                if re.search(r"\.(com|org|gov|edu)\.br$", dominio):
                    partes = dominio.split(".")
                    dominio = ".".join(partes[-3:])
                else:
                    partes = dominio.split(".")
                    dominio = ".".join(partes[-2:]) if len(partes) >= 2 else dominio
                return dominio
            except Exception:
                return None

        dominios = df_filt["Link Da Vaga"].dropna().apply(extrair_dominio)
        top_dominios = dominios.value_counts().head(10).reset_index()
        top_dominios.columns = ["Origem", "Vagas"]

        if not top_dominios.empty:
            fig_dom = px.pie(
                top_dominios,
                names="Origem",
                values="Vagas",
                title="🌐 Origem das Vagas de TI",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Vivid
            )

            fig_dom.update_traces(
                textinfo="label+percent",
                textfont_size=14,
                hovertemplate="%{label}: %{value:,} vagas (%{percent})",
                pull=[0.03] * len(top_dominios)
            )

            fig_dom.update_layout(
                height=530,
                margin=dict(l=50, r=50, t=80, b=50),
                showlegend=True,
                legend_title="Origem da Vaga",
                plot_bgcolor="#ffffff",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
            )

            st.plotly_chart(fig_dom, use_container_width=True)
            # Botão de download removido
        else:
            st.info("Nenhum link de vaga válido encontrado para gerar o gráfico.")

# ---
# ===========================================================
# 💼 VAGAS
# ===========================================================
elif menu == "💼 Vagas":
    st.header("💼 Vagas (Detalhes)")

    if df_filt.empty:
        st.info("Nenhuma vaga encontrada com os filtros atuais.")
    else:
        def limpar_valor(v, padrao="-"):
            if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
                return padrao
            texto = str(v).strip()
            return texto if texto else padrao

        vagas_por_pagina = 10
        total_vagas = len(df_filt)
        total_paginas = (total_vagas - 1) // vagas_por_pagina + 1

        # 🔹 Controle de página via sessão — mantém posição ao mudar filtros
        if "pagina" not in st.session_state:
            st.session_state.pagina = 1

        cols = st.columns(12)

        # Botão voltar <
        if cols[0].button("<"):
            st.session_state.pagina = max(1, st.session_state.pagina - 1)

        current = st.session_state.pagina

        # Lógica para renderizar botões inteligentemente (1… atual ±1, última)
        pages_to_show = {
            1, 2, total_paginas,
            current, current - 1, current + 1
        }
        pages_to_show = [p for p in sorted(pages_to_show) if 1 <= p <= total_paginas]

        last_rendered = None
        col_idx = 1

        for p in pages_to_show:
            if last_rendered and p - last_rendered > 1:
                cols[col_idx].markdown("**…**")
                col_idx += 1

            if p == current:
                cols[col_idx].markdown(
                    f"<div style='background:#4A8CF7;color:white;"
                    f"padding:6px 10px;border-radius:4px;text-align:center;"
                    f"font-weight:bold'>{p}</div>",
                    unsafe_allow_html=True
                )
            else:
                if cols[col_idx].button(str(p), key=f"page_{p}"):
                    st.session_state.pagina = p
            last_rendered = p
            col_idx += 1

        # Botão avançar >
        if cols[min(col_idx, 11)].button(">"):
            st.session_state.pagina = min(total_paginas, st.session_state.pagina + 1)

        # Intervalo das vagas visíveis
        pagina = st.session_state.pagina
        inicio, fim = (pagina - 1) * vagas_por_pagina, pagina * vagas_por_pagina

        st.caption(f"Mostrando vagas {inicio+1}–{min(fim, total_vagas)} de {total_vagas}")

        # 🔹 Exibição das vagas atuais
        for _, r in df_filt.iloc[inicio:fim].iterrows():
            titulo = limpar_valor(r.get("Titulo Da Vaga"), "Não Informado")
            st.subheader(titulo)
            empresa = limpar_valor(r.get("Empresa"))
            local = limpar_valor(r.get("Localizacao"))
            tipo = limpar_valor(r.get("Tipo De Vaga"))
            salario = limpar_valor(r.get("Salario"))
            horario = limpar_valor(r.get("Horario"))
            habilidades = limpar_valor(r.get("Habilidades"))

            st.markdown(f"**🏢 Empresa:** {empresa} **📍 Local:** {local}")
            st.markdown(f"**📁 Tipo:** {tipo} **💰 Salário:** {salario}")
            st.markdown(f"**⏰ Horário:** {horario}")
            st.markdown(f"**🧩 Habilidades:** {habilidades}")
            st.divider()

# ---
# ===========================================================
# 🧾 CAGED
# ===========================================================

elif menu == "🧾 CAGED":
    st.subheader("🧠 Ocupações de TI — Análise Dinâmica")

    ARQUIVOS_TI = [DATASETS[k] for k in DATASETS if "CAGED" in k]
    CBO_LINK = DICTS["CBO 2002"]

    # O código abaixo só é executado após o clique no botão
    if st.button("📊 Gerar Gráfico de Ocupações de TI"):
        # ===========================================================
        # 1️⃣ Baixar todos os arquivos CAGED (Parquet)
        # ===========================================================
        dfs = [carregar_parquet(url) for url in ARQUIVOS_TI]
        df = pd.concat(dfs, ignore_index=True)

        if df.empty:
            st.error("Não foi possível carregar dados do CAGED. Verifique os links ou a estrutura dos arquivos.")
            st.stop()

        # ===========================================================
        # 2️⃣ Dicionário CBO 2002 (Carregamento garantido)
        # ===========================================================
        try:
            # Lógica para baixar o CBO
            fid = CBO_LINK.split("/d/")[1].split("/")[0]
            cbo_url = f"https://drive.google.com/uc?export=download&id={fid}"
            cbo = pd.read_csv(cbo_url, sep=";", encoding="latin1", dtype=str)
            cbo.columns = ["CODIGO", "TITULO"]
            cbo["CODIGO"] = cbo["CODIGO"].str.strip().str.zfill(6)
        except Exception as e:
            st.error(f"Erro ao carregar dicionário CBO: {e}")
            cbo = pd.DataFrame(columns=["CODIGO", "TITULO"]) # Fallback
            st.stop()
            
        # ===========================================================
        # 5️⃣ Lista de códigos TI (Ocupações de Interesse)
        # ===========================================================
        lista_cbo_ti = [
            "142505","142510","142515","142520","142525","142530","142535",
            "212205","212210","212215","212305","212310","212315","212320",
            "212405","212410","212415","212420","212425","212430",
            "317105","317110","317120","317205","317210",
        ]


        # ===========================================================
        # 3️⃣ Apenas admissões e desligamentos (preparação)
        # ===========================================================
        if "saldomovimentacao" in df.columns:
            df_adm = df[df["saldomovimentacao"] > 0].copy()
            df_desl = df[df["saldomovimentacao"] < 0].copy()
        elif "tipomovimentacao" in df.columns:
            df_adm = df[df["tipomovimentacao"].isin([1, 31, 32, 33])].copy()
            df_desl = df[~df["tipomovimentacao"].isin([1, 31, 32, 33])].copy()
        else:
            st.error("Nenhuma coluna de movimentação ('saldomovimentacao' ou 'tipomovimentacao') encontrada.")
            st.stop()

        # ===========================================================
        # 4️⃣ Merge com CBO (para obter o TÍTULO da ocupação)
        # ===========================================================
        if not cbo.empty:
            df["cbo2002ocupacao"] = df["cbo2002ocupacao"].astype(str).str.zfill(6)
            df_adm["cbo2002ocupacao"] = df_adm["cbo2002ocupacao"].astype(str).str.zfill(6)
            df_desl["cbo2002ocupacao"] = df_desl["cbo2002ocupacao"].astype(str).str.zfill(6)

            df_adm = df_adm.merge(cbo, left_on="cbo2002ocupacao", right_on="CODIGO", how="left")
            df_desl = df_desl.merge(cbo, left_on="cbo2002ocupacao", right_on="CODIGO", how="left")
        else:
             st.warning("O dicionário CBO está vazio. Os gráficos serão gerados por código numérico.")


        # ===========================================================
        # 📊 Gráfico 1 — Top 10 Admissões
        # ===========================================================
        df_ti = df_adm[df_adm["CODIGO"].isin(lista_cbo_ti)].copy()
        top10_adm = (
            df_ti.groupby("TITULO")
            .size()
            .reset_index(name="Total_Admissoes")
            .sort_values("Total_Admissoes", ascending=False)
            .head(10)
        )
        st.subheader("📈 Top 10 Ocupações de TI – Admissões CAGED")
        if not top10_adm.empty:
            fig_adm = px.bar(
                top10_adm.sort_values("Total_Admissoes"),
                x="Total_Admissoes",
                y="TITULO",
                orientation="h",
                color="Total_Admissoes",
                color_continuous_scale="Blues",
                title="📈 Top 10 Ocupações de TI – Admissões CAGED",
            )
            fig_adm.update_layout(height=500, margin=dict(l=80, r=40, t=60, b=30))
            fig_adm.update_traces(texttemplate="%{x}", textposition="outside")
            st.plotly_chart(fig_adm, use_container_width=True)
            # Download REMOVIDO
        else:
             st.info("Nenhuma admissão de TI encontrada no conjunto CAGED.")

        # ===========================================================
        # 📉 Gráfico 2 — Top 10 Desligamentos
        # ===========================================================
        df_desl_ti = df_desl[df_desl["CODIGO"].isin(lista_cbo_ti)].copy()
        st.subheader("📉 Top 10 Ocupações de TI – Desligamentos CAGED")
        if not df_desl_ti.empty:
            top10_desl = (
                df_desl_ti.groupby("TITULO")
                .size()
                .reset_index(name="Total_Desligamentos")
                .sort_values("Total_Desligamentos", ascending=False)
                .head(10)
            )

            fig_desl = px.bar(
                top10_desl.sort_values("Total_Desligamentos"),
                x="Total_Desligamentos",
                y="TITULO",
                orientation="h",
                color="Total_Desligamentos",
                color_continuous_scale="Reds",
                title="📉 Top 10 Ocupações de TI – Desligamentos CAGED",
            )
            fig_desl.update_layout(height=500, margin=dict(l=80, r=40, t=60, b=30))
            fig_desl.update_traces(texttemplate="%{x}", textposition="outside")
            st.plotly_chart(fig_desl, use_container_width=True)
            # Download REMOVIDO
        else:
            st.info("Nenhum dado de desligamento de TI encontrado no conjunto CAGED.")


        # ===========================================================
        # ⚖️ Gráfico 3 — Saldo Líquido de Empregos de TI
        # ===========================================================
        if "saldomovimentacao" in df.columns:
            df_saldo_base = df.copy()
            df_saldo_ti = df_saldo_base[df_saldo_base["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
            
            if not cbo.empty and "TITULO" not in df_saldo_ti.columns:
                df_saldo_ti = df_saldo_ti.merge(cbo, left_on="cbo2002ocupacao", right_on="CODIGO", how="left")

            saldo_ti = (
                df_saldo_ti.groupby("TITULO")["saldomovimentacao"]
                .sum()
                .reset_index(name="Saldo_Emprego")
                .sort_values("Saldo_Emprego", ascending=False)
            )

            st.subheader("⚖️ Saldo Líquido de Empregos – Ocupações de TI (CAGED)")

            if not saldo_ti.empty:
                fig_saldo = px.bar(
                    saldo_ti,
                    x="Saldo_Emprego",
                    y="TITULO",
                    orientation="h",
                    color="Saldo_Emprego",
                    color_continuous_scale=["crimson", "lightgreen"],
                    title="⚖️ Saldo Líquido de Empregos – Ocupações de TI (CAGED)",
                )
                fig_saldo.update_layout(height=500, margin=dict(l=80, r=40, t=60, b=30))
                fig_saldo.update_traces(texttemplate="%{x}", textposition="outside")
                st.plotly_chart(fig_saldo, use_container_width=True)
                # Download REMOVIDO
            else:
                 st.info("Nenhum dado de Saldo Líquido de Empregos em TI encontrado.")

        else:
            st.info("❕ O conjunto CAGED carregado não contém 'saldomovimentacao' para calcular o Saldo Líquido.")

        # ===========================================================
        # 📅 Gráfico 4 — Admissões × Desligamentos por Mês (CAGED TI)
        # ===========================================================
        if "competenciamov" in df.columns and "saldomovimentacao" in df.columns:
            st.subheader("📅 Admissões x Desligamentos de TI – Mensal (CAGED)")
            
            df["competenciamov"] = df["competenciamov"].astype(str).str.strip()
            df["Mes"] = pd.to_datetime(df["competenciamov"].str[:6], format="%Y%m", errors="coerce")
            df = df.dropna(subset=["Mes"])
            df["Mes_nome"] = df["Mes"].dt.strftime("%b/%Y")

            df["cbo2002ocupacao"] = df["cbo2002ocupacao"].astype(str).str.zfill(6)
            df_ti = df[df["cbo2002ocupacao"].isin(lista_cbo_ti)]

            adm_mes = (
                df_ti[df_ti["saldomovimentacao"] > 0]
                .groupby("Mes_nome")["saldomovimentacao"]
                .count()
                .reset_index(name="Admissoes")
            )
            des_mes = (
                df_ti[df_ti["saldomovimentacao"] < 0]
                .groupby("Mes_nome")["saldomovimentacao"]
                .count()
                .reset_index(name="Desligamentos")
            )

            serie = pd.merge(adm_mes, des_mes, on="Mes_nome", how="outer").fillna(0)
            serie["Mes_ord"] = pd.to_datetime(serie["Mes_nome"], format="%b/%Y", errors="coerce")
            serie = serie.sort_values("Mes_ord")

            serie_melt = serie.melt(
                id_vars=["Mes_nome", "Mes_ord"],
                value_vars=["Admissoes", "Desligamentos"],
                var_name="Tipo",
                value_name="Quantidade"
            )

            cores = {"Admissoes": "#2ECC71", "Desligamentos": "#E74C3C"}

            fig_mes = px.bar(
                serie_melt,
                x="Mes_nome",
                y="Quantidade",
                color="Tipo",
                barmode="group",
                text_auto=".0f",
                color_discrete_map=cores,
                title="📅 Admissões × Desligamentos de TI – CAGED (Mês a Mês)",
            )

            fig_mes.update_layout(
                height=620,
                margin=dict(l=60, r=40, t=80, b=150),
                xaxis_title="Mês de Movimentação",
                yaxis_title="Quantidade de Movimentações",
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1
                ),
                plot_bgcolor="#ffffff", 
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000000"),
                bargap=0.28,
                xaxis=dict(
                    tickangle=-60,
                    tickfont=dict(size=12, color="#000000"), 
                    showgrid=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.15)",
                    tickfont=dict(size=12, color="#111111")
                )
            )

            fig_mes.update_traces(
                marker_line_width=0,
                textfont=dict(size=13, color="#111111"),
                textposition="outside",
                cliponaxis=False,
                opacity=0.95
            )

            st.plotly_chart(fig_mes, use_container_width=True)

            # Download REMOVIDO
        else:
            st.info("⚠️ O conjunto CAGED não possui colunas 'competenciamov' e/ou 'saldomovimentacao' para este gráfico.")
            
        # ===========================================================
        # 🗺️ Gráfico 5 — Top 10 Estados com Mais Admissões de TI (CAGED)
        # ===========================================================
        if "uf" in df.columns and "saldomovimentacao" in df.columns:
            
            df_ti = df[df["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
            df_adm_ti = df_ti[df_ti["saldomovimentacao"] > 0].copy()

            mapa_uf = {
                11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
                21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
                28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
                42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"
            }
            df_adm_ti["UF"] = df_adm_ti["uf"].map(mapa_uf)

            top_estados = (
                df_adm_ti["UF"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            top_estados.columns = ["UF", "Admissoes"]

            fig_uf = px.bar(
                top_estados.sort_values("Admissoes"),
                x="Admissoes",
                y="UF",
                orientation="h",
                text="Admissoes",
                color="Admissoes",
                color_continuous_scale="Tealgrn",
                title="🗺️ Top 10 Estados com Mais Admissões de TI (CAGED)",
            )

            fig_uf.update_layout(
                height=580,
                margin=dict(l=120, r=40, t=70, b=40),
                plot_bgcolor="#ffffff",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                xaxis_title="Número de Admissões em TI",
                yaxis_title="Estado (UF)",
                coloraxis_showscale=False,
            )

            fig_uf.update_traces(
                texttemplate="%{text}",
                textposition="outside",
                marker_line_color="#117a65",
                marker_line_width=1.2,
                opacity=0.9,
            )

            st.plotly_chart(fig_uf, use_container_width=True)

            # Download REMOVIDO
        else:
            st.info("⚠️ O conjunto CAGED não contém as colunas 'uf' e/ou 'saldomovimentacao'.")
            
        # ===========================================================
        # 👶👨‍💻👴 Gráfico 6 — Faixa Etária (5 em 5 anos) das Admissões em TI (CAGED)
        # ===========================================================
        if "idade" in df.columns and "saldomovimentacao" in df.columns:
            
            df_ti = df[df["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
            df_adm_ti = df_ti[df_ti["saldomovimentacao"] > 0].copy()

            df_adm_ti = df_adm_ti[(df_adm_ti["idade"] >= 15) & (df_adm_ti["idade"] <= 100)]

            bins = [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 120]
            labels = [
                "18–20 anos", "21–25 anos", "26–30 anos", "31–35 anos",
                "36–40 anos", "41–45 anos", "46–50 anos", "51–55 anos",
                "56–60 anos", "61–65 anos", "66 ou +",
            ]
            df_adm_ti["Faixa_Idade"] = pd.cut(df_adm_ti["idade"], bins=bins, labels=labels, right=True, include_lowest=True)

            dist_idade = (
                df_adm_ti["Faixa_Idade"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            dist_idade.columns = ["Faixa", "Admissoes"]

            fig_idade = px.bar(
                dist_idade,
                x="Faixa",
                y="Admissoes",
                text="Admissoes",
                color="Admissoes",
                color_continuous_scale="Sunsetdark",
                title="👶 👨‍💻 👴 Distribuição Etária das Admissões em TI (CAGED)",
            )

            fig_idade.update_traces(
                textposition="outside",
                marker_line_color="#6C3483",
                marker_line_width=1.1,
                opacity=0.9,
            )

            fig_idade.update_layout(
                height=540,
                margin=dict(l=60, r=40, t=70, b=100),
                plot_bgcolor="#ffffff",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                xaxis_title="Faixa Etária (anos)",
                yaxis_title="Número de Admissões em TI",
                coloraxis_showscale=False,
                xaxis=dict(tickangle=-40)
            )

            st.plotly_chart(fig_idade, use_container_width=True)

            # Download REMOVIDO
        else:
            st.info("⚠️ O conjunto CAGED não contém as colunas 'idade' ou 'saldomovimentacao'.")
            
        # ===========================================================
        # 🚻 Gráfico — Distribuição de Gênero nas Admissões de TI (CAGED)
        # ===========================================================
        if "sexo" in df.columns and "saldomovimentacao" in df.columns:
            
            df_ti = df[df["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
            df_adm_ti = df_ti[df_ti["saldomovimentacao"] > 0].copy()

            df_adm_ti["sexo"] = df_adm_ti["sexo"].astype(str).str.strip().str.upper()

            mapa_genero = {
                "1": "Masculino", "2": "Feminino", "3": "Feminino",
                "M": "Masculino", "MASCULINO": "Masculino", "HOMEM": "Masculino",
                "F": "Feminino", "FEMININO": "Feminino", "MULHER": "Feminino",
                "NAN": "Feminino", "NONE": "Feminino", "0": "Feminino", "": "Feminino",
            }

            df_adm_ti["Genero"] = (
                df_adm_ti["sexo"].map(mapa_genero).fillna("Feminino")
            )

            genero_counts = (
                df_adm_ti["Genero"]
                .value_counts()
                .reset_index()
            )
            genero_counts.columns = ["Genero", "Admissoes"]

            fig_genero = px.pie(
                genero_counts,
                names="Genero",
                values="Admissoes",
                color="Genero",
                color_discrete_map={
                    "Masculino": "#1f77b4",  # azul
                    "Feminino": "#ff69b4",  # rosa
                },
                title="🚻 Distribuição de Gênero nas Admissões de TI (CAGED)",
                hole=0.35,
            )

            fig_genero.update_traces(
                textinfo="label+percent",
                textfont_size=13,
                hovertemplate="%{label}: %{value:,} admissões (%{percent})"
            )

            fig_genero.update_layout(
                height=520,
                margin=dict(l=60, r=60, t=80, b=80),
                showlegend=True,
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_genero, use_container_width=True)

            # Download REMOVIDO
        else:
            st.info("⚠️ O conjunto CAGED não possui as colunas 'sexo' e/ou 'saldomovimentacao'.")

        # ===========================================================
        # 🎓 Gráfico — Escolaridade nas Admissões de TI (CAGED)
        # ===========================================================
        if "graudeinstrucao" in df.columns and "saldomovimentacao" in df.columns:
            
            df_ti = df[df["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
            df_adm_ti = df_ti[df_ti["saldomovimentacao"] > 0].copy()

            mapa_instrucao = {
                1: "Analfabeto", 2: "Até 5ª Incompleto", 3: "5ª Completa", 4: "6ª–9ª Incompleto",
                5: "Fundamental Completo", 6: "Médio Incompleto", 7: "Médio Completo",
                8: "Superior Incompleto", 9: "Superior Completo", 10: "Mestrado", 11: "Doutorado",
            }

            df_adm_ti["GrauInstrucao"] = (
                pd.to_numeric(df_adm_ti["graudeinstrucao"], errors="coerce")
                .map(mapa_instrucao)
                .fillna("Não Informado")
            )

            esc_counts = df_adm_ti["GrauInstrucao"].value_counts().reset_index()
            esc_counts.columns = ["Escolaridade", "Admissoes"]

            esc_counts = esc_counts.sort_values("Admissoes", ascending=True)

            fig_esc = px.bar(
                esc_counts,
                x="Admissoes",
                y="Escolaridade",
                orientation="h",
                text="Admissoes",
                color="Admissoes",
                color_continuous_scale="Purples",
                title="🎓 Nível de Escolaridade nas Admissões de TI (CAGED)",
            )

            fig_esc.update_traces(
                textposition="outside",
                marker_line_color="#4A235A",
                marker_line_width=1.2,
                opacity=0.9,
            )

            fig_esc.update_layout(
                height=620,
                margin=dict(l=180, r=40, t=70, b=40),
                plot_bgcolor="#ffffff",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                xaxis_title="Número de Admissões em TI",
                yaxis_title="Grau de Instrução",
                coloraxis_showscale=False,
            )

            st.plotly_chart(fig_esc, use_container_width=True)

            # Download REMOVIDO
            
            # ===========================================================
            # ☁️ Gráfico — Nuvem de Palavras de Regiões (CAGED)
            # ===========================================================

            try:
                df_ti = df[df["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
                df_adm_ti = df_ti[df_ti["saldomovimentacao"] > 0].copy()

                mapa_regiao = {
                    1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro‑Oeste"
                }

                df_adm_ti["RegiaoTXT"] = pd.to_numeric(df_adm_ti["regiao"], errors="coerce").map(mapa_regiao)

                texto_regioes = " ".join(df_adm_ti["RegiaoTXT"].dropna().astype(str).tolist())

                if not texto_regioes.strip():
                    st.info("⚠️ Nenhum registro de região encontrado para gerar a nuvem de palavras.")
                else:
                    wc_regiao = WordCloud(
                        width=1000, height=400, background_color="white",
                        colormap="Greens", collocations=False, max_words=10
                    ).generate(texto_regioes)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wc_regiao, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title("☁️ Distribuição Regional das Admissões de TI (CAGED)", fontsize=14, pad=10)
                    st.pyplot(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao gerar nuvem de regiões: {e}")

        else:
            st.info("⚠️ O conjunto CAGED não possui as colunas 'graudeinstrucao' e/ou 'saldomovimentacao'.")

# ---
# ===========================================================
# 📊 RAIS
# ===========================================================
elif menu == "📊 RAIS":
    st.header("📊 RAIS – Distribuição Etária dos Empregos de TI (2023)")
    nome = [k for k in DATASETS if "RAIS" in k][0]

    if st.button("🚀 Carregar RAIS"):
        df_r = carregar_parquet(DATASETS[nome])

        if not df_r.empty:
            
            # --- 1. CARREGAR DICIONÁRIO CBO 2002 (CORREÇÃO DE ESCOPO) ---
            try:
                CBO_LINK = DICTS["CBO 2002"]
                fid = CBO_LINK.split("/d/")[1].split("/")[0]
                cbo_url = f"https://drive.google.com/uc?export=download&id={fid}"
                cbo = pd.read_csv(cbo_url, sep=";", encoding="latin1", dtype=str)
                cbo.columns = ["CODIGO", "TITULO"]
                cbo["CODIGO"] = cbo["CODIGO"].str.strip().str.zfill(6)
            except Exception as e:
                st.error(f"Erro ao carregar dicionário CBO: {e}. Alguns gráficos podem falhar.")
                cbo = pd.DataFrame(columns=["CODIGO", "TITULO"]) # Fallback
                
            # --- 2. LISTA DE CÓDIGOS TI ---
            lista_cbo_ti = [
                "142505","142510","142515","142520","142525","142530","142535",
                "212205","212210","212215","212305","212310","212315","212320",
                "212405","212410","212415","212420","212425","212430",
                "317105","317110","317120","317205","317210",
            ]
            
            if "cbo2002ocupacao" in df_r.columns:
                df_r["cbo2002ocupacao"] = df_r["cbo2002ocupacao"].astype(str).str.zfill(6)
            
            df_r_ti = df_r[df_r["cbo2002ocupacao"].isin(lista_cbo_ti)].copy()
            
            if not cbo.empty and "CODIGO" in cbo.columns:
                df_r_ti = df_r_ti.merge(cbo, left_on="cbo2002ocupacao", right_on="CODIGO", how="left")


            try:
                df_r_ti.rename(columns={"idade": "Idade"}, inplace=True, errors="ignore") 

                if "Idade" in df_r_ti.columns:

                    bins = [17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 120]
                    labels = [
                        "18–20 anos", "21–25 anos", "26–30 anos", "31–35 anos",
                        "36–40 anos", "41–45 anos", "46–50 anos",
                        "51–55 anos", "56–60 anos", "61–65 anos", "65+ anos"
                    ]

                    df_r_ti["Faixa_Idade"] = pd.cut(
                        df_r_ti["Idade"],
                        bins=bins,
                        labels=labels,
                        ordered=True
                    )

                    faixa_counts = (
                        df_r_ti.groupby("Faixa_Idade")
                        .size()
                        .reset_index(name="Empregos")
                        .sort_values("Faixa_Idade", key=lambda x: x.astype("category").cat.codes)
                    )

                    # ====== GRÁFICO (Faixa Etária) ======
                    fig_idade = px.bar(
                        faixa_counts,
                        x="Empregos",
                        y="Faixa_Idade",
                        orientation="h",
                        text="Empregos",
                        color="Empregos",
                        color_continuous_scale="Sunsetdark",
                        title="👶 Distribuição Etária dos Empregados de TI (RAIS 2023)",
                    )

                    fig_idade.update_traces(
                        textposition="outside",
                        marker_line_color="#6C3483",
                        marker_line_width=1.1,
                        opacity=0.9,
                    )

                    fig_idade.update_layout(
                        height=540,
                        margin=dict(l=100, r=40, t=70, b=60),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                        xaxis_title="Número de Empregos de TI",
                        yaxis_title="Faixa Etária (anos)",
                        coloraxis_showscale=False,
                        yaxis=dict(
                            categoryorder="array",
                            categoryarray=labels,
                            autorange="reversed"
                        )
                    )

                    st.plotly_chart(fig_idade, use_container_width=True)

                    # Download REMOVIDO

                else:
                    st.warning("⚠️ Coluna 'Idade' não encontrada no parquet da RAIS.")

            except Exception as e:
                st.error(f"Erro ao gerar gráfico de faixa etária: {e}")
            
            
            # ===========================================================
            # 🧭 Gráfico — Top 10 Ocupações de TI (RAIS)
            # ===========================================================
            
            if "TITULO" in df_r_ti.columns:
                
                st.subheader("🧭 Top 10 Ocupações de TI (RAIS 2023)")
                
                top10_ocup = df_r_ti["TITULO"].value_counts().head(10).reset_index()
                top10_ocup.columns = ["Ocupação", "Empregos"]

                if not top10_ocup.empty:
                    fig_ocup = px.bar(
                        top10_ocup.sort_values("Empregos"),
                        x="Empregos",
                        y="Ocupação",
                        orientation="h",
                        color="Empregos",
                        color_continuous_scale="Tealgrn",
                        text="Empregos",
                        title="🧭 Top 10 Ocupações de TI (RAIS 2023)"
                    )

                    fig_ocup.update_traces(
                        textposition="outside",
                        marker_line_color="#117a65",
                        marker_line_width=1.2,
                        opacity=0.9
                    )

                    fig_ocup.update_layout(
                        height=500,
                        margin=dict(l=120, r=40, t=70, b=50),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Segoe UI, sans-serif", size=13, color="#000"),
                        xaxis_title="Número de Empregos",
                        yaxis_title="Ocupação",
                        coloraxis_showscale=False,
                    )

                    st.plotly_chart(fig_ocup, use_container_width=True)
                    
                    # Download REMOVIDO

                else:
                    st.info("Nenhuma ocupação de TI encontrada no conjunto RAIS.")
            else:
                st.info("O dicionário CBO é necessário para este gráfico e não foi carregado corretamente.")
            
        else:
            st.warning("Não foi possível carregar o conjunto de dados RAIS.")


# ===========================================================
# Rodapé
# ===========================================================
st.markdown(f"""
---
💡 **Notas**
- Campo “Tipo De Vaga” normalizado automaticamente (aceita *tipo_vaga*, *Tipo de vaga*, etc.)  
- Filtros dinâmicos de *Horário* e *Salário* mantidos  
- Faixa salarial fixa: 0 – 12 000  
- Paginação de vagas = 10 por página  
- PyArrow ativo? → **{USE_PYARROW}**
""")