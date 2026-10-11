"""
Aplicativo Streamlit — Previsão de preços de imóveis (Califórnia, censo de 1990).

O usuário escolhe um condado, a idade do imóvel e a renda mediana da região.
As demais features do modelo vêm das medianas do condado escolhido
(arquivo dados/gdf_counties.parquet, gerado no notebook 03-jv-geo.ipynb).

Executar localmente:
    streamlit run home.py
"""

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st
from joblib import load

from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

# ---------------------------------------------------------------------------
# Configuração da página (precisa ser o primeiro comando Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Previsão de preços de imóveis",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constantes do projeto
# ---------------------------------------------------------------------------

# Mesma ordem de colunas usada no treino (04-jv-modelos.ipynb).
# Se o modelo tiver o atributo feature_names_in_, ele tem prioridade.
FEATURES_MODELO = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    "median_income_cat",
    "rooms_per_household",
    "population_per_household",
    "bedrooms_per_room",
]

# Mesmos cortes do pd.cut do notebook 01-jv-eda.ipynb (intervalos fechados à direita).
BINS_RENDA = [0, 1.5, 3, 4.5, 6, np.inf]
LABELS_RENDA = [1, 2, 3, 4, 5]

# RMSE do melhor modelo na validação cruzada (grid_search.best_score_ no notebook 04).
RMSE_VALIDACAO = 50_441

# Teto do target após a limpeza dos dados (median_house_value < quantil 0,99).
TETO_TARGET = 500_000

# Rampa de cores sequencial (viridis) para o mapa coroplético.
RAMPA_CORES = [
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
]

TRADUCAO_PROXIMIDADE = {
    "<1H OCEAN": "A menos de 1h do oceano",
    "INLAND": "Interior",
    "NEAR OCEAN": "Perto do oceano",
    "NEAR BAY": "Perto da baía",
    "ISLAND": "Ilha",
}

NOMES_AMIGAVEIS = {
    "longitude": "Longitude",
    "latitude": "Latitude",
    "housing_median_age": "Idade do imóvel (anos)",
    "total_rooms": "Total de cômodos",
    "total_bedrooms": "Total de quartos",
    "population": "População",
    "households": "Domicílios",
    "median_income": "Renda mediana (x US$ 10 mil)",
    "ocean_proximity": "Proximidade do oceano",
    "median_income_cat": "Categoria de renda (1 a 5)",
    "rooms_per_household": "Cômodos por domicílio",
    "population_per_household": "Pessoas por domicílio",
    "bedrooms_per_room": "Quartos por cômodo",
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def formatar_moeda(valor: float) -> str:
    """Formata um valor em dólares no padrão brasileiro: US$ 1.234.567."""
    return "US$ " + f"{valor:,.0f}".replace(",", ".")


def cor_por_valor(valor: float, minimo: float, maximo: float) -> list[int]:
    """Interpola a rampa de cores de acordo com a posição do valor entre mínimo e máximo."""
    if maximo == minimo:
        t = 0.0
    else:
        t = float(np.clip((valor - minimo) / (maximo - minimo), 0, 1))
    posicao = t * (len(RAMPA_CORES) - 1)
    i = min(int(posicao), len(RAMPA_CORES) - 2)
    frac = posicao - i
    c1, c2 = RAMPA_CORES[i], RAMPA_CORES[i + 1]
    return [int(a + (b - a) * frac) for a, b in zip(c1, c2)] + [170]


def categoria_renda(renda: float) -> int:
    """Replica exatamente a criação de median_income_cat feita no treino (pd.cut)."""
    return int(pd.cut([renda], bins=BINS_RENDA, labels=LABELS_RENDA)[0])


# ---------------------------------------------------------------------------
# Carregamento de dados e modelo (com cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dados...")
def carregar_dados_limpos() -> pd.DataFrame:
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data(show_spinner="Carregando condados...")
def carregar_condados() -> pd.DataFrame:
    """Uma linha por condado, com as medianas usadas como entrada do modelo."""
    gdf = gpd.read_parquet(DADOS_GEO_MEDIAN)
    colunas = ["name", "median_house_value"] + FEATURES_MODELO
    df = pd.DataFrame(gdf[colunas]).copy()
    df["ocean_proximity"] = df["ocean_proximity"].astype(str)
    return df.sort_values("name").reset_index(drop=True)


@st.cache_data(show_spinner="Preparando o mapa...")
def carregar_poligonos() -> pd.DataFrame:
    """
    Converte as geometrias para o formato aceito pelo PolygonLayer do pydeck:
    uma linha por polígono, com a lista de anéis (contorno externo + buracos).
    """
    gdf = gpd.read_parquet(DADOS_GEO_MEDIAN)[["name", "median_house_value", "median_income", "geometry"]]
    gdf = gdf.explode(ignore_index=True)  # MultiPolygon -> vários Polygon

    def corrigir_e_orientar(geometria):
        if not geometria.is_valid:
            geometria = geometria.buffer(0)
        if isinstance(geometria, shapely.geometry.Polygon):
            geometria = shapely.geometry.polygon.orient(geometria, sign=1.0)
        return geometria

    def aneis(geometria):
        if isinstance(geometria, shapely.geometry.MultiPolygon):
            # buffer(0) pode devolver MultiPolygon: fica só com a maior parte
            geometria = max(geometria.geoms, key=lambda g: g.area)
        externo = [[x, y] for x, y in geometria.exterior.coords]
        internos = [[[x, y] for x, y in anel.coords] for anel in geometria.interiors]
        return [externo] + internos

    gdf["geometry"] = gdf["geometry"].apply(corrigir_e_orientar)

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["poligono"] = gdf["geometry"].apply(aneis)

    minimo, maximo = df["median_house_value"].min(), df["median_house_value"].max()
    df["cor"] = df["median_house_value"].apply(cor_por_valor, args=(minimo, maximo))
    df["valor_fmt"] = df["median_house_value"].apply(formatar_moeda)
    df["renda_fmt"] = (df["median_income"] * 10_000).apply(formatar_moeda)
    return df


@st.cache_resource(show_spinner="Carregando modelo...")
def carregar_modelo():
    return load(MODELO_FINAL)


try:
    df = carregar_dados_limpos()
    df_condados = carregar_condados()
    df_poligonos = carregar_poligonos()
    modelo = carregar_modelo()
except FileNotFoundError as erro:
    st.error(f"Arquivo não encontrado: `{erro.filename}`. Confira as pastas `dados/` e `modelos/`.")
    st.stop()

FEATURES = list(getattr(modelo, "feature_names_in_", FEATURES_MODELO))


def montar_entrada(linha_condado: pd.Series, idade: int, renda: float) -> pd.DataFrame:
    """Monta o DataFrame de 1 linha no formato que o pipeline espera."""
    entrada = linha_condado[FEATURES_MODELO].to_frame().T.copy()
    entrada["housing_median_age"] = idade
    entrada["median_income"] = renda
    entrada["median_income_cat"] = categoria_renda(renda)
    entrada = entrada[FEATURES]
    colunas_numericas = [c for c in FEATURES if c != "ocean_proximity"]
    entrada[colunas_numericas] = entrada[colunas_numericas].astype(float)
    return entrada


def prever(entrada: pd.DataFrame) -> np.ndarray:
    """O modelo foi treinado com y em DataFrame, então devolve 2D: achatamos para 1D."""
    return np.clip(np.ravel(modelo.predict(entrada)), 0, None)


# ---------------------------------------------------------------------------
# Barra lateral: entradas do usuário
# ---------------------------------------------------------------------------
renda_min_mil = int(np.ceil(df["median_income"].min() * 10))
renda_max_mil = int(np.floor(df["median_income"].max() * 10))
renda_padrao_mil = int(round(df["median_income"].median() * 10))
idade_max = int(df["housing_median_age"].max())

with st.sidebar:
    st.header("Parâmetros")

    condados = df_condados["name"].tolist()
    condado = st.selectbox(
        "Condado",
        condados,
        index=condados.index("Los Angeles") if "Los Angeles" in condados else 0,
        help="As demais características (cômodos, população, localização...) "
        "vêm das medianas deste condado.",
    )

    idade = st.slider(
        "Idade do imóvel (anos)",
        min_value=1,
        max_value=idade_max,
        value=10,
        help=f"Faixa dos dados de treino: 1 a {idade_max} anos.",
    )

    renda_mil = st.slider(
        "Renda mediana anual da região (US$ mil)",
        min_value=renda_min_mil,
        max_value=renda_max_mil,
        value=renda_padrao_mil,
        step=1,
        help="Renda mediana dos moradores do grupo de blocos censitários. "
        "Ex.: 45 = US$ 45 mil por ano.",
    )

    st.divider()
    st.caption(
        "Dados: censo dos EUA de 1990 (Kaggle — California Housing Prices). "
        "Sugestões: [GitHub VieiraJulio](https://github.com/VieiraJulio)."
    )

# ---------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------
renda = renda_mil / 10  # o modelo usa renda em dezenas de milhares de dólares
linha_condado = df_condados.loc[df_condados["name"] == condado].iloc[0]
entrada = montar_entrada(linha_condado, idade, renda)
preco = float(prever(entrada)[0])

mediana_condado = float(linha_condado["median_house_value"])
limite_inferior = max(preco - RMSE_VALIDACAO, 0)
limite_superior = preco + RMSE_VALIDACAO

# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------
st.title("🏠 Previsão de preços de imóveis")
st.caption(
    "Estimativa do valor mediano de imóveis na Califórnia com um modelo Ridge "
    "(features polinomiais + transformação do target). Ajuste os parâmetros na barra lateral."
)

aba_previsao, aba_modelo = st.tabs(["📈 Previsão", "🧠 Sobre o modelo"])

with aba_previsao:
    m1, m2, m3 = st.columns(3)
    m1.metric("Preço previsto", formatar_moeda(preco))
    m2.metric(
        "Faixa provável (± RMSE)",
        f"{formatar_moeda(limite_inferior)} – {formatar_moeda(limite_superior)}".replace("US$ ", ""),
        help=f"O erro típico do modelo na validação cruzada (RMSE) é de cerca de "
        f"{formatar_moeda(RMSE_VALIDACAO)}. Valores em US$.",
    )
    diferenca = (preco / mediana_condado - 1) * 100 if mediana_condado else 0
    m3.metric(
        f"Mediana real em {condado}",
        formatar_moeda(mediana_condado),
        delta=f"{diferenca:+.1f}% previsão vs. mediana",
        delta_color="off",
        help="Valor mediano observado nos dados do condado (todas as idades e rendas).",
    )

    if preco >= TETO_TARGET * 0.95:
        st.warning(
            f"A previsão está perto do teto dos dados de treino ({formatar_moeda(TETO_TARGET)}). "
            "Imóveis mais caros que isso não foram vistos pelo modelo."
        )

    col_mapa, col_info = st.columns([3, 2], gap="large")

    with col_mapa:
        st.subheader("Mapa dos condados")

        camada_condados = pdk.Layer(
            "PolygonLayer",
            data=df_poligonos,
            get_polygon="poligono",
            get_fill_color="cor",
            get_line_color=[255, 255, 255, 120],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )

        camada_destaque = pdk.Layer(
            "PolygonLayer",
            data=df_poligonos.loc[df_poligonos["name"] == condado],
            get_polygon="poligono",
            get_fill_color=[255, 75, 75, 60],
            get_line_color=[255, 75, 75, 255],
            line_width_min_pixels=3,
            pickable=True,
        )

        visao = pdk.ViewState(
            latitude=float(linha_condado["latitude"]),
            longitude=float(linha_condado["longitude"]),
            zoom=5.5,
            min_zoom=4,
            max_zoom=12,
        )

        tooltip = {
            "html": "<b>{name}</b><br/>Valor mediano: {valor_fmt}<br/>Renda mediana: {renda_fmt}",
            "style": {"backgroundColor": "#262730", "color": "white", "fontSize": "12px"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[camada_condados, camada_destaque],
                initial_view_state=visao,
                map_provider="carto",
                map_style=pdk.map_styles.CARTO_DARK,  # não exige token do Mapbox
                tooltip=tooltip,
            ),
            use_container_width=True,
        )

        v_min = formatar_moeda(df_poligonos["median_house_value"].min())
        v_max = formatar_moeda(df_poligonos["median_house_value"].max())
        gradiente = ", ".join(f"rgb{c}" for c in RAMPA_CORES)
        st.markdown(
            f"""
            <div style="font-size:0.8rem;opacity:0.8">Valor mediano dos imóveis por condado
            (contorno vermelho = condado selecionado)</div>
            <div style="height:10px;border-radius:4px;background:linear-gradient(90deg,{gradiente});"></div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;opacity:0.8">
                <span>{v_min}</span><span>{v_max}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_info:
        st.subheader("Como a renda afeta o preço")
        st.caption(f"{condado}, imóvel com {idade} anos — variando só a renda.")

        rendas_mil = np.arange(renda_min_mil, renda_max_mil + 1, 1)
        entradas = pd.concat(
            [montar_entrada(linha_condado, idade, r / 10) for r in rendas_mil],
            ignore_index=True,
        )
        df_curva = pd.DataFrame({"renda_mil": rendas_mil, "preco": prever(entradas)})

        base = alt.Chart(df_curva).encode(
            x=alt.X("renda_mil:Q", title="Renda mediana (US$ mil)"),
            y=alt.Y("preco:Q", title="Preço previsto (US$)", axis=alt.Axis(format="~s")),
        )
        linha = base.mark_line(strokeWidth=2.5).encode(
            tooltip=[
                alt.Tooltip("renda_mil:Q", title="Renda (US$ mil)"),
                alt.Tooltip("preco:Q", title="Preço (US$)", format=",.0f"),
            ]
        )
        ponto = (
            alt.Chart(pd.DataFrame({"renda_mil": [renda_mil], "preco": [preco]}))
            .mark_point(size=120, filled=True, color="#FF4B4B")
            .encode(x="renda_mil:Q", y="preco:Q")
        )
        st.altair_chart((linha + ponto).properties(height=280), use_container_width=True)

        st.markdown(
            f"""
            **Perfil do condado (medianas)**
            - Proximidade do oceano: **{TRADUCAO_PROXIMIDADE.get(linha_condado['ocean_proximity'], linha_condado['ocean_proximity'])}**
            - Cômodos por domicílio: **{linha_condado['rooms_per_household']:.1f}**
            - Pessoas por domicílio: **{linha_condado['population_per_household']:.1f}**
            - Categoria de renda usada: **{categoria_renda(renda)}** (de 1 a 5)
            """
        )

    with st.expander("🔎 Ver os dados enviados ao modelo"):
        tabela = entrada.T.rename(index=NOMES_AMIGAVEIS)
        tabela.columns = ["Valor"]
        st.dataframe(tabela.astype(str), use_container_width=True)

with aba_modelo:
    n_registros = f"{len(df):,}".replace(",", ".")
    st.markdown(
        f"""
        ### Como funciona

        1. **Dados**: {n_registros} grupos de blocos censitários da Califórnia (censo de 1990),
           após remover outliers acima do quantil 0,99 e a categoria `ISLAND`.
        2. **Pré-processamento**: `OneHotEncoder` em `ocean_proximity`, `OrdinalEncoder` na
           categoria de renda e `RobustScaler` + `PolynomialFeatures` nas variáveis numéricas.
        3. **Modelo**: regressão `Ridge` com o target transformado por `QuantileTransformer`
           (distribuição normal); grau polinomial e `alpha` escolhidos via `GridSearchCV`.
        4. **Desempenho**: RMSE de aproximadamente **{formatar_moeda(RMSE_VALIDACAO)}**
           na validação cruzada (5 folds).

        ### Limitações importantes

        - Os preços são de **1990** — não representam o mercado atual.
        - O modelo prevê o **valor mediano de uma região**, não o preço de um imóvel específico.
        - No app, características como cômodos e população são **medianas do condado**,
          enquanto o modelo foi treinado com grupos de blocos (áreas menores).
        - O target foi limitado a {formatar_moeda(TETO_TARGET)}: valores acima disso
          não são previstos com confiança.
        - Por usar termos polinomiais, a curva pode se comportar de forma estranha
          nos extremos de renda.
        """
    )
