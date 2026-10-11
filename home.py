"""
Aplicativo Streamlit — Estimativa de preço de imóveis (Califórnia, censo de 1990).

O usuário escolhe um condado, a idade do imóvel e a renda mediana da região.
As demais features do modelo vêm das medianas do condado escolhido
(arquivo dados/gdf_counties.parquet, gerado no notebook 03-jv-geo.ipynb).

Executar localmente:
    streamlit run home.py
"""

import html
import textwrap

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
    page_title="Estimativa de preço de imóveis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
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

# Paleta: tinta (azul Pacífico escuro), acento, papoula (flor do estado) e neutros.
COR_TINTA = "#0B2B40"
COR_ACENTO = "#1B7F8C"
COR_PAPOULA = "#E8832A"
COR_LINHA = "#D5DFE4"
COR_TEXTO_SUAVE = "#5A7280"

# Rampa sequencial única, usada no mapa e na escala de preços (claro = barato, escuro = caro).
RAMPA_CORES = [
    (214, 230, 235),
    (153, 196, 208),
    (78, 150, 170),
    (25, 100, 125),
    (11, 52, 82),
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
# Estilo (CSS)
# ---------------------------------------------------------------------------
ESTILO = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp, [data-testid="stMarkdownContainer"], label, button, input,
[data-testid="stSlider"] div {{
    font-family: 'Public Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.stApp {{ font-variant-numeric: tabular-nums; }}

/* Esconde elementos de desenvolvimento do Streamlit */
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {{ display: none !important; }}

.block-container {{
    max-width: 1240px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}}

/* Faixa de título */
.topo {{
    background: {COR_TINTA};
    border-left: 6px solid {COR_PAPOULA};
    border-radius: 4px 14px 14px 4px;
    padding: 26px 32px 24px 30px;
    margin-bottom: 1.1rem;
}}
.topo h1 {{
    color: #fff;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin: 0;
    padding: 0;
}}
.topo p {{
    color: #B9CCD6;
    font-size: 1rem;
    line-height: 1.5;
    margin: 8px 0 0 0;
    max-width: 62ch;
}}

/* Painel de controles */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #fff;
    border-color: {COR_LINHA};
    border-radius: 8px;
}}

/* Indicadores */
.kpis {{
    display: grid;
    grid-template-columns: 1.25fr 1.25fr 1fr;
    gap: 14px;
    margin: 0.4rem 0 1rem 0;
}}
.kpi {{
    background: #fff;
    border: 1px solid {COR_LINHA};
    border-radius: 8px;
    padding: 18px 20px 16px 20px;
}}
.kpi.principal {{
    background: {COR_TINTA};
    border-color: {COR_TINTA};
}}
.kpi .rotulo {{
    color: {COR_TEXTO_SUAVE};
    font-size: 0.9rem;
    font-weight: 500;
}}
.kpi .valor {{
    color: {COR_TINTA};
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    line-height: 1.25;
    margin-top: 4px;
}}
.kpi.principal .rotulo {{ color: #9FB8C6; }}
.kpi.principal .valor {{ color: #fff; font-size: 2.35rem; }}
.kpi .nota {{
    color: {COR_TEXTO_SUAVE};
    font-size: 0.85rem;
    line-height: 1.45;
    margin-top: 6px;
}}
.kpi.principal .nota {{ color: #9FB8C6; }}

/* Escala de preços */
.painel {{
    background: #fff;
    border: 1px solid {COR_LINHA};
    border-radius: 8px;
    padding: 18px 22px 16px 22px;
    margin-bottom: 1rem;
}}
.painel h3 {{
    color: {COR_TINTA};
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0;
    padding: 0;
}}
.painel .sub {{
    color: {COR_TEXTO_SUAVE};
    font-size: 0.88rem;
    margin: 2px 0 0 0;
}}
.escala {{ margin: 6px 10px 0 10px; }}
.escala .acima, .escala .abaixo {{ position: relative; height: 46px; }}
.escala .abaixo {{ height: 50px; }}
.escala .bandeira {{
    position: absolute;
    transform: translateX(-50%);
    text-align: center;
    font-size: 0.8rem;
    line-height: 1.3;
    color: {COR_TEXTO_SUAVE};
    white-space: nowrap;
}}
.escala .acima .bandeira {{ bottom: 4px; }}
.escala .abaixo .bandeira {{ top: 8px; }}
.escala .bandeira b {{ display: block; font-size: 0.95rem; font-weight: 700; color: {COR_TINTA}; }}
.escala .abaixo .bandeira b {{ color: #B8601A; }}
.escala .trilho-area {{ position: relative; height: 40px; }}
.escala .faixa {{
    position: absolute;
    top: 0;
    height: 40px;
    background: rgba(11, 43, 64, 0.10);
    border-left: 2px solid rgba(11, 43, 64, 0.35);
    border-right: 2px solid rgba(11, 43, 64, 0.35);
    box-sizing: border-box;
}}
.escala .trilho {{
    position: absolute;
    top: 14px;
    left: 0;
    right: 0;
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg, {", ".join(f"rgb{c}" for c in RAMPA_CORES)});
}}
.escala .marcador {{
    position: absolute;
    top: 0;
    width: 5px;
    height: 40px;
    margin-left: -2.5px;
    border-radius: 3px;
    background: {COR_TINTA};
    box-shadow: 0 0 0 2px #fff;
}}
.escala .mediana {{
    position: absolute;
    top: 33px;
    margin-left: -7px;
    width: 0;
    height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-bottom: 11px solid {COR_PAPOULA};
}}
.escala .marcas {{
    display: flex;
    justify-content: space-between;
    margin-top: 14px;
    color: {COR_TEXTO_SUAVE};
    font-size: 0.78rem;
}}
.legenda-faixa {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    color: {COR_TEXTO_SUAVE};
    font-size: 0.82rem;
    margin-top: 2px;
}}
.legenda-faixa i {{ display: inline-block; vertical-align: -2px; margin-right: 6px; }}

/* Perfil do condado */
.perfil {{ margin: 0; padding: 0; }}
.perfil div {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 9px 0;
    border-bottom: 1px solid #E6EDF0;
    font-size: 0.92rem;
}}
.perfil div:last-child {{ border-bottom: 0; }}
.perfil dt {{ color: {COR_TEXTO_SUAVE}; font-weight: 400; }}
.perfil dd {{ color: {COR_TINTA}; font-weight: 600; margin: 0; text-align: right; }}

/* Legenda do mapa */
.mapa-legenda .barra {{
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, {", ".join(f"rgb{c}" for c in RAMPA_CORES)});
}}
.mapa-legenda .extremos {{
    display: flex;
    justify-content: space-between;
    color: {COR_TEXTO_SUAVE};
    font-size: 0.8rem;
    margin-top: 4px;
}}
.mapa-legenda .titulo {{ color: {COR_TEXTO_SUAVE}; font-size: 0.82rem; margin-bottom: 4px; }}

.rodape {{
    color: {COR_TEXTO_SUAVE};
    font-size: 0.82rem;
    margin-top: 1.6rem;
    padding-top: 12px;
    border-top: 1px solid {COR_LINHA};
}}

@media (max-width: 860px) {{
    .kpis {{ grid-template-columns: 1fr; }}
    .topo h1 {{ font-size: 1.5rem; }}
    .kpi.principal .valor {{ font-size: 2rem; }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def html_bloco(texto: str) -> str:
    """Remove a indentação das linhas para o Markdown não tratar o HTML como bloco de código."""
    return "\n".join(linha.strip() for linha in textwrap.dedent(texto).splitlines() if linha.strip())


def formatar_moeda(valor: float) -> str:
    """Formata um valor em dólares no padrão brasileiro: US$ 1.234.567."""
    return "US$ " + f"{valor:,.0f}".replace(",", ".")


def formatar_percentual(valor: float) -> str:
    """Formata percentual com vírgula decimal: 11,1%."""
    return f"{valor:.1f}".replace(".", ",") + "%"


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
    return [int(a + (b - a) * frac) for a, b in zip(c1, c2)] + [215]


def categoria_renda(renda: float) -> int:
    """Replica exatamente a criação de median_income_cat feita no treino (pd.cut)."""
    return int(pd.cut([renda], bins=BINS_RENDA, labels=LABELS_RENDA)[0])


def posicao_escala(valor: float) -> float:
    """Posição (0 a 100) de um valor na escala de 0 até o teto do target."""
    return float(np.clip(valor / TETO_TARGET * 100, 0, 100))


def posicao_rotulo(posicao: float) -> float:
    """Evita que o rótulo de uma bandeira saia da área da escala."""
    return float(np.clip(posicao, 10, 90))


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
# Cabeçalho e controles
# ---------------------------------------------------------------------------
st.markdown(ESTILO, unsafe_allow_html=True)

st.markdown(
    html_bloco(
        """
        <div class="topo">
        <h1>Estimativa de preço de imóveis</h1>
        <p>Valor mediano das casas na Califórnia, calculado a partir do condado,
        da idade do imóvel e da renda da região.</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

renda_min_mil = int(np.ceil(df["median_income"].min() * 10))
renda_max_mil = int(np.floor(df["median_income"].max() * 10))
renda_padrao_mil = int(round(df["median_income"].median() * 10))
idade_max = int(df["housing_median_age"].max())
condados = df_condados["name"].tolist()

with st.container(border=True):
    col_condado, col_idade, col_renda = st.columns([1.1, 1, 1.4], gap="large")

    with col_condado:
        condado = st.selectbox(
            "Condado",
            condados,
            index=condados.index("Los Angeles") if "Los Angeles" in condados else 0,
            help="As demais características (cômodos, população, localização) "
            "vêm das medianas deste condado.",
        )
    with col_idade:
        idade = st.slider(
            "Idade do imóvel (anos)",
            min_value=1,
            max_value=idade_max,
            value=10,
            help=f"Faixa dos dados de treino: 1 a {idade_max} anos.",
        )
    with col_renda:
        renda_mil = st.slider(
            "Renda mediana anual da região (US$ mil)",
            min_value=renda_min_mil,
            max_value=renda_max_mil,
            value=renda_padrao_mil,
            step=1,
            help="Renda mediana dos moradores do grupo de blocos censitários. "
            "Exemplo: 45 significa US$ 45 mil por ano.",
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
nome_condado = html.escape(condado)

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
aba_previsao, aba_modelo = st.tabs(["Previsão", "Sobre o modelo"])

with aba_previsao:
    # --- Indicadores -------------------------------------------------------
    diferenca = (preco / mediana_condado - 1) * 100 if mediana_condado else 0
    if abs(diferenca) < 0.05:
        texto_diferenca = "Previsão igual à mediana do condado."
    else:
        sentido = "acima" if diferenca > 0 else "abaixo"
        texto_diferenca = f"A previsão está {formatar_percentual(abs(diferenca))} {sentido} da mediana."

    st.markdown(
        html_bloco(
            f"""
            <div class="kpis">
            <div class="kpi principal">
            <div class="rotulo">Preço previsto</div>
            <div class="valor">{formatar_moeda(preco)}</div>
            <div class="nota">Valor mediano estimado para esta combinação de condado, idade e renda.</div>
            </div>
            <div class="kpi">
            <div class="rotulo">Faixa provável</div>
            <div class="valor">{formatar_moeda(limite_inferior).replace("US$ ", "")} a {formatar_moeda(limite_superior).replace("US$ ", "")}</div>
            <div class="nota">Previsão mais ou menos o erro típico do modelo (RMSE de {formatar_moeda(RMSE_VALIDACAO)}), em US$.</div>
            </div>
            <div class="kpi">
            <div class="rotulo">Mediana real em {nome_condado}</div>
            <div class="valor">{formatar_moeda(mediana_condado)}</div>
            <div class="nota">{texto_diferenca}</div>
            </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if preco >= TETO_TARGET * 0.95:
        st.warning(
            f"A previsão está perto do teto dos dados de treino ({formatar_moeda(TETO_TARGET)}). "
            "Imóveis mais caros que isso não foram vistos pelo modelo."
        )

    # --- Escala de preços --------------------------------------------------
    p_estimativa = posicao_escala(preco)
    p_inferior = posicao_escala(limite_inferior)
    p_superior = posicao_escala(limite_superior)
    p_mediana = posicao_escala(mediana_condado)
    marcas = "".join(
        f"<span>{'US$ 0' if v == 0 else f'{v // 1000} mil'}</span>" for v in range(0, TETO_TARGET + 1, 100_000)
    )

    st.markdown(
        html_bloco(
            f"""
            <div class="painel">
            <h3>Onde a estimativa cai na escala de preços</h3>
            <p class="sub">A escala usa as mesmas cores do mapa, de US$ 0 até o teto dos dados de treino.</p>
            <div class="escala">
            <div class="acima">
            <div class="bandeira" style="left:{posicao_rotulo(p_estimativa)}%">Previsão<b>{formatar_moeda(preco)}</b></div>
            </div>
            <div class="trilho-area">
            <div class="faixa" style="left:{p_inferior}%;width:{max(p_superior - p_inferior, 0.4)}%"></div>
            <div class="trilho"></div>
            <div class="marcador" style="left:{p_estimativa}%"></div>
            <div class="mediana" style="left:{p_mediana}%"></div>
            </div>
            <div class="marcas">{marcas}</div>
            <div class="abaixo">
            <div class="bandeira" style="left:{posicao_rotulo(p_mediana)}%">Mediana em {nome_condado}<b>{formatar_moeda(mediana_condado)}</b></div>
            </div>
            </div>
            <div class="legenda-faixa">
            <span><i style="width:16px;height:10px;background:rgba(11,43,64,.10);border-left:2px solid rgba(11,43,64,.35);border-right:2px solid rgba(11,43,64,.35)"></i>Faixa provável</span>
            </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # --- Mapa e gráfico ----------------------------------------------------
    col_mapa, col_info = st.columns([3, 2], gap="large")

    with col_mapa:
        st.markdown("##### Mapa dos condados")

        camada_condados = pdk.Layer(
            "PolygonLayer",
            data=df_poligonos,
            get_polygon="poligono",
            get_fill_color="cor",
            get_line_color=[255, 255, 255, 230],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )

        camada_destaque = pdk.Layer(
            "PolygonLayer",
            data=df_poligonos.loc[df_poligonos["name"] == condado],
            get_polygon="poligono",
            get_fill_color=[232, 131, 42, 35],
            get_line_color=[232, 131, 42, 255],
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
            "style": {"backgroundColor": COR_TINTA, "color": "white", "fontSize": "12px"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[camada_condados, camada_destaque],
                initial_view_state=visao,
                map_provider="carto",
                map_style=pdk.map_styles.CARTO_LIGHT,  # não exige token do Mapbox
                tooltip=tooltip,
            ),
            use_container_width=True,
        )

        v_min = formatar_moeda(df_poligonos["median_house_value"].min())
        v_max = formatar_moeda(df_poligonos["median_house_value"].max())
        st.markdown(
            html_bloco(
                f"""
                <div class="mapa-legenda">
                <div class="titulo">Valor mediano dos imóveis por condado. O contorno laranja marca o condado selecionado.</div>
                <div class="barra"></div>
                <div class="extremos"><span>{v_min}</span><span>{v_max}</span></div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown("##### Como a renda afeta o preço")
        st.caption(f"{condado}, imóvel com {idade} anos. Só a renda varia.")

        rendas_mil = np.arange(renda_min_mil, renda_max_mil + 1, 1)
        entradas = pd.concat(
            [montar_entrada(linha_condado, idade, r / 10) for r in rendas_mil],
            ignore_index=True,
        )
        df_curva = pd.DataFrame({"renda_mil": rendas_mil, "preco": prever(entradas)})

        y_min = float(min(df_curva["preco"].min(), mediana_condado, preco))
        y_max = float(max(df_curva["preco"].max(), mediana_condado, preco))
        margem = (y_max - y_min) * 0.12 or 1
        escala_y = alt.Scale(domain=[max(y_min - margem, 0), y_max + margem], nice=True)

        eixo_x = alt.X(
            "renda_mil:Q",
            title="Renda mediana anual da região (US$)",
            scale=alt.Scale(domain=[renda_min_mil, renda_max_mil], nice=False),
            axis=alt.Axis(labelExpr="datum.value + ' mil'", values=list(range(10, renda_max_mil + 1, 20))),
        )
        eixo_y = alt.Y(
            "preco:Q",
            title="Preço previsto",
            scale=escala_y,
            axis=alt.Axis(labelExpr="'US$ ' + format(datum.value / 1000, ',.0f') + ' mil'", tickCount=5),
        )

        base = alt.Chart(df_curva).encode(x=eixo_x, y=eixo_y)
        area = base.mark_area(color=COR_ACENTO, opacity=0.10)
        linha = base.mark_line(color=COR_ACENTO, strokeWidth=2.5).encode(
            tooltip=[
                alt.Tooltip("renda_mil:Q", title="Renda (US$ mil)"),
                alt.Tooltip("preco:Q", title="Preço (US$)", format=",.0f"),
            ]
        )
        regua_mediana = (
            alt.Chart(pd.DataFrame({"preco": [mediana_condado], "rotulo": ["Mediana real do condado"]}))
            .mark_rule(color=COR_PAPOULA, strokeDash=[5, 4], strokeWidth=1.5)
            .encode(y=alt.Y("preco:Q", scale=escala_y))
        )
        texto_mediana = (
            alt.Chart(pd.DataFrame({"x": [renda_max_mil], "preco": [mediana_condado], "t": ["Mediana real do condado"]}))
            .mark_text(align="right", dx=-4, dy=-7, color="#B8601A", fontSize=11, fontWeight=600)
            .encode(x=alt.X("x:Q"), y=alt.Y("preco:Q", scale=escala_y), text="t:N")
        )
        ponto = (
            alt.Chart(pd.DataFrame({"renda_mil": [renda_mil], "preco": [preco]}))
            .mark_point(size=150, filled=True, color=COR_TINTA, stroke="white", strokeWidth=2, opacity=1)
            .encode(x="renda_mil:Q", y=alt.Y("preco:Q", scale=escala_y))
        )

        grafico = (
            (area + linha + regua_mediana + texto_mediana + ponto)
            .properties(height=270)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=True,
                gridColor="#E6EDF0",
                domain=False,
                tickColor="#E6EDF0",
                labelColor=COR_TEXTO_SUAVE,
                titleColor=COR_TEXTO_SUAVE,
                labelFont="Public Sans, sans-serif",
                titleFont="Public Sans, sans-serif",
                titleFontWeight=500,
                labelFontSize=11,
                titleFontSize=12,
            )
        )
        st.altair_chart(grafico, use_container_width=True)

        proximidade = TRADUCAO_PROXIMIDADE.get(linha_condado["ocean_proximity"], linha_condado["ocean_proximity"])
        st.markdown("##### Perfil do condado")
        st.markdown(
            html_bloco(
                f"""
                <dl class="perfil">
                <div><dt>Proximidade do oceano</dt><dd>{html.escape(str(proximidade))}</dd></div>
                <div><dt>Cômodos por domicílio</dt><dd>{linha_condado["rooms_per_household"]:.1f}</dd></div>
                <div><dt>Pessoas por domicílio</dt><dd>{linha_condado["population_per_household"]:.1f}</dd></div>
                <div><dt>Categoria de renda usada (1 a 5)</dt><dd>{categoria_renda(renda)}</dd></div>
                </dl>
                """
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Ver os dados enviados ao modelo"):
        tabela = entrada.T.rename(index=NOMES_AMIGAVEIS)
        tabela.columns = ["Valor"]
        st.dataframe(tabela.astype(str), use_container_width=True)

with aba_modelo:
    n_registros = f"{len(df):,}".replace(",", ".")
    col_como, col_limites = st.columns(2, gap="large")

    with col_como:
        st.markdown(
            f"""
            ##### Como funciona

            1. **Dados:** {n_registros} grupos de blocos censitários da Califórnia (censo de 1990),
               depois de remover outliers acima do quantil 0,99 e a categoria `ISLAND`.
            2. **Pré-processamento:** `OneHotEncoder` em `ocean_proximity`, `OrdinalEncoder` na
               categoria de renda e `RobustScaler` com `PolynomialFeatures` nas variáveis numéricas.
            3. **Modelo:** regressão `Ridge` com o target transformado por `QuantileTransformer`
               (distribuição normal). O `GridSearchCV` escolheu polinômio de grau 3 e `alpha` = 20.
            4. **Desempenho:** RMSE de cerca de **{formatar_moeda(RMSE_VALIDACAO)}**
               na validação cruzada (5 folds).
            """
        )

    with col_limites:
        st.markdown(
            f"""
            ##### Limitações importantes

            - Os preços são de **1990** e não representam o mercado atual.
            - O modelo prevê o **valor mediano de uma região**, não o preço de um imóvel específico.
            - No app, cômodos e população são **medianas do condado**, mas o modelo foi
              treinado com grupos de blocos, que são áreas menores.
            - O target foi limitado a {formatar_moeda(TETO_TARGET)}: acima disso a previsão
              não é confiável.
            - Por usar termos polinomiais, a curva pode se comportar de forma estranha
              nos extremos de renda.
            """
        )

st.markdown(
    html_bloco(
        """
        <div class="rodape">
        Dados: censo dos EUA de 1990 (Kaggle, California Housing Prices).
        Projeto de Julio Vieira. Sugestões pelo GitHub:
        <a href="https://github.com/VieiraJulio" target="_blank" rel="noopener">VieiraJulio</a>.
        </div>
        """
    ),
    unsafe_allow_html=True,
)
