import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('CSV.SelvadePrata', encoding='utf-8-sig')
data.columns = [col.replace('_', ' ') for col in data.columns]

# --- TRATAR DADOS CATEGÓRICOS ---
data = pd.get_dummies(data, drop_first=True)
data.columns = [col.replace('_', ' ') for col in data.columns]

# Separar features e target
X = data.drop('churn', axis=1)
y = data['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar modelo
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)

# Para classificação binária (pega classe 1)
if isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[1]
    base_value = explainer.expected_value[1]
elif shap_values_raw.ndim == 3:
    shap_values = shap_values_raw[:, :, 1]
    base_value = explainer.expected_value[1]
else:
    shap_values = shap_values_raw
    base_value = explainer.expected_value if not hasattr(explainer.expected_value, '__len__') else explainer.expected_value[1]

# --- GRÁFICO GLOBAL (Summary Plot) ---
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, show=False, plot_size=None, color_bar=True)
plt.title('Importância Global das Variáveis (SHAP)', fontsize=15, fontweight='bold', pad=16)
plt.xlabel('Valor SHAP (impacto no churn)', fontsize=11)
plt.tight_layout()
plt.savefig('shap_global_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# --- GRÁFICO LOCAL (Waterfall) ---
fig = plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_test.iloc[0].values,
        feature_names=list(X_test.columns)
    ),
    show=False
)
plt.title('Explicação Individual — 1º Usuário', fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('shap_local_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

# --- GRÁFICO DE BARRAS (Top Features) ---
mean_shap = abs(shap_values).mean(axis=0)
insights = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_shap
}).sort_values(by='Importance', ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(insights)))
bars = ax.barh(insights['Feature'], insights['Importance'], color=colors, edgecolor='white', height=0.6)
ax.set_xlabel('Importância Média (|SHAP|)', fontsize=11)
ax.set_title('Top Variáveis por Importância SHAP', fontsize=14, fontweight='bold', pad=14)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, insights['Importance']):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, color='#333')
plt.tight_layout()
plt.savefig('shap_bar_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# --- INSIGHTS CSV ---
insights_full = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_shap
}).sort_values(by='Importance', ascending=False)
insights_full.to_csv('shap_insights.csv', index=False, encoding='utf-8-sig')

print("Análise SHAP concluída! Gráficos salvos.")
