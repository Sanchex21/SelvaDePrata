import random
import pandas as pd

# Constants
NUM_USERS = 100
AGE_RANGE = (18, 70)
EDUCATION_LEVELS = ['Fundamental', 'Médio', 'Superior', 'Pós-graduação']
TECH_LEVELS = ['Baixo', 'Médio', 'Alto']
DEVICES = ['Desktop', 'Tablet', 'Mobile']

def simulate_user(user_id):
    # --- PERFIL ---
    idade = random.randint(*AGE_RANGE)
    escolaridade = random.choice(EDUCATION_LEVELS)
    nivel_tecnologia = random.choice(TECH_LEVELS)
    dispositivo = random.choice(DEVICES)
    experiencia_formularios = random.randint(0, 10)

    # --- ACESSIBILIDADE ---
    deficiencia_visual = random.random() < 0.15
    if deficiencia_visual:
        usa_leitor_tela = random.random() < 0.7
    else:
        usa_leitor_tela = random.random() < 0.02

    # --- COMPORTAMENTO ---
    qtd_erros = random.randint(1, 3)
    if nivel_tecnologia == 'Baixo':
        qtd_erros += random.randint(2, 4)

    qtd_voltas_campos = random.randint(1, 4)
    if usa_leitor_tela:
        qtd_voltas_campos += random.randint(5, 15)

    tempo_total_segundos = 60 + (qtd_erros * 20) + (qtd_voltas_campos * 10)
    if deficiencia_visual and not usa_leitor_tela:
        tempo_total_segundos += random.randint(100, 200)

    qtd_cliques_inuteis = random.randint(2, 8)
    if not usa_leitor_tela:
        qtd_cliques_inuteis += random.randint(5, 10)

    qtd_cliques_totais = 10 + qtd_cliques_inuteis + qtd_erros
    qtd_tentativas_reenvio = random.randint(1, 3)
    qtd_pausas_longas = random.randint(0, 5)
    qtd_campos_confusos = random.randint(2, 5)

    # --- SCORE ---
    score_dificuldade = (
        qtd_erros * 1.2 +
        qtd_voltas_campos * 0.5 +
        qtd_cliques_inuteis * 0.3 +
        tempo_total_segundos / 100
    )
    score_dificuldade = min(10, round(score_dificuldade, 2))

    # --- CHURN ---
    prob_churn = score_dificuldade / 10
    churn = 1 if random.random() < prob_churn else 0
    conseguiu_finalizar = 0 if churn == 1 else 1
    houve_abandono_parcial = 1 if churn == 1 else 0

    return {
        'user_id': user_id,
        'idade': idade,
        'escolaridade': escolaridade,
        'nivel_tecnologia': nivel_tecnologia,
        'dispositivo': dispositivo,
        'experiencia_formularios': experiencia_formularios,
        'usa_leitor_tela': int(usa_leitor_tela),
        'deficiencia_visual': int(deficiencia_visual),
        'tempo_total_segundos': tempo_total_segundos,
        'qtd_erros': qtd_erros,
        'qtd_tentativas_reenvio': qtd_tentativas_reenvio,
        'qtd_cliques_totais': qtd_cliques_totais,
        'qtd_cliques_inuteis': qtd_cliques_inuteis,
        'qtd_voltas_campos': qtd_voltas_campos,
        'qtd_pausas_longas': qtd_pausas_longas,
        'qtd_campos_confusos': qtd_campos_confusos,
        'houve_abandono_parcial': houve_abandono_parcial,
        'conseguiu_finalizar': conseguiu_finalizar,
        'score_dificuldade': score_dificuldade,
        'churn': churn
    }

users = [simulate_user(i) for i in range(1, NUM_USERS + 1)]
df = pd.DataFrame(users)
df.columns = [col.replace('_', ' ') for col in df.columns]
df.columns = [col.replace('qtd ', 'Quantidade de ') for col in df.columns]
df.to_csv('CSV.SelvadePrata', index=False, encoding='utf-8-sig')

print(f"Dataset gerado com {len(df)} usuários.")
print(df.head())
print("\nTaxa de churn:", round(df['churn'].mean() * 100, 2), "%")
