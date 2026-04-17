import joblib
import pandas as pd

# Load model
model = joblib.load('models/churn_xgboost.pkl')

class User:
    def __init__(self, name, idade, nivel_tecnologia, deficiencia_visual,
                 usa_leitor_tela, qtd_erros, tempo_total, churn):

        self.name = name
        self.idade = idade
        self.nivel_tecnologia = nivel_tecnologia
        self.deficiencia_visual = deficiencia_visual
        self.usa_leitor_tela = usa_leitor_tela
        self.qtd_erros = qtd_erros
        self.tempo_total = tempo_total
        self.churn = churn

    def calcular_dificuldade(self):
        score = self.qtd_erros * 10 + self.tempo_total / 10
        if self.deficiencia_visual and not self.usa_leitor_tela:
            score += 20
        return min(round(score, 2), 100)

    def calcular_acessibilidade(self):
        if self.deficiencia_visual and not self.usa_leitor_tela:
            return 'Alto'
        elif self.deficiencia_visual and self.usa_leitor_tela:
            return 'Médio'
        elif self.nivel_tecnologia == 'Baixo':
            return 'Baixo'
        else:
            return 'Nenhum'

    def prever_churn(self, user_id=0):
        pontuacao = self.calcular_dificuldade()
        acessibilidade = self.calcular_acessibilidade()

        entrada = pd.DataFrame([{
            'id_usuario': user_id,
            'pontuacao_dificuldade': pontuacao,
            'desafios_acessibilidade_Baixo': 1 if acessibilidade == 'Baixo' else 0,
            'desafios_acessibilidade_Médio': 1 if acessibilidade == 'Médio' else 0,
            'desafios_acessibilidade_Nenhum': 1 if acessibilidade == 'Nenhum' else 0,
        }])

        predicao = model.predict(entrada)[0]
        probabilidade = model.predict_proba(entrada)[0][1]
        return predicao, round(probabilidade * 100, 1)

    def resumo(self, user_id=0):
        predicao, prob = self.prever_churn(user_id)
        print(f"\nUsuário: {self.name}")
        print(f"Idade: {self.idade}")
        print(f"Nível tecnológico: {self.nivel_tecnologia}")
        print(f"Deficiência visual: {self.deficiencia_visual}")
        print(f"Usa leitor de tela: {self.usa_leitor_tela}")
        print(f"Erros no formulário: {self.qtd_erros}")
        print(f"Tempo gasto (s): {self.tempo_total}")
        print(f"Churn real: {'Sim' if self.churn else 'Não'}")
        print(f"Pontuação de dificuldade: {self.calcular_dificuldade()}")
        print(f"Desafios de acessibilidade: {self.calcular_acessibilidade()}")
        print(f"Previsão do modelo: {'Churn' if predicao == 1 else 'Sem churn'} ({prob}% de probabilidade)")


# --- Usuário 1 (baixo risco) ---
user1 = User(
    name="Alice",
    idade=30,
    nivel_tecnologia="Alto",
    deficiencia_visual=False,
    usa_leitor_tela=False,
    qtd_erros=1,
    tempo_total=120,
    churn=0
)

# --- Usuário 2 (alto risco) ---
user2 = User(
    name="Bob",
    idade=45,
    nivel_tecnologia="Baixo",
    deficiencia_visual=True,
    usa_leitor_tela=False,
    qtd_erros=6,
    tempo_total=450,
    churn=1
)

# Mostrar resultados com previsão
user1.resumo(user_id=1)
user2.resumo(user_id=2)
