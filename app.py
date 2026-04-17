from flask import Flask, render_template, request, redirect, url_for, send_file
import csv
import os

app = Flask(__name__)

CSV_FILE = 'CSV.SelvadePrata'

FIELDNAMES = [
    'user_id', 'nome', 'idade', 'escolaridade', 'nivel_tecnologia',
    'dispositivo', 'experiencia_formularios', 'usa_leitor_tela',
    'deficiencia_visual', 'data_nascimento', 'renda_mensal',
    'ja_usou_plataforma', 'conseguiu_finalizar', 'churn'
]

def get_next_user_id():
    if not os.path.exists(CSV_FILE):
        return 1
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        if not rows:
            return 1
        return int(rows[-1]['user_id']) + 1

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/cadastro', methods=['POST'])
def cadastro():
    init_csv()
    user_id = get_next_user_id()

    dd = request.form.get('dd', '')
    mm = request.form.get('mm', '')
    aaaa = request.form.get('aaaa', '')
    data_nascimento = f"{dd}/{mm}/{aaaa}"

    dispositivo = request.form.get('aparelho', '')
    nivel_tecnologia = 'Alto' if request.form.get('pc') == 'sim' else 'Baixo'

    row = {
        'user_id': user_id,
        'nome': request.form.get('nome', ''),
        'idade': request.form.get('idade', ''),
        'escolaridade': request.form.get('esc', ''),
        'nivel_tecnologia': nivel_tecnologia,
        'dispositivo': dispositivo,
        'experiencia_formularios': '',
        'usa_leitor_tela': 'Sim' if request.form.get('leitor_tela') else 'Não',
        'deficiencia_visual': request.form.get('def', 'Não'),
        'data_nascimento': data_nascimento,
        'renda_mensal': request.form.get('renda', ''),
        'ja_usou_plataforma': 'Sim' if request.form.get('ja_usou') else 'Não',
        'conseguiu_finalizar': 'Sim',
        'churn': ''
    }

    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    return redirect(url_for('sucesso', nome=row['nome']))

@app.route('/sucesso')
def sucesso():
    nome = request.args.get('nome', 'usuário')
    total = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='', encoding='utf-8') as f:
            total = sum(1 for _ in csv.DictReader(f))
    return render_template('sucesso.html', nome=nome, total=total)

@app.route('/download')
def download():
    if not os.path.exists(CSV_FILE):
        return "Nenhum dado disponível ainda.", 404
    return send_file(CSV_FILE, as_attachment=True, download_name='CSV.SelvadePrata', mimetype='text/csv')

@app.route('/download/metricas')
def download_metricas():
    path = 'outputs/reports/model_metrics.txt'
    if not os.path.exists(path):
        return "Arquivo não encontrado.", 404
    return send_file(path, as_attachment=True, download_name='model_metrics.txt', mimetype='text/plain')

@app.route('/download/importancia')
def download_importancia():
    path = 'outputs/reports/feature_importance.csv'
    if not os.path.exists(path):
        return "Arquivo não encontrado.", 404
    return send_file(path, as_attachment=True, download_name='feature_importance.csv', mimetype='text/csv')

@app.route('/download/shap-global')
def download_shap_global():
    path = 'shap_global_summary.png'
    if not os.path.exists(path):
        return "Arquivo não encontrado.", 404
    return send_file(path, as_attachment=True, download_name='shap_global_summary.png', mimetype='image/png')

@app.route('/download/shap-local')
def download_shap_local():
    path = 'shap_local_waterfall.png'
    if not os.path.exists(path):
        return "Arquivo não encontrado.", 404
    return send_file(path, as_attachment=True, download_name='shap_local_waterfall.png', mimetype='image/png')

@app.route('/download/shap-insights')
def download_shap_insights():
    path = 'shap_insights.csv'
    if not os.path.exists(path):
        return "Arquivo não encontrado.", 404
    return send_file(path, as_attachment=True, download_name='shap_insights.csv', mimetype='text/csv')

@app.route('/download/shap-barras')
def download_shap_barras():
    path = 'shap_bar_importance.png'
    if not os.path.exists(path):
        return "Arquivo não encontrado.", 404
    return send_file(path, as_attachment=True, download_name='shap_bar_importance.png', mimetype='image/png')

if __name__ == '__main__':
    init_csv()
    app.run(host='0.0.0.0', port=5000)
